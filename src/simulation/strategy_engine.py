"""Phase 2A/2E strategy engine for automated strategy search and recommendation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.simulation.strategy import (
    find_optimal_pit_lap,
    optimize_pit_window,
    predict_lap_time,
)

logger = logging.getLogger(__name__)


ONE_STOP_SELECTION_TOLERANCE_S = 1.0
TIMING_TRACE_NEAR_OPTIMAL_TOLERANCE_S = 1.25
LOW_MARGIN_SOFT_TIEBREAK_S = 1.0


@dataclass
class StrategyPlan:
    """Represents a complete pit strategy from current state to race end."""

    strategy_type: str
    current_compound: str
    current_tyre_life: int
    next_compound: str
    final_compound: Optional[str]
    pit_lap: int
    second_pit_lap: Optional[int]
    total_race_time: float
    feasible: bool
    feasibility_reason: str
    one_stop_estimate: Optional[float] = None
    explanation: str = ""
    model_info: Dict | None = None

    def __post_init__(self):
        if self.strategy_type == "one-stop":
            assert self.final_compound is None, "One-stop should have no final_compound"
            assert self.second_pit_lap is None, "One-stop should have no second_pit_lap"
        elif self.strategy_type == "two-stop":
            assert self.final_compound is not None, "Two-stop must have final_compound"
            assert self.second_pit_lap is not None, "Two-stop must have second_pit_lap"
        else:
            raise ValueError(f"Unknown strategy_type: {self.strategy_type}")

        if self.model_info is None:
            self.model_info = {}


def _is_valid_prediction(value: Optional[float]) -> bool:
    """Return True when a predicted lap time is finite and usable."""
    return value is not None and np.isfinite(value)


def _compound_model_info(degradation_models: Union[Dict, object], compound: str) -> Dict:
    """Return model metadata when the richer evaluation result is available."""
    if hasattr(degradation_models, "get_model_info"):
        return degradation_models.get_model_info(compound)
    return {"compound": compound, "model_type": "LEGACY", "samples": None}


def _compound_support_info(degradation_models: Union[Dict, object], compound: str) -> Dict:
    """Return richer support metadata when available."""
    if hasattr(degradation_models, "get_support_info"):
        return degradation_models.get_support_info(compound)
    return {}


def _simulate_stint_time(
    degradation_models: Union[Dict, object],
    compound: str,
    starting_tyre_life: int,
    stint_laps: int,
) -> Optional[float]:
    """Simulate a stint and return total time, or None if predictions fail."""
    total_time = 0.0
    for lap in range(stint_laps):
        tyre_age = starting_tyre_life + lap
        lap_time = predict_lap_time(degradation_models, compound, tyre_age)
        if not _is_valid_prediction(lap_time):
            return None
        total_time += float(lap_time)
    return total_time


def _plan_future_compounds(plan: StrategyPlan) -> List[str]:
    """Return the future compounds used after the current stint."""
    compounds = [plan.next_compound]
    if plan.final_compound is not None:
        compounds.append(plan.final_compound)
    return compounds


def _plan_uses_moderate_or_low_support_soft(
    degradation_models: Union[Dict, object],
    plan: StrategyPlan,
) -> bool:
    """Return True when a plan leans on SOFT with weaker-than-high support."""
    for compound in _plan_future_compounds(plan):
        if compound != "SOFT":
            continue
        support_info = _compound_support_info(degradation_models, compound)
        support_tier = support_info.get("support_tier")
        if support_tier is None:
            samples = _compound_model_info(degradation_models, compound).get("samples")
            if isinstance(samples, int) and samples < 250:
                support_tier = "Moderate"
        if support_tier in {"Low", "Moderate"}:
            return True
    return False


def _plan_has_only_high_support_futures(
    degradation_models: Union[Dict, object],
    plan: StrategyPlan,
) -> bool:
    """Return True when every future stint uses a high-support compound."""
    future_compounds = _plan_future_compounds(plan)
    if not future_compounds:
        return False
    for compound in future_compounds:
        support_info = _compound_support_info(degradation_models, compound)
        support_tier = support_info.get("support_tier")
        if support_tier is None:
            samples = _compound_model_info(degradation_models, compound).get("samples")
            support_tier = "High" if isinstance(samples, int) and samples >= 250 else "Moderate"
        if support_tier != "High":
            return False
    return True


def _apply_low_margin_soft_tie_break(
    ranked_plans: List[StrategyPlan],
    degradation_models: Union[Dict, object],
    max_gap_s: float = LOW_MARGIN_SOFT_TIEBREAK_S,
) -> List[StrategyPlan]:
    """Prefer the high-support non-SOFT equivalent when SOFT only wins narrowly.

    This is intentionally narrow:
    - only triggers when the nominal best plan uses SOFT under Moderate/Low support
    - only looks for an alternative with the same structure and pit timing
    - only flips when the non-SOFT alternative is within a 1.0s band

    The goal is to avoid overclaiming a weakly-supported SOFT edge when the
    higher-support alternative is effectively tied on total race time.
    """
    if len(ranked_plans) < 2:
        return ranked_plans

    best_plan = ranked_plans[0]
    if not _plan_uses_moderate_or_low_support_soft(degradation_models, best_plan):
        return ranked_plans

    for index, candidate in enumerate(ranked_plans[1:], start=1):
        if candidate.total_race_time - best_plan.total_race_time > max_gap_s:
            break
        same_structure = (
            candidate.strategy_type == best_plan.strategy_type
            and candidate.pit_lap == best_plan.pit_lap
            and candidate.second_pit_lap == best_plan.second_pit_lap
        )
        if not same_structure:
            continue
        if _plan_uses_moderate_or_low_support_soft(degradation_models, candidate):
            continue
        if not _plan_has_only_high_support_futures(degradation_models, candidate):
            continue

        reordered = [candidate, best_plan]
        reordered.extend(
            plan
            for idx, plan in enumerate(ranked_plans[1:], start=1)
            if idx != index
        )
        return reordered

    return ranked_plans


def estimate_stint_feasibility(
    degradation_models: Union[Dict, object],
    compound: str,
    tyre_life: int,
    stint_laps: int,
    pit_loss_value: float,
) -> Tuple[bool, str]:
    """Estimate whether a tyre compound can realistically complete a stint."""
    usable_tyre_life = {
        "SOFT": 20,
        "MEDIUM": 35,
        "HARD": 50,
    }

    max_viable = usable_tyre_life.get(compound, 30)
    estimated_stint_length = stint_laps
    ending_tyre_life = tyre_life + max(estimated_stint_length - 1, 0)

    if ending_tyre_life > max_viable:
        return (
            False,
            f"Unrealistic: {compound} would reach tyre age {ending_tyre_life} "
            f"after {estimated_stint_length} laps from age {tyre_life} "
            f"(max viable ~{max_viable})",
        )

    predicted_time = predict_lap_time(degradation_models, compound, tyre_life)
    if not _is_valid_prediction(predicted_time):
        return (False, f"No degradation model available for {compound}")

    support_info = _compound_model_info(degradation_models, compound)
    support_samples = support_info.get("samples")
    support_note = ""
    if isinstance(support_samples, int) and support_samples < 250:
        support_note = f"; low support ({support_samples} laps)"
    support_tier = _compound_support_info(degradation_models, compound).get("support_tier")
    if support_tier in {"Low", "Moderate"}:
        support_note += f"; {support_tier.lower()} confidence"

    return (
        True,
        f"{compound} is viable for {estimated_stint_length} laps from tyre age {tyre_life} "
        f"(ending age {ending_tyre_life}, estimated max ~{max_viable}){support_note}",
    )


def _build_feasibility_chain(reasons: List[Tuple[str, str]]) -> str:
    """Join per-stint feasibility messages into a single explanation."""
    return "; ".join(f"{label}: {reason}" for label, reason in reasons)


def _select_latest_near_optimal_one_stop_candidate(
    feasible_candidates: List[Tuple[int, float, str]],
    tolerance_s: float = ONE_STOP_SELECTION_TOLERANCE_S,
) -> Tuple[int, float, str]:
    """Prefer the latest pit lap inside a flat near-optimal timing band.

    Stop-timing backtests showed that some one-stop curves are nearly flat for
    several laps but the previous logic always chose the earliest minimum-time
    lap. In a deterministic decision-support setting, later laps inside a
    sub-second band are more honest than forcing an early pit stop on a
    marginal timing difference.
    """
    best_time = min(candidate[1] for candidate in feasible_candidates)
    near_optimal = [
        candidate for candidate in feasible_candidates
        if candidate[1] <= best_time + tolerance_s
    ]
    return max(near_optimal, key=lambda candidate: candidate[0])


def _count_local_minima(rows: List[dict]) -> int:
    """Count distinct local minima in a timing trace."""
    if len(rows) <= 2:
        return len(rows)

    minima = 0
    for index, row in enumerate(rows):
        current = row["total_race_time"]
        prev_value = rows[index - 1]["total_race_time"] if index > 0 else None
        next_value = rows[index + 1]["total_race_time"] if index < len(rows) - 1 else None

        left_ok = prev_value is None or current <= prev_value
        right_ok = next_value is None or current <= next_value
        strictly_better = (
            (prev_value is not None and current < prev_value)
            or (next_value is not None and current < next_value)
            or (prev_value is None or next_value is None)
        )
        if left_ok and right_ok and strictly_better:
            minima += 1

    return minima


def _curve_shape_from_band(near_optimal_laps: List[int]) -> str:
    """Return a simple label for the width of a near-optimal timing band."""
    if len(near_optimal_laps) >= 6:
        return "flat"
    if len(near_optimal_laps) >= 3:
        return "moderate"
    return "sharp"


def build_strategy_timing_trace(
    degradation_models: Union[Dict, object],
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    next_compound: str,
    final_compound: Optional[str] = None,
    min_stint_length: int = 3,
    near_optimal_tolerance_s: float = TIMING_TRACE_NEAR_OPTIMAL_TOLERANCE_S,
) -> dict:
    """Trace total race time across feasible first-stop timings for a fixed plan.

    For one-stop plans this evaluates every feasible pit lap directly. For
    fixed compound two-stop plans it traces first-stop timing while optimizing
    the second stop for that first-stop choice.
    """
    rows: List[dict] = []

    if final_compound is None:
        strategy_df = optimize_pit_window(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            compound=current_compound,
            post_pit_compound=next_compound,
            strict_predictions=True,
        )

        for row in strategy_df.loc[strategy_df["TotalTime"].notna(), ["PitLap", "TotalTime"]].itertuples(index=False):
            first_pit_lap = int(row.PitLap)
            remaining_after_pit = laps_remaining - first_pit_lap

            feasible_current, reason_current = estimate_stint_feasibility(
                degradation_models=degradation_models,
                compound=current_compound,
                tyre_life=current_tyre_life,
                stint_laps=first_pit_lap,
                pit_loss_value=pit_loss_value,
            )
            feasible_next, reason_next = estimate_stint_feasibility(
                degradation_models=degradation_models,
                compound=next_compound,
                tyre_life=1,
                stint_laps=remaining_after_pit,
                pit_loss_value=pit_loss_value,
            )

            if not (feasible_current and feasible_next):
                continue

            rows.append(
                {
                    "first_pit_lap": first_pit_lap,
                    "second_pit_lap": None,
                    "total_race_time": float(row.TotalTime),
                    "feasibility_reason": _build_feasibility_chain(
                        [
                            ("Current stint", reason_current),
                            (f"{next_compound} finish stint", reason_next),
                        ]
                    ),
                }
            )
    else:
        max_first_pit = laps_remaining - (2 * min_stint_length)
        for first_pit_lap in range(min_stint_length, max_first_pit + 1):
            best_row: Optional[dict] = None
            min_second_pit = first_pit_lap + min_stint_length
            max_second_pit = laps_remaining - min_stint_length

            for second_pit_lap in range(min_second_pit, max_second_pit + 1):
                stint1_laps = first_pit_lap
                stint2_laps = second_pit_lap - first_pit_lap
                stint3_laps = laps_remaining - second_pit_lap

                stint1_time = _simulate_stint_time(
                    degradation_models,
                    current_compound,
                    current_tyre_life,
                    stint1_laps,
                )
                stint2_time = _simulate_stint_time(
                    degradation_models,
                    next_compound,
                    1,
                    stint2_laps,
                )
                stint3_time = _simulate_stint_time(
                    degradation_models,
                    final_compound,
                    1,
                    stint3_laps,
                )
                if any(t is None for t in (stint1_time, stint2_time, stint3_time)):
                    continue

                feasible_1, reason_1 = estimate_stint_feasibility(
                    degradation_models,
                    current_compound,
                    current_tyre_life,
                    stint_laps=stint1_laps,
                    pit_loss_value=pit_loss_value,
                )
                feasible_2, reason_2 = estimate_stint_feasibility(
                    degradation_models,
                    next_compound,
                    1,
                    stint_laps=stint2_laps,
                    pit_loss_value=pit_loss_value,
                )
                feasible_3, reason_3 = estimate_stint_feasibility(
                    degradation_models,
                    final_compound,
                    1,
                    stint_laps=stint3_laps,
                    pit_loss_value=pit_loss_value,
                )
                if not (feasible_1 and feasible_2 and feasible_3):
                    continue

                total_time = (
                    float(stint1_time)
                    + pit_loss_value
                    + float(stint2_time)
                    + pit_loss_value
                    + float(stint3_time)
                )
                row_payload = {
                    "first_pit_lap": first_pit_lap,
                    "second_pit_lap": second_pit_lap,
                    "total_race_time": total_time,
                    "feasibility_reason": _build_feasibility_chain(
                        [
                            ("Current stint", reason_1),
                            (f"{next_compound} middle stint", reason_2),
                            (f"{final_compound} finish stint", reason_3),
                        ]
                    ),
                }
                if best_row is None or total_time < best_row["total_race_time"]:
                    best_row = row_payload

            if best_row is not None:
                rows.append(best_row)

    if not rows:
        raise ValueError(
            f"No feasible timing trace could be built for {current_compound}->{next_compound}"
            + (f"->{final_compound}" if final_compound else "")
        )

    rows.sort(key=lambda row: row["first_pit_lap"])
    best_row = min(rows, key=lambda row: row["total_race_time"])
    best_total = float(best_row["total_race_time"])
    near_optimal_rows = [
        row for row in rows
        if row["total_race_time"] <= best_total + near_optimal_tolerance_s
    ]
    near_optimal_laps = [int(row["first_pit_lap"]) for row in near_optimal_rows]
    first_pit_window = [int(row["first_pit_lap"]) for row in rows]
    expected_band = list(range(min(near_optimal_laps), max(near_optimal_laps) + 1))
    local_minima_count = _count_local_minima(rows)

    trace_rows = []
    for row in rows:
        trace_rows.append(
            {
                "first_pit_lap": int(row["first_pit_lap"]),
                "second_pit_lap": int(row["second_pit_lap"]) if row["second_pit_lap"] is not None else None,
                "total_race_time": round(float(row["total_race_time"]), 3),
                "delta_vs_best_s": round(float(row["total_race_time"] - best_total), 3),
            }
        )

    return {
        "strategy_type": "one-stop" if final_compound is None else "two-stop",
        "current_compound": current_compound,
        "current_tyre_life": int(current_tyre_life),
        "next_compound": next_compound,
        "final_compound": final_compound,
        "laps_remaining": int(laps_remaining),
        "best_first_pit_lap": int(best_row["first_pit_lap"]),
        "best_second_pit_lap": int(best_row["second_pit_lap"]) if best_row["second_pit_lap"] is not None else None,
        "best_total_race_time": round(best_total, 3),
        "near_optimal_tolerance_s": round(float(near_optimal_tolerance_s), 3),
        "near_optimal_band_laps": near_optimal_laps,
        "near_optimal_band_width_laps": len(near_optimal_laps),
        "curve_shape": _curve_shape_from_band(near_optimal_laps),
        "best_on_window_edge": bool(
            best_row["first_pit_lap"] == first_pit_window[0]
            or best_row["first_pit_lap"] == first_pit_window[-1]
        ),
        "near_optimal_band_fragmented": near_optimal_laps != expected_band,
        "local_minima_count": local_minima_count,
        "trace": trace_rows,
    }


def evaluate_one_stop_strategies(
    degradation_models: Union[Dict, object],
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    candidate_compounds: Optional[List[str]] = None,
) -> List[StrategyPlan]:
    """Evaluate all candidate one-stop strategies (current -> next -> end)."""
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    plans = []

    for next_compound in candidate_compounds:
        if next_compound == current_compound:
            continue

        strategy_df = optimize_pit_window(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            compound=current_compound,
            post_pit_compound=next_compound,
            strict_predictions=True,
        )

        if strategy_df.empty or strategy_df["TotalTime"].isna().all():
            logger.info(
                "One-stop strategy %s->%s skipped because no valid prediction path exists.",
                current_compound,
                next_compound,
            )
            continue

        feasible_candidates: List[Tuple[int, float, str]] = []
        for row in strategy_df.loc[strategy_df["TotalTime"].notna(), ["PitLap", "TotalTime"]].itertuples(index=False):
            pit_in_laps = int(row.PitLap)
            remaining_after_pit = laps_remaining - pit_in_laps

            feasible_current, reason_current = estimate_stint_feasibility(
                degradation_models=degradation_models,
                compound=current_compound,
                tyre_life=current_tyre_life,
                stint_laps=pit_in_laps,
                pit_loss_value=pit_loss_value,
            )
            feasible_next, reason_next = estimate_stint_feasibility(
                degradation_models=degradation_models,
                compound=next_compound,
                tyre_life=1,
                stint_laps=remaining_after_pit,
                pit_loss_value=pit_loss_value,
            )

            if not (feasible_current and feasible_next):
                continue

            feasible_candidates.append(
                (
                    pit_in_laps,
                    float(row.TotalTime),
                    _build_feasibility_chain(
                        [
                            ("Current stint", reason_current),
                            (f"{next_compound} finish stint", reason_next),
                        ]
                    ),
                )
            )

        if not feasible_candidates:
            logger.info(
                "One-stop strategy %s->%s skipped because no feasible pit window survived validation.",
                current_compound,
                next_compound,
            )
            continue

        optimal_pit_lap, total_race_time, feasibility_reason = _select_latest_near_optimal_one_stop_candidate(
            feasible_candidates,
            tolerance_s=ONE_STOP_SELECTION_TOLERANCE_S,
        )
        feasible = True

        plans.append(
            StrategyPlan(
                strategy_type="one-stop",
                current_compound=current_compound,
                current_tyre_life=current_tyre_life,
                next_compound=next_compound,
                final_compound=None,
                pit_lap=optimal_pit_lap,
                second_pit_lap=None,
                total_race_time=total_race_time,
                feasible=feasible,
                feasibility_reason=feasibility_reason,
                explanation=(
                    f"Pit in {optimal_pit_lap} lap{'s' if optimal_pit_lap != 1 else ''} "
                    f"for {next_compound}, then finish on {next_compound}"
                ),
                model_info={
                    "current_compound": _compound_model_info(degradation_models, current_compound),
                    "next_compound": _compound_model_info(degradation_models, next_compound),
                },
            )
        )

    plans.sort(key=lambda p: p.total_race_time)
    return plans


def evaluate_two_stop_strategies(
    degradation_models: Union[Dict, object],
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    candidate_compounds: Optional[List[str]] = None,
    min_stint_length: int = 3,
) -> List[StrategyPlan]:
    """Evaluate bounded two-stop strategies (current -> next -> final -> end).

    Phase 2E refinement:
    - replaces the old fixed 1/3 + 2/3 heuristic with a compact exhaustive
      search across valid first/second pit-lap pairs
    - keeps the search bounded via ``min_stint_length``
    - drops plans when any required model prediction is invalid
    """
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    plans = []

    for next_compound in candidate_compounds:
        if next_compound == current_compound:
            continue

        for final_compound in candidate_compounds:
            if final_compound == next_compound:
                continue

            best_candidate: Optional[Tuple[int, int, float]] = None

            max_first_pit = laps_remaining - (2 * min_stint_length)
            for first_pit_lap in range(min_stint_length, max_first_pit + 1):
                min_second_pit = first_pit_lap + min_stint_length
                max_second_pit = laps_remaining - min_stint_length

                for second_pit_lap in range(min_second_pit, max_second_pit + 1):
                    stint1_laps = first_pit_lap
                    stint2_laps = second_pit_lap - first_pit_lap
                    stint3_laps = laps_remaining - second_pit_lap

                    stint1_time = _simulate_stint_time(
                        degradation_models,
                        current_compound,
                        current_tyre_life,
                        stint1_laps,
                    )
                    stint2_time = _simulate_stint_time(
                        degradation_models,
                        next_compound,
                        1,
                        stint2_laps,
                    )
                    stint3_time = _simulate_stint_time(
                        degradation_models,
                        final_compound,
                        1,
                        stint3_laps,
                    )

                    if any(t is None for t in (stint1_time, stint2_time, stint3_time)):
                        continue

                    feasible_1, _ = estimate_stint_feasibility(
                        degradation_models,
                        current_compound,
                        current_tyre_life,
                        stint_laps=stint1_laps,
                        pit_loss_value=pit_loss_value,
                    )
                    feasible_2, _ = estimate_stint_feasibility(
                        degradation_models,
                        next_compound,
                        1,
                        stint_laps=stint2_laps,
                        pit_loss_value=pit_loss_value,
                    )
                    feasible_3, _ = estimate_stint_feasibility(
                        degradation_models,
                        final_compound,
                        1,
                        stint_laps=stint3_laps,
                        pit_loss_value=pit_loss_value,
                    )

                    if not (feasible_1 and feasible_2 and feasible_3):
                        continue

                    total_time = (
                        float(stint1_time)
                        + pit_loss_value
                        + float(stint2_time)
                        + pit_loss_value
                        + float(stint3_time)
                    )

                    if best_candidate is None or total_time < best_candidate[2]:
                        best_candidate = (first_pit_lap, second_pit_lap, total_time)

            if best_candidate is None:
                logger.info(
                    "Two-stop strategy %s->%s->%s skipped because no valid prediction path exists.",
                    current_compound,
                    next_compound,
                    final_compound,
                )
                continue

            first_pit_lap, second_pit_lap, total_time = best_candidate
            stint2_laps = second_pit_lap - first_pit_lap
            stint3_laps = laps_remaining - second_pit_lap

            feasible_1, reason_1 = estimate_stint_feasibility(
                degradation_models,
                current_compound,
                current_tyre_life,
                stint_laps=first_pit_lap,
                pit_loss_value=pit_loss_value,
            )
            feasible_2, reason_2 = estimate_stint_feasibility(
                degradation_models,
                next_compound,
                1,
                stint_laps=stint2_laps,
                pit_loss_value=pit_loss_value,
            )
            feasible_3, reason_3 = estimate_stint_feasibility(
                degradation_models,
                final_compound,
                1,
                stint_laps=stint3_laps,
                pit_loss_value=pit_loss_value,
            )

            feasible = feasible_1 and feasible_2 and feasible_3
            feasibility_reason = _build_feasibility_chain(
                [
                    ("Current stint", reason_1),
                    (f"{next_compound} middle stint", reason_2),
                    (f"{final_compound} finish stint", reason_3),
                ]
            )

            if not feasible:
                logger.info(
                    "Two-stop strategy %s->%s->%s skipped as infeasible. %s",
                    current_compound,
                    next_compound,
                    final_compound,
                    feasibility_reason,
                )
                continue

            one_stop_df = optimize_pit_window(
                degradation_models=degradation_models,
                pit_loss_value=pit_loss_value,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
                compound=current_compound,
                post_pit_compound=next_compound,
                strict_predictions=True,
            )
            one_stop_time = None
            if not one_stop_df.empty and not one_stop_df["TotalTime"].isna().all():
                _, one_stop_time = find_optimal_pit_lap(one_stop_df)

            plans.append(
                StrategyPlan(
                    strategy_type="two-stop",
                    current_compound=current_compound,
                    current_tyre_life=current_tyre_life,
                    next_compound=next_compound,
                    final_compound=final_compound,
                    pit_lap=first_pit_lap,
                    second_pit_lap=second_pit_lap,
                    total_race_time=total_time,
                    feasible=feasible,
                    feasibility_reason=feasibility_reason,
                    one_stop_estimate=one_stop_time,
                    explanation=(
                        f"Pit in {first_pit_lap} lap{'s' if first_pit_lap != 1 else ''} for {next_compound}, "
                        f"then {second_pit_lap - first_pit_lap} lap{'s' if (second_pit_lap - first_pit_lap) != 1 else ''} "
                        f"later for {final_compound}, finish on {final_compound}"
                    ),
                    model_info={
                        "current_compound": _compound_model_info(degradation_models, current_compound),
                        "next_compound": _compound_model_info(degradation_models, next_compound),
                        "final_compound": _compound_model_info(degradation_models, final_compound),
                    },
                )
            )

    plans.sort(key=lambda p: p.total_race_time)
    return plans


def rank_strategy_plans(
    one_stop_plans: List[StrategyPlan],
    two_stop_plans: List[StrategyPlan],
    prioritize_feasible: bool = True,
) -> List[StrategyPlan]:
    """Rank all feasible strategy options by total time."""
    all_plans = one_stop_plans + two_stop_plans
    all_plans.sort(key=lambda p: p.total_race_time)
    return all_plans


def recommend_best_strategy(
    degradation_models: Union[Dict, object],
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    candidate_compounds: Optional[List[str]] = None,
    include_two_stop: bool = True,
) -> Tuple[StrategyPlan, List[StrategyPlan]]:
    """Automatically recommend the best strategy from ranked options."""
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    one_stop_plans = evaluate_one_stop_strategies(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        candidate_compounds=candidate_compounds,
    )

    two_stop_plans = []
    if include_two_stop:
        two_stop_plans = evaluate_two_stop_strategies(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            candidate_compounds=candidate_compounds,
        )

    all_ranked_plans = rank_strategy_plans(one_stop_plans, two_stop_plans, prioritize_feasible=True)
    if not all_ranked_plans:
        raise ValueError("No valid strategies were generated.")

    all_ranked_plans = _apply_low_margin_soft_tie_break(
        all_ranked_plans,
        degradation_models=degradation_models,
    )

    return all_ranked_plans[0], all_ranked_plans
