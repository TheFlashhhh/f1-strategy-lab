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


def estimate_stint_feasibility(
    degradation_models: Union[Dict, object],
    compound: str,
    tyre_life: int,
    laps_remaining: int,
    pit_loss_value: float,
) -> Tuple[bool, str]:
    """Estimate whether a tyre compound can realistically complete a stint."""
    usable_tyre_life = {
        "SOFT": 20,
        "MEDIUM": 35,
        "HARD": 50,
    }

    max_viable = usable_tyre_life.get(compound, 30)
    estimated_stint_length = laps_remaining

    if estimated_stint_length > max_viable:
        return (
            False,
            f"Unrealistic: {compound} tyre cannot cover {estimated_stint_length} laps "
            f"(max viable ~{max_viable} laps)",
        )

    predicted_time = predict_lap_time(degradation_models, compound, tyre_life)
    if not _is_valid_prediction(predicted_time):
        return (False, f"No degradation model available for {compound}")

    support_info = _compound_model_info(degradation_models, compound)
    support_samples = support_info.get("samples")
    support_note = ""
    if isinstance(support_samples, int) and support_samples < 250:
        support_note = f"; low support ({support_samples} laps)"

    return (
        True,
        f"{compound} is viable for {estimated_stint_length} laps (estimated max ~{max_viable}){support_note}",
    )


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

        try:
            optimal_pit_lap, total_race_time = find_optimal_pit_lap(strategy_df)
        except ValueError as e:
            logger.info(
                "One-stop strategy %s->%s failed to find optimal pit lap: %s",
                current_compound,
                next_compound,
                e,
            )
            continue

        remaining_after_pit = laps_remaining - optimal_pit_lap
        feasible, feasibility_reason = estimate_stint_feasibility(
            degradation_models=degradation_models,
            compound=next_compound,
            tyre_life=1,
            laps_remaining=remaining_after_pit,
            pit_loss_value=pit_loss_value,
        )

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
                explanation=f"Pit to {next_compound} at lap {optimal_pit_lap}, finish on {next_compound}",
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
                first_pit_lap,
                pit_loss_value,
            )
            feasible_2, reason_2 = estimate_stint_feasibility(
                degradation_models,
                next_compound,
                1,
                stint2_laps,
                pit_loss_value,
            )
            feasible_3, reason_3 = estimate_stint_feasibility(
                degradation_models,
                final_compound,
                1,
                stint3_laps,
                pit_loss_value,
            )

            feasible = feasible_1 and feasible_2 and feasible_3
            feasibility_reason = "; ".join([reason_1, reason_2, reason_3])

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
                        f"Pit to {next_compound} (lap {first_pit_lap}), "
                        f"then {final_compound} (lap {second_pit_lap}), finish on {final_compound}"
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
    """Rank all strategy options by total time, with optional feasibility prioritization."""
    all_plans = one_stop_plans + two_stop_plans

    if prioritize_feasible:
        feasible_plans = [p for p in all_plans if p.feasible]
        infeasible_plans = [p for p in all_plans if not p.feasible]

        feasible_plans.sort(key=lambda p: p.total_race_time)
        infeasible_plans.sort(key=lambda p: p.total_race_time)

        return feasible_plans + infeasible_plans

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

    return all_ranked_plans[0], all_ranked_plans
