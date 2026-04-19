#!/usr/bin/env python
"""Run the Pre-3 defensibility audit and held-out Miami backtest."""

from __future__ import annotations

from collections import Counter
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.evaluate_degradation import evaluate_all_degradation
from src.features.hybrid_modeling import build_role_based_hybrid_model, summarize_hybrid_context
from src.simulation.strategy import estimate_pit_loss_window, predict_lap_time
from src.simulation.strategy_engine import (
    StrategyPlan,
    TIMING_TRACE_NEAR_OPTIMAL_TOLERANCE_S,
    build_strategy_timing_trace,
    recommend_best_strategy,
)
from src.simulation.strategy_validation import build_pre3_backtest_checkpoint_suite


def _to_python(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    return value


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_to_python)


def _plan_to_dict(plan: Optional[StrategyPlan]) -> Optional[dict]:
    if plan is None:
        return None
    return {
        "strategy_type": plan.strategy_type,
        "current_compound": plan.current_compound,
        "current_tyre_life": plan.current_tyre_life,
        "next_compound": plan.next_compound,
        "final_compound": plan.final_compound,
        "pit_lap": plan.pit_lap,
        "second_pit_lap": plan.second_pit_lap,
        "total_race_time": round(float(plan.total_race_time), 3),
        "feasible": bool(plan.feasible),
        "feasibility_reason": plan.feasibility_reason,
        "explanation": plan.explanation,
    }


def _simulate_plan_time(
    degradation_models,
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    next_compound: str,
    pit_lap: int,
    final_compound: Optional[str] = None,
    second_pit_lap: Optional[int] = None,
) -> Optional[float]:
    total_time = 0.0

    for lap in range(pit_lap):
        lap_time = predict_lap_time(degradation_models, current_compound, current_tyre_life + lap)
        if lap_time is None:
            return None
        total_time += float(lap_time)

    total_time += float(pit_loss_value)

    if final_compound is None or second_pit_lap is None:
        for lap in range(laps_remaining - pit_lap):
            lap_time = predict_lap_time(degradation_models, next_compound, 1 + lap)
            if lap_time is None:
                return None
            total_time += float(lap_time)
        return total_time

    second_stint_laps = second_pit_lap - pit_lap
    final_stint_laps = laps_remaining - second_pit_lap

    for lap in range(second_stint_laps):
        lap_time = predict_lap_time(degradation_models, next_compound, 1 + lap)
        if lap_time is None:
            return None
        total_time += float(lap_time)

    total_time += float(pit_loss_value)

    for lap in range(final_stint_laps):
        lap_time = predict_lap_time(degradation_models, final_compound, 1 + lap)
        if lap_time is None:
            return None
        total_time += float(lap_time)

    return total_time


def _match_actual_rank(ranked_plans: list[StrategyPlan], actual_plan: dict) -> tuple[Optional[int], Optional[StrategyPlan]]:
    for index, plan in enumerate(ranked_plans, start=1):
        if (
            plan.strategy_type == actual_plan["strategy_type"]
            and plan.next_compound == actual_plan["next_compound"]
            and plan.final_compound == actual_plan["final_compound"]
            and plan.pit_lap == actual_plan["pit_lap"]
            and plan.second_pit_lap == actual_plan["second_pit_lap"]
        ):
            return index, plan
    return None, None


def _match_compound_choice_rank(
    ranked_plans: list[StrategyPlan],
    actual_plan: dict,
) -> tuple[Optional[int], Optional[StrategyPlan]]:
    """Match strategy type + future compounds while ignoring pit timing."""
    for index, plan in enumerate(ranked_plans, start=1):
        if (
            plan.strategy_type == actual_plan["strategy_type"]
            and plan.next_compound == actual_plan["next_compound"]
            and plan.final_compound == actual_plan["final_compound"]
        ):
            return index, plan
    return None, None


def _build_actual_remaining_plan(stints: pd.DataFrame, checkpoint_lap: int) -> dict:
    first = stints.iloc[0]
    second = stints.iloc[1]
    actual = {
        "strategy_type": "one-stop" if len(stints) == 2 else "two-stop",
        "next_compound": second["Compound"],
        "pit_lap": int(first["end_lap"] - checkpoint_lap + 1),
        "final_compound": None,
        "second_pit_lap": None,
    }
    if len(stints) >= 3:
        third = stints.iloc[2]
        actual["final_compound"] = third["Compound"]
        actual["second_pit_lap"] = int(second["end_lap"] - checkpoint_lap + 1)
    return actual


def _classify_miss(model_best: dict, actual_plan: dict) -> str:
    """Classify whether a miss came from compounds, timing, or both."""
    if model_best is None or actual_plan is None:
        return "unjudgeable from available data"

    same_strategy_type = model_best["strategy_type"] == actual_plan["strategy_type"]
    same_compounds = (
        model_best["next_compound"] == actual_plan["next_compound"]
        and model_best["final_compound"] == actual_plan["final_compound"]
    )
    same_first_pit = model_best["pit_lap"] == actual_plan["pit_lap"]
    same_second_pit = model_best["second_pit_lap"] == actual_plan["second_pit_lap"]

    compound_match = same_strategy_type and same_compounds
    timing_match = same_strategy_type and same_first_pit and same_second_pit

    if compound_match and timing_match:
        return "exact match"
    if compound_match:
        return "timing-only"
    if timing_match:
        return "compound-only"
    return "both"


def _stop_timing_delta(model_best: dict, actual_plan: dict) -> dict:
    """Return first/second stop timing deltas when they can be compared."""
    if model_best is None or actual_plan is None:
        return {
            "first_stop_delta_laps": None,
            "second_stop_delta_laps": None,
            "absolute_total_stop_error_laps": None,
        }

    first_delta = (
        int(model_best["pit_lap"] - actual_plan["pit_lap"])
        if model_best["pit_lap"] is not None and actual_plan["pit_lap"] is not None
        else None
    )
    second_delta = (
        int(model_best["second_pit_lap"] - actual_plan["second_pit_lap"])
        if model_best["second_pit_lap"] is not None and actual_plan["second_pit_lap"] is not None
        else None
    )
    abs_total = None
    if first_delta is not None or second_delta is not None:
        abs_total = abs(first_delta or 0) + abs(second_delta or 0)

    return {
        "first_stop_delta_laps": first_delta,
        "second_stop_delta_laps": second_delta,
        "absolute_total_stop_error_laps": abs_total,
    }


def _actual_timing_context(
    degradation_models,
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    actual_plan: dict,
    tolerance_s: float = TIMING_TRACE_NEAR_OPTIMAL_TOLERANCE_S,
) -> dict:
    """Build timing-trace context for the actual future compound sequence."""
    try:
        trace = build_strategy_timing_trace(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            next_compound=actual_plan["next_compound"],
            final_compound=actual_plan["final_compound"],
            near_optimal_tolerance_s=tolerance_s,
        )
    except ValueError as exc:
        return {
            "trace_available": False,
            "trace_error": str(exc),
            "trace_summary": None,
            "actual_first_stop_lap": actual_plan.get("pit_lap"),
            "actual_first_stop_in_near_optimal_band": None,
            "actual_first_stop_delta_vs_sequence_best_s": None,
        }

    actual_first_stop = actual_plan.get("pit_lap")
    actual_trace_row = next(
        (row for row in trace["trace"] if row["first_pit_lap"] == actual_first_stop),
        None,
    )
    actual_delta = actual_trace_row["delta_vs_best_s"] if actual_trace_row else None
    within_band = actual_first_stop in trace["near_optimal_band_laps"]
    return {
        "trace_available": True,
        "trace_error": None,
        "trace_summary": {
            "strategy_type": trace["strategy_type"],
            "next_compound": trace["next_compound"],
            "final_compound": trace["final_compound"],
            "best_first_stop_lap": trace["best_first_pit_lap"],
            "best_second_stop_lap": trace["best_second_pit_lap"],
            "best_total_race_time": trace["best_total_race_time"],
            "near_optimal_tolerance_s": trace["near_optimal_tolerance_s"],
            "near_optimal_band_laps": trace["near_optimal_band_laps"],
            "curve_shape": trace["curve_shape"],
            "best_on_window_edge": trace["best_on_window_edge"],
            "local_minima_count": trace["local_minima_count"],
        },
        "actual_first_stop_lap": actual_first_stop,
        "actual_first_stop_in_near_optimal_band": within_band,
        "actual_first_stop_delta_vs_sequence_best_s": actual_delta,
    }


def _classify_failure_mode(
    model_best: dict,
    actual_plan: dict,
    actual_timing_context: dict,
) -> str:
    """Return a more honest backtest failure label."""
    baseline_label = _classify_miss(model_best, actual_plan)
    if baseline_label == "exact match":
        return baseline_label

    same_strategy_type = model_best["strategy_type"] == actual_plan["strategy_type"]
    same_compounds = (
        model_best["next_compound"] == actual_plan["next_compound"]
        and model_best["final_compound"] == actual_plan["final_compound"]
    )
    within_band = actual_timing_context.get("actual_first_stop_in_near_optimal_band", False)
    if same_strategy_type and same_compounds and within_band:
        return "effectively near-equivalent despite rank miss"
    return baseline_label


def _build_backtest_diagnostics(summary: dict) -> dict:
    """Build an inspectable diagnostics artifact from checkpoint results."""
    checkpoints = summary["checkpoints"]
    miss_type_counts: Counter = Counter()
    miss_type_delta_buckets: dict[str, list[float]] = {}
    miss_type_stop_error_buckets: dict[str, list[float]] = {}

    for item in checkpoints:
        miss_type = item["failure_mode"]
        miss_type_counts[miss_type] += 1

        delta = item.get("delta_model_best_vs_actual_s")
        if delta is not None:
            miss_type_delta_buckets.setdefault(miss_type, []).append(float(delta))

        stop_error = item["stop_timing_delta_laps"].get("absolute_total_stop_error_laps")
        if stop_error is not None:
            miss_type_stop_error_buckets.setdefault(miss_type, []).append(float(stop_error))

    def _mean_or_none(values: list[float]) -> Optional[float]:
        return round(float(np.mean(values)), 3) if values else None

    aggregate = {
        "checkpoint_count": len(checkpoints),
        "exact_match_count": miss_type_counts.get("exact match", 0),
        "miss_type_counts": dict(miss_type_counts),
        "real_failure_count": sum(
            1
            for item in checkpoints
            if item["failure_mode"] not in {"exact match", "effectively near-equivalent despite rank miss"}
        ),
        "actual_strategy_top_3_hits": summary["summary"]["actual_strategy_top_3_hits"],
        "exact_top_3_hits": summary["summary"]["actual_strategy_top_3_hits"],
        "actual_next_compound_match_count": sum(
            1
            for item in checkpoints
            if item["model_best"]["next_compound"] == item["actual_remaining_strategy"]["next_compound"]
        ),
        "actual_next_compound_match_rate": round(
            float(
                np.mean(
                    [
                        item["model_best"]["next_compound"] == item["actual_remaining_strategy"]["next_compound"]
                        for item in checkpoints
                    ]
                )
            ),
            3,
        ) if checkpoints else None,
        "compound_choice_top_3_hits": sum(
            1 for item in checkpoints if item["actual_compound_choice_rank"] is not None and item["actual_compound_choice_rank"] <= 3
        ),
        "near_equivalent_rank_miss_count": miss_type_counts.get("effectively near-equivalent despite rank miss", 0),
        "average_delta_model_best_vs_actual_s": summary["summary"]["average_delta_model_best_vs_actual_s"],
        "average_abs_total_stop_error_laps": _mean_or_none(
            [
                item["stop_timing_delta_laps"]["absolute_total_stop_error_laps"]
                for item in checkpoints
                if item["stop_timing_delta_laps"]["absolute_total_stop_error_laps"] is not None
            ]
        ),
        "average_abs_first_stop_error_laps": _mean_or_none(
            [
                abs(item["stop_timing_delta_laps"]["first_stop_delta_laps"])
                for item in checkpoints
                if item["stop_timing_delta_laps"]["first_stop_delta_laps"] is not None
            ]
        ),
        "actual_stop_in_near_optimal_band_count": sum(
            1
            for item in checkpoints
            if item["actual_sequence_timing_context"]["actual_first_stop_in_near_optimal_band"] is True
        ),
        "actual_sequence_trace_unavailable_count": sum(
            1
            for item in checkpoints
            if not item["actual_sequence_timing_context"].get("trace_available", True)
        ),
        "flat_or_edge_optimum_count": sum(
            1
            for item in checkpoints
            if item["model_best_timing_trace"]["curve_shape"] == "flat"
            or item["model_best_timing_trace"]["best_on_window_edge"]
        ),
        "average_delta_by_miss_type_s": {
            miss_type: _mean_or_none(values)
            for miss_type, values in miss_type_delta_buckets.items()
        },
        "average_abs_stop_error_by_miss_type_laps": {
            miss_type: _mean_or_none(values)
            for miss_type, values in miss_type_stop_error_buckets.items()
        },
        "observations": [],
    }

    if aggregate["actual_strategy_top_3_hits"] == 0:
        aggregate["observations"].append(
            "No checkpoints matched the exact actual remaining strategy in the model top 3."
        )
    if aggregate["compound_choice_top_3_hits"] > aggregate["actual_strategy_top_3_hits"]:
        aggregate["observations"].append(
            "Compound-choice alignment is better than exact-timing alignment, so timing error is a major miss driver."
        )
    if miss_type_counts.get("both", 0):
        aggregate["observations"].append(
            "Several misses involve both compound choice and stop timing, which points to pace-shape issues rather than timing only."
        )
    if miss_type_counts.get("timing-only", 0):
        aggregate["observations"].append(
            "Some misses preserve the broad compound call but pit too early or too late versus the actual race."
        )
    if miss_type_counts.get("effectively near-equivalent despite rank miss", 0):
        aggregate["observations"].append(
            "Some rank misses are effectively near-equivalent: the actual stop falls inside the model's near-optimal timing band."
        )
    if aggregate["actual_sequence_trace_unavailable_count"]:
        aggregate["observations"].append(
            "Some actual future sequences are not even feasible under the current model assumptions, which is stronger evidence of a pace-shape or compound-choice gap."
        )
    if aggregate["flat_or_edge_optimum_count"]:
        aggregate["observations"].append(
            "Several model-best timing traces are flat or sit on a feasible-window edge, so exact pit-lap rank should not be over-read."
        )

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": summary["phase"],
        "holdout_year": summary["holdout_year"],
        "training_years": summary["training_years"],
        "aggregate": aggregate,
        "checkpoints": checkpoints,
    }


def main() -> None:
    print("\n" + "=" * 88)
    print("PRE-3 DEFENSIBILITY AUDIT + MIAMI HOLDOUT BACKTEST")
    print("=" * 88)

    loader = DataLoader(project_root=ROOT)
    miami_raw = loader.load_data("miami_historical")

    role_model, hybrid_context = build_role_based_hybrid_model(project_root=ROOT)
    support_summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "pre3",
        "blend_summary": summarize_hybrid_context(hybrid_context),
        "compound_support": role_model.to_dict(),
        "warnings": [
            "SOFT should not be treated as equally supported as MEDIUM/HARD unless its tier improves.",
            "2026 data is now framed as bounded recency support rather than direct Miami truth.",
        ],
    }
    support_path = ROOT / "data" / "processed" / "pre3_compound_support_summary.json"
    _json_dump(support_path, support_summary)

    holdout_year = 2024
    train_raw = miami_raw[miami_raw["season"] < holdout_year].copy()
    holdout_raw = miami_raw[miami_raw["season"] == holdout_year].copy()
    if train_raw.empty or holdout_raw.empty:
        raise RuntimeError("Held-out Miami backtest could not be constructed from local data.")

    train_model_df = build_model_df(clean_laps(detect_pit_stops(select_relevant_columns(train_raw))))
    holdout_df = detect_pit_stops(select_relevant_columns(holdout_raw))
    train_models = evaluate_all_degradation(train_model_df, use_fuel_correction=True, use_piecewise=True)

    pit_loss_samples = estimate_pit_loss_window(detect_pit_stops(select_relevant_columns(train_raw)))
    if len(pit_loss_samples) == 0:
        raise RuntimeError("No pit-loss samples were available in the Miami training subset.")
    pit_loss_value = float(np.median(pit_loss_samples))

    print(f"Holdout year: Miami {holdout_year}")
    print(f"Training years: {sorted(train_raw['season'].unique().tolist())}")
    print(f"Training Miami pit-loss baseline: {pit_loss_value:.2f}s")

    checkpoints = build_pre3_backtest_checkpoint_suite()

    holdout_max_lap = int(holdout_df["LapNumber"].max())
    backtest_results = []

    for checkpoint in checkpoints:
        driver_df = holdout_df[holdout_df["Driver"] == checkpoint.driver].copy()
        stints = (
            driver_df.dropna(subset=["Compound"])
            .groupby(["Stint", "Compound"], as_index=False)
            .agg(start_lap=("LapNumber", "min"), end_lap=("LapNumber", "max"))
            .sort_values("start_lap")
            .reset_index(drop=True)
        )
        if len(stints) < 2:
            continue
        first_stint = stints.iloc[0]
        if not (int(first_stint["start_lap"]) <= checkpoint.checkpoint_lap <= int(first_stint["end_lap"])):
            continue

        row = driver_df[driver_df["LapNumber"] == checkpoint.checkpoint_lap].iloc[0]
        current_compound = row["Compound"]
        current_tyre_life = int(row["TyreLife"])
        laps_remaining = holdout_max_lap - checkpoint.checkpoint_lap + 1

        best_plan, ranked_plans = recommend_best_strategy(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            include_two_stop=True,
        )

        actual_plan = _build_actual_remaining_plan(stints, checkpoint.checkpoint_lap)
        actual_rank, actual_ranked_plan = _match_actual_rank(ranked_plans, actual_plan)
        compound_choice_rank, compound_choice_plan = _match_compound_choice_rank(ranked_plans, actual_plan)
        actual_plan_time = _simulate_plan_time(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            next_compound=actual_plan["next_compound"],
            pit_lap=actual_plan["pit_lap"],
            final_compound=actual_plan["final_compound"],
            second_pit_lap=actual_plan["second_pit_lap"],
        )
        delta_vs_actual = (
            round(float(actual_plan_time - best_plan.total_race_time), 3)
            if actual_plan_time is not None
            else None
        )
        model_best_dict = _plan_to_dict(best_plan)
        model_best_timing_trace = build_strategy_timing_trace(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            next_compound=best_plan.next_compound,
            final_compound=best_plan.final_compound,
        )
        actual_sequence_timing_context = _actual_timing_context(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            actual_plan=actual_plan,
        )
        failure_mode = _classify_failure_mode(
            model_best=model_best_dict,
            actual_plan=actual_plan,
            actual_timing_context=actual_sequence_timing_context,
        )
        stop_timing_delta = _stop_timing_delta(model_best_dict, actual_plan)

        result = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "season": checkpoint.season,
            "driver": checkpoint.driver,
            "checkpoint_lap": checkpoint.checkpoint_lap,
            "rationale": checkpoint.rationale,
            "current_compound": current_compound,
            "current_tyre_life": current_tyre_life,
            "laps_remaining": laps_remaining,
            "model_best": model_best_dict,
            "actual_remaining_strategy": actual_plan,
            "actual_strategy_rank": actual_rank,
            "actual_strategy_in_top_3": actual_rank is not None and actual_rank <= 3,
            "actual_compound_choice_rank": compound_choice_rank,
            "actual_compound_choice_in_top_3": compound_choice_rank is not None and compound_choice_rank <= 3,
            "actual_strategy_estimated_time": round(float(actual_plan_time), 3) if actual_plan_time is not None else None,
            "delta_model_best_vs_actual_s": delta_vs_actual,
            "top_3_model_plans": [_plan_to_dict(plan) for plan in ranked_plans[:3]],
            "actual_ranked_plan": _plan_to_dict(actual_ranked_plan),
            "actual_compound_choice_ranked_plan": _plan_to_dict(compound_choice_plan),
            "actual_stop_timing": {
                "first_stop_lap": actual_plan["pit_lap"],
                "second_stop_lap": actual_plan["second_pit_lap"],
            },
            "model_best_stop_timing": {
                "first_stop_lap": model_best_dict["pit_lap"],
                "second_stop_lap": model_best_dict["second_pit_lap"],
            },
            "failure_mode": failure_mode,
            "stop_timing_delta_laps": stop_timing_delta,
            "absolute_first_stop_timing_error_laps": (
                abs(stop_timing_delta["first_stop_delta_laps"])
                if stop_timing_delta["first_stop_delta_laps"] is not None
                else None
            ),
            "model_best_timing_trace": {
                "strategy_type": model_best_timing_trace["strategy_type"],
                "next_compound": model_best_timing_trace["next_compound"],
                "final_compound": model_best_timing_trace["final_compound"],
                "best_first_stop_lap": model_best_timing_trace["best_first_pit_lap"],
                "best_second_stop_lap": model_best_timing_trace["best_second_pit_lap"],
                "best_total_race_time": model_best_timing_trace["best_total_race_time"],
                "near_optimal_tolerance_s": model_best_timing_trace["near_optimal_tolerance_s"],
                "near_optimal_band_laps": model_best_timing_trace["near_optimal_band_laps"],
                "curve_shape": model_best_timing_trace["curve_shape"],
                "best_on_window_edge": model_best_timing_trace["best_on_window_edge"],
                "local_minima_count": model_best_timing_trace["local_minima_count"],
            },
            "actual_sequence_timing_context": actual_sequence_timing_context,
        }
        backtest_results.append(result)

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "pre3",
        "holdout_year": holdout_year,
        "training_years": sorted(train_raw["season"].unique().tolist()),
        "design": {
            "target": "held-out Miami decision-support backtest",
            "why_2024": "Miami 2024 is the primary clean holdout in the local data and supports more first-stint checkpoints than Miami 2025.",
            "candidate_years_considered": [2024, 2025],
            "excluded_year_notes": {
                "2025": "Not used for canonical first-stint checkpoints because many drivers have unknown compounds through laps 23-24 in the local file."
            },
            "leakage_control": "The backtest trains on earlier Miami races only and excludes 2026 recency data to avoid future leakage.",
        },
        "training_pit_loss_value": round(pit_loss_value, 3),
        "checkpoints": backtest_results,
        "summary": {
            "checkpoint_count": len(backtest_results),
            "actual_strategy_top_3_hits": sum(1 for item in backtest_results if item["actual_strategy_in_top_3"]),
            "exact_top_3_hits": sum(1 for item in backtest_results if item["actual_strategy_in_top_3"]),
            "actual_strategy_rank_available": sum(1 for item in backtest_results if item["actual_strategy_rank"] is not None),
            "actual_compound_choice_top_3_hits": sum(
                1 for item in backtest_results if item["actual_compound_choice_in_top_3"]
            ),
            "actual_next_compound_match_count": sum(
                1
                for item in backtest_results
                if item["model_best"]["next_compound"] == item["actual_remaining_strategy"]["next_compound"]
            ),
            "actual_stop_in_near_optimal_band_count": sum(
                1
                for item in backtest_results
                if item["actual_sequence_timing_context"]["actual_first_stop_in_near_optimal_band"] is True
            ),
            "near_equivalent_rank_miss_count": sum(
                1
                for item in backtest_results
                if item["failure_mode"] == "effectively near-equivalent despite rank miss"
            ),
            "real_failure_count": sum(
                1
                for item in backtest_results
                if item["failure_mode"] not in {"exact match", "effectively near-equivalent despite rank miss"}
            ),
            "average_abs_first_stop_error_laps": (
                round(
                    float(
                        np.mean(
                            [
                                item["absolute_first_stop_timing_error_laps"]
                                for item in backtest_results
                                if item["absolute_first_stop_timing_error_laps"] is not None
                            ]
                        )
                    ),
                    3,
                )
                if any(item["absolute_first_stop_timing_error_laps"] is not None for item in backtest_results)
                else None
            ),
            "average_delta_model_best_vs_actual_s": (
                round(
                    float(
                        np.mean(
                            [
                                item["delta_model_best_vs_actual_s"]
                                for item in backtest_results
                                if item["delta_model_best_vs_actual_s"] is not None
                            ]
                        )
                    ),
                    3,
                )
                if any(item["delta_model_best_vs_actual_s"] is not None for item in backtest_results)
                else None
            ),
        },
    }

    backtest_path = ROOT / "data" / "processed" / "pre3_backtest_summary.json"
    _json_dump(backtest_path, summary)
    diagnostics = _build_backtest_diagnostics(summary)
    diagnostics_path = ROOT / "data" / "processed" / "pre3_backtest_diagnostics.json"
    _json_dump(diagnostics_path, diagnostics)

    print("\nCompound support tiers:")
    for compound, support in role_model.to_dict().items():
        print(
            f"  {compound}: {support['support_tier']} | "
            f"Miami model laps {support['miami']['model_laps']} | "
            f"2026 model laps {support['recency']['model_laps']}"
        )

    print("\nBacktest checkpoints:")
    for item in backtest_results:
        actual = item["actual_remaining_strategy"]
        top3 = "yes" if item["actual_strategy_in_top_3"] else "no"
        compound_top3 = "yes" if item["actual_compound_choice_in_top_3"] else "no"
        actual_trace = item["actual_sequence_timing_context"]
        near_band = actual_trace["actual_first_stop_in_near_optimal_band"]
        near_band_str = (
            "n/a (sequence infeasible in-model)"
            if near_band is None
            else str(bool(near_band))
        )
        print(
            f"  {item['checkpoint_id']}: model={item['model_best']['next_compound']} @ L{item['model_best']['pit_lap']} | "
            f"actual={actual['next_compound']} @ L{actual['pit_lap']} | "
            f"top3={top3} | compound-top3={compound_top3} | miss={item['failure_mode']} | "
            f"near-band={near_band_str}"
        )

    print("\nBacktest diagnostics summary:")
    print(f"  Exact matches: {diagnostics['aggregate']['exact_match_count']}")
    print(f"  Exact top-3 hits: {diagnostics['aggregate']['exact_top_3_hits']}")
    print(f"  Actual strategy top-3 hits: {diagnostics['aggregate']['actual_strategy_top_3_hits']}")
    print(f"  Near-equivalent timing/rank misses: {diagnostics['aggregate']['near_equivalent_rank_miss_count']}")
    print(f"  Compound-choice top-3 hits: {diagnostics['aggregate']['compound_choice_top_3_hits']}")
    print(f"  Actual next-compound matches: {diagnostics['aggregate']['actual_next_compound_match_count']}")
    print(f"  Actual next-compound match rate: {diagnostics['aggregate']['actual_next_compound_match_rate']}")
    print(f"  Actual stop in near-optimal band: {diagnostics['aggregate']['actual_stop_in_near_optimal_band_count']}")
    print(f"  Avg first-stop timing error: {diagnostics['aggregate']['average_abs_first_stop_error_laps']}")
    print(f"  Avg model-best vs actual delta: {diagnostics['aggregate']['average_delta_model_best_vs_actual_s']}")
    print(f"  Remaining real pace/compound failures: {diagnostics['aggregate']['real_failure_count']}")
    print(f"  Avg abs total stop error: {diagnostics['aggregate']['average_abs_total_stop_error_laps']}")
    for miss_type, count in diagnostics["aggregate"]["miss_type_counts"].items():
        print(f"  {miss_type}: {count}")

    print("\nArtifacts:")
    print(f"  - {support_path}")
    print(f"  - {backtest_path}")
    print(f"  - {diagnostics_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
