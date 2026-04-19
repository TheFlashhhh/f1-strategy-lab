#!/usr/bin/env python
"""Run a Pre-Phase-3 stop-timing audit across representative and backtest cases."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.evaluate_degradation import evaluate_all_degradation
from src.features.hybrid_modeling import build_role_based_hybrid_model
from src.simulation.strategy import estimate_pit_loss_window
from src.simulation.strategy_engine import build_strategy_timing_trace, recommend_best_strategy
from src.simulation.strategy_validation import (
    build_pre3_backtest_checkpoint_suite,
    build_representative_scenario_suite,
)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def _representative_case_audit(deg_result, pit_loss_value: float) -> list[dict]:
    audits = []
    for scenario in build_representative_scenario_suite():
        best_plan, ranked_plans = recommend_best_strategy(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=scenario.current_compound,
            current_tyre_life=scenario.current_tyre_life,
            laps_remaining=scenario.laps_remaining,
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            include_two_stop=True,
        )
        timing_trace = build_strategy_timing_trace(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=scenario.current_compound,
            current_tyre_life=scenario.current_tyre_life,
            laps_remaining=scenario.laps_remaining,
            next_compound=best_plan.next_compound,
            final_compound=best_plan.final_compound,
        )
        audits.append(
            {
                "scenario_id": scenario.scenario_id,
                "current_compound": scenario.current_compound,
                "current_tyre_life": scenario.current_tyre_life,
                "laps_remaining": scenario.laps_remaining,
                "rationale": scenario.rationale,
                "best_strategy_type": best_plan.strategy_type,
                "best_next_compound": best_plan.next_compound,
                "best_final_compound": best_plan.final_compound,
                "best_pit_lap": best_plan.pit_lap,
                "best_second_pit_lap": best_plan.second_pit_lap,
                "runner_up_gap_s": round(ranked_plans[1].total_race_time - best_plan.total_race_time, 3) if len(ranked_plans) > 1 else None,
                "timing_trace_summary": {
                    "best_first_pit_lap": timing_trace["best_first_pit_lap"],
                    "best_second_pit_lap": timing_trace["best_second_pit_lap"],
                    "best_total_race_time": timing_trace["best_total_race_time"],
                    "near_optimal_tolerance_s": timing_trace["near_optimal_tolerance_s"],
                    "near_optimal_band_laps": timing_trace["near_optimal_band_laps"],
                    "curve_shape": timing_trace["curve_shape"],
                    "best_on_window_edge": timing_trace["best_on_window_edge"],
                    "local_minima_count": timing_trace["local_minima_count"],
                },
                "trace": timing_trace["trace"],
            }
        )
    return audits


def _backtest_case_audit() -> tuple[list[dict], float]:
    loader = DataLoader(project_root=ROOT)
    miami_raw = loader.load_data("miami_historical")
    train_raw = miami_raw[miami_raw["season"] < 2024].copy()
    holdout_raw = miami_raw[miami_raw["season"] == 2024].copy()

    train_model_df = build_model_df(clean_laps(detect_pit_stops(select_relevant_columns(train_raw))))
    holdout_df = detect_pit_stops(select_relevant_columns(holdout_raw))
    train_models = evaluate_all_degradation(train_model_df, use_fuel_correction=True, use_piecewise=True)
    pit_loss_value = float(np.median(estimate_pit_loss_window(detect_pit_stops(select_relevant_columns(train_raw)))))

    audits = []
    holdout_max_lap = int(holdout_df["LapNumber"].max())
    for checkpoint in build_pre3_backtest_checkpoint_suite():
        checkpoint_id = checkpoint.checkpoint_id
        driver = checkpoint.driver
        checkpoint_lap = checkpoint.checkpoint_lap
        driver_df = holdout_df[holdout_df["Driver"] == driver].copy()
        stints = (
            driver_df.dropna(subset=["Compound"])
            .groupby(["Stint", "Compound"], as_index=False)
            .agg(start_lap=("LapNumber", "min"), end_lap=("LapNumber", "max"))
            .sort_values("start_lap")
            .reset_index(drop=True)
        )
        row = driver_df[driver_df["LapNumber"] == checkpoint_lap].iloc[0]
        current_compound = row["Compound"]
        current_tyre_life = int(row["TyreLife"])
        laps_remaining = holdout_max_lap - checkpoint_lap + 1

        best_plan, _ = recommend_best_strategy(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            include_two_stop=True,
        )
        actual_plan = _build_actual_remaining_plan(stints, checkpoint_lap)
        model_trace = build_strategy_timing_trace(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            next_compound=best_plan.next_compound,
            final_compound=best_plan.final_compound,
        )
        actual_trace_summary = None
        actual_trace_rows = []
        actual_trace_error = None
        try:
            actual_trace = build_strategy_timing_trace(
                degradation_models=train_models,
                pit_loss_value=pit_loss_value,
                current_compound=current_compound,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
                next_compound=actual_plan["next_compound"],
                final_compound=actual_plan["final_compound"],
            )
            actual_trace_summary = {
                "best_first_pit_lap": actual_trace["best_first_pit_lap"],
                "best_second_pit_lap": actual_trace["best_second_pit_lap"],
                "near_optimal_band_laps": actual_trace["near_optimal_band_laps"],
                "curve_shape": actual_trace["curve_shape"],
                "best_on_window_edge": actual_trace["best_on_window_edge"],
                "actual_stop_in_near_optimal_band": actual_plan["pit_lap"] in actual_trace["near_optimal_band_laps"],
            }
            actual_trace_rows = actual_trace["trace"]
        except ValueError as exc:
            actual_trace_error = str(exc)
            actual_trace_summary = {
                "best_first_pit_lap": None,
                "best_second_pit_lap": None,
                "near_optimal_band_laps": [],
                "curve_shape": None,
                "best_on_window_edge": None,
                "actual_stop_in_near_optimal_band": None,
                "trace_error": actual_trace_error,
            }
            actual_trace_rows = []

        audits.append(
            {
                "checkpoint_id": checkpoint_id,
                "season": checkpoint.season,
                "driver": driver,
                "checkpoint_lap": checkpoint_lap,
                "current_compound": current_compound,
                "current_tyre_life": current_tyre_life,
                "laps_remaining": laps_remaining,
                "model_best": {
                    "strategy_type": best_plan.strategy_type,
                    "next_compound": best_plan.next_compound,
                    "final_compound": best_plan.final_compound,
                    "pit_lap": best_plan.pit_lap,
                    "second_pit_lap": best_plan.second_pit_lap,
                },
                "actual_remaining_strategy": actual_plan,
                "actual_next_compound_matches_model_best": best_plan.next_compound == actual_plan["next_compound"],
                "absolute_first_stop_timing_error_laps": abs(best_plan.pit_lap - actual_plan["pit_lap"]),
                "model_best_timing_trace_summary": {
                    "best_first_pit_lap": model_trace["best_first_pit_lap"],
                    "best_second_pit_lap": model_trace["best_second_pit_lap"],
                    "near_optimal_band_laps": model_trace["near_optimal_band_laps"],
                    "curve_shape": model_trace["curve_shape"],
                    "best_on_window_edge": model_trace["best_on_window_edge"],
                },
                "actual_sequence_timing_trace_summary": actual_trace_summary,
                "model_best_trace": model_trace["trace"],
                "actual_sequence_trace": actual_trace_rows,
                "actual_sequence_trace_error": actual_trace_error,
            }
        )
    return audits, pit_loss_value


def main() -> None:
    print("\n" + "=" * 88)
    print("PRE-3 STOP-TIMING AUDIT")
    print("=" * 88)

    loader = DataLoader(project_root=ROOT)
    pit_raw = loader.load_data("miami_historical")
    deg_result, _ = build_role_based_hybrid_model(project_root=ROOT)
    pit_loss_value = float(np.median(estimate_pit_loss_window(detect_pit_stops(select_relevant_columns(pit_raw)))))

    representative_cases = _representative_case_audit(deg_result, pit_loss_value)
    backtest_cases, backtest_pit_loss_value = _backtest_case_audit()

    aggregate = {
        "representative_case_count": len(representative_cases),
        "representative_flat_curve_count": sum(
            1 for case in representative_cases if case["timing_trace_summary"]["curve_shape"] == "flat"
        ),
        "representative_edge_optimum_count": sum(
            1 for case in representative_cases if case["timing_trace_summary"]["best_on_window_edge"]
        ),
        "representative_two_stop_case_count": sum(
            1 for case in representative_cases if case["best_strategy_type"] == "two-stop"
        ),
        "backtest_checkpoint_count": len(backtest_cases),
        "backtest_actual_next_compound_match_count": sum(
            1 for case in backtest_cases if case["actual_next_compound_matches_model_best"]
        ),
        "backtest_actual_stop_in_near_optimal_band_count": sum(
            1
            for case in backtest_cases
            if case["actual_sequence_timing_trace_summary"]["actual_stop_in_near_optimal_band"] is True
        ),
        "backtest_actual_sequence_trace_unavailable_count": sum(
            1 for case in backtest_cases if case["actual_sequence_trace_error"] is not None
        ),
    }

    artifact = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "pre3",
        "timing_selection_tolerance_note": "One-stop recommendations now prefer the latest lap inside a 1.0s near-optimal band.",
        "representative_suite_pit_loss_value": round(pit_loss_value, 3),
        "backtest_pit_loss_value": round(backtest_pit_loss_value, 3),
        "aggregate_summary": aggregate,
        "representative_cases": representative_cases,
        "backtest_cases": backtest_cases,
    }

    output_path = ROOT / "data" / "processed" / "pre3_stop_timing_audit.json"
    _json_dump(output_path, artifact)

    print(f"Representative cases audited: {aggregate['representative_case_count']}")
    print(f"Flat timing curves: {aggregate['representative_flat_curve_count']}")
    print(f"Edge optima: {aggregate['representative_edge_optimum_count']}")
    print(f"Two-stop stress cases in suite: {aggregate['representative_two_stop_case_count']}")
    print(f"Backtest next-compound matches: {aggregate['backtest_actual_next_compound_match_count']}")
    print(f"Backtest actual stops inside near-optimal band: {aggregate['backtest_actual_stop_in_near_optimal_band_count']}")
    print(f"Backtest actual-sequence traces unavailable: {aggregate['backtest_actual_sequence_trace_unavailable_count']}")
    print(f"Artifact: {output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
