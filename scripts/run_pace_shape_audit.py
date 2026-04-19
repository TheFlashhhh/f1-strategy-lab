#!/usr/bin/env python
"""Run a Pre-Phase-3 pace-shape audit for the remaining real backtest misses."""

from __future__ import annotations

from collections import Counter
import json
import sys
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
from src.features.fuel_correction import apply_fuel_correction, estimate_fuel_effect
from src.features.hybrid_modeling import build_role_based_hybrid_model
from src.simulation.strategy import estimate_pit_loss_window, predict_lap_time
from src.simulation.strategy_engine import StrategyPlan, build_strategy_timing_trace, recommend_best_strategy
from src.simulation.strategy_validation import (
    build_representative_scenario_suite,
    decompose_strategy_plan,
)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _json_load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dict_to_plan(plan_dict: dict) -> StrategyPlan:
    """Convert a serialized plan dict back into a StrategyPlan."""
    return StrategyPlan(
        strategy_type=plan_dict["strategy_type"],
        current_compound=plan_dict["current_compound"],
        current_tyre_life=plan_dict["current_tyre_life"],
        next_compound=plan_dict["next_compound"],
        final_compound=plan_dict["final_compound"],
        pit_lap=plan_dict["pit_lap"],
        second_pit_lap=plan_dict["second_pit_lap"],
        total_race_time=plan_dict["total_race_time"],
        feasible=plan_dict["feasible"],
        feasibility_reason=plan_dict["feasibility_reason"],
        explanation=plan_dict["explanation"],
    )


def _predicted_stint_shape(
    degradation_models,
    compound: str,
    starting_tyre_life: int,
    laps: int,
) -> dict:
    """Return predicted lap-time shape for one stint."""
    lap_rows = []
    for offset in range(laps):
        tyre_life = starting_tyre_life + offset
        lap_time = predict_lap_time(degradation_models, compound, tyre_life)
        if lap_time is None:
            raise ValueError(
                f"Missing prediction for {compound} at tyre life {tyre_life} while building pace-shape audit."
            )
        lap_rows.append(
            {
                "lap_in_stint": offset + 1,
                "tyre_life": tyre_life,
                "predicted_lap_time_s": round(float(lap_time), 3),
            }
        )

    first_lap = lap_rows[0]["predicted_lap_time_s"] if lap_rows else None
    last_lap = lap_rows[-1]["predicted_lap_time_s"] if lap_rows else None
    return {
        "compound": compound,
        "starting_tyre_life": starting_tyre_life,
        "laps": laps,
        "first_lap_time_s": first_lap,
        "last_lap_time_s": last_lap,
        "predicted_delta_s": round(last_lap - first_lap, 3) if first_lap is not None and last_lap is not None else None,
        "predicted_average_lap_time_s": round(float(np.mean([row["predicted_lap_time_s"] for row in lap_rows])), 3) if lap_rows else None,
        "trace": lap_rows,
    }


def _plan_pace_shape(
    degradation_models,
    pit_loss_value: float,
    plan_dict: dict,
    laps_remaining: int,
) -> dict:
    """Return a pace-shape view of a serialized strategy plan."""
    plan = _dict_to_plan(plan_dict)
    breakdown = decompose_strategy_plan(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        plan=plan,
        laps_remaining=laps_remaining,
    )

    stint_shapes = []
    for stint in breakdown["stints"]:
        stint_shapes.append(
            {
                **stint,
                **_predicted_stint_shape(
                    degradation_models=degradation_models,
                    compound=stint["compound"],
                    starting_tyre_life=stint["starting_tyre_life"],
                    laps=stint["laps"],
                ),
            }
        )

    return {
        "strategy_type": plan.strategy_type,
        "pit_stop_count": breakdown["pit_stop_count"],
        "pit_loss_total_s": breakdown["pit_loss_total_s"],
        "stint_time_total_s": breakdown["stint_time_total_s"],
        "reconstructed_total_s": breakdown["reconstructed_total_s"],
        "stints": stint_shapes,
    }


def _observed_remaining_stints(
    corrected_driver_df: pd.DataFrame,
    checkpoint_lap: int,
    current_compound: str,
    actual_plan: dict,
) -> list[dict]:
    """Return observed clean-lap summaries for the actual remaining stints."""
    first_end = checkpoint_lap + actual_plan["pit_lap"] - 1
    stint_windows = [
        {
            "label": "current_stint",
            "compound": current_compound,
            "start_lap": checkpoint_lap,
            "end_lap": first_end,
        }
    ]

    if actual_plan["final_compound"] is None:
        finish_start = first_end + 1
        finish_end = int(corrected_driver_df["LapNumber"].max())
        stint_windows.append(
            {
                "label": "finish_stint",
                "compound": actual_plan["next_compound"],
                "start_lap": finish_start,
                "end_lap": finish_end,
            }
        )
    else:
        second_end = checkpoint_lap + actual_plan["second_pit_lap"] - 1
        stint_windows.extend(
            [
                {
                    "label": "middle_stint",
                    "compound": actual_plan["next_compound"],
                    "start_lap": first_end + 1,
                    "end_lap": second_end,
                },
                {
                    "label": "finish_stint",
                    "compound": actual_plan["final_compound"],
                    "start_lap": second_end + 1,
                    "end_lap": int(corrected_driver_df["LapNumber"].max()),
                },
            ]
        )

    observed = []
    for stint in stint_windows:
        raw_rows = corrected_driver_df[
            (corrected_driver_df["LapNumber"] >= stint["start_lap"])
            & (corrected_driver_df["LapNumber"] <= stint["end_lap"])
            & (corrected_driver_df["Compound"] == stint["compound"])
        ].copy()
        clean_rows = raw_rows[
            raw_rows["TrackStatus"].astype(str) == "1"
        ].copy()
        clean_rows = clean_rows[clean_rows["Deleted"] == False].copy()
        clean_rows = clean_rows[clean_rows["PitInTime"].isna()].copy()
        clean_rows = clean_rows[clean_rows["PitOutTime"].isna()].copy()

        if clean_rows.empty:
            observed.append(
                {
                    **stint,
                    "observed_clean_lap_count": 0,
                    "observed_first_lap_time_s": None,
                    "observed_last_lap_time_s": None,
                    "observed_delta_s": None,
                    "observed_average_lap_time_s": None,
                    "clean_trace": [],
                }
            )
            continue

        clean_trace = [
            {
                "lap_number": int(row.LapNumber),
                "tyre_life": int(row.TyreLife),
                "fuel_corrected_lap_time_s": round(float(row.FuelCorrectedLapTime), 3),
            }
            for row in clean_rows.itertuples(index=False)
        ]
        first_lap = clean_trace[0]["fuel_corrected_lap_time_s"]
        last_lap = clean_trace[-1]["fuel_corrected_lap_time_s"]
        observed.append(
            {
                **stint,
                "observed_clean_lap_count": len(clean_trace),
                "observed_first_lap_time_s": first_lap,
                "observed_last_lap_time_s": last_lap,
                "observed_delta_s": round(last_lap - first_lap, 3),
                "observed_average_lap_time_s": round(float(np.mean([row["fuel_corrected_lap_time_s"] for row in clean_trace])), 3),
                "clean_trace": clean_trace,
            }
        )

    return observed


def _load_backtest_artifact() -> dict:
    """Load the latest backtest diagnostics artifact."""
    path = ROOT / "data" / "processed" / "pre3_backtest_diagnostics.json"
    if not path.exists():
        raise RuntimeError(
            "Backtest diagnostics artifact is missing. Run python scripts/run_pre3_backtest.py first."
        )
    return _json_load(path)


def _load_representative_probe_ids() -> list[str]:
    """Return a compact set of representative Phase 2D probe cases."""
    path = ROOT / "data" / "processed" / "phase2d_validation_summary.json"
    if path.exists():
        summary = _json_load(path)
        aggregate = summary.get("aggregate_summary", {})
        probe_ids = [
            item["scenario_id"]
            for item in aggregate.get("pathological_cases", [])
        ]
        probe_ids.extend(aggregate.get("strategy_mix_analysis", {}).get("two_stop_selected_scenarios", []))
        deduped = []
        for scenario_id in probe_ids:
            if scenario_id not in deduped:
                deduped.append(scenario_id)
        if deduped:
            return deduped

    return [
        "medium_high_short",
        "hard_medium_short",
        "soft_extreme_long",
        "hard_extreme_long",
    ]


def _pace_shape_issue_summary(
    checkpoint: dict,
    model_shape: dict,
    actual_shape: Optional[dict],
    observed_actual_stints: list[dict],
) -> dict:
    """Return a practical root-cause summary for a real miss."""
    same_strategy_type = checkpoint["model_best"]["strategy_type"] == checkpoint["actual_remaining_strategy"]["strategy_type"]
    same_compounds = (
        checkpoint["model_best"]["next_compound"] == checkpoint["actual_remaining_strategy"]["next_compound"]
        and checkpoint["model_best"]["final_compound"] == checkpoint["actual_remaining_strategy"]["final_compound"]
    )
    current_observed = next((stint for stint in observed_actual_stints if stint["label"] == "current_stint"), None)
    current_model_actual = (
        next((stint for stint in actual_shape["stints"] if stint["label"] == "current_stint"), None)
        if actual_shape is not None
        else None
    )

    observed_delta = current_observed.get("observed_delta_s") if current_observed else None
    predicted_delta = current_model_actual.get("predicted_delta_s") if current_model_actual else None
    model_too_flat = (
        observed_delta is not None
        and predicted_delta is not None
        and observed_delta - predicted_delta > 0.5
    )
    actual_trace_available = checkpoint["actual_sequence_timing_context"].get("trace_available", True)
    actual_in_band = checkpoint["actual_sequence_timing_context"].get("actual_first_stop_in_near_optimal_band")
    first_stop_error = checkpoint.get("absolute_first_stop_timing_error_laps")

    if not actual_trace_available:
        dominant_issue = "pace-shape / compound mismatch"
        mismatch_source = "future compound path"
        explanation = (
            "The actual remaining strategy is not even feasible under the current model assumptions, "
            "so the miss is stronger than a timing-only disagreement."
        )
    elif not same_strategy_type and not same_compounds:
        dominant_issue = "multiple: structure + compound mismatch"
        mismatch_source = "compound transition and pit structure"
        explanation = "The model disagrees on both the number of stops and the future compound path."
    elif not same_compounds:
        dominant_issue = "compound-choice mismatch"
        mismatch_source = "compound transition"
        explanation = "The model picks a different future compound path even before timing is considered."
    elif first_stop_error is not None and actual_in_band is not True:
        dominant_issue = "stop-timing mismatch"
        mismatch_source = "late current stint"
        explanation = "The future compound call is broadly aligned, but the model wants a materially different first-stop lap."
    else:
        dominant_issue = "multiple"
        mismatch_source = "mixed"
        explanation = "The miss combines several smaller disagreements rather than one clean failure mode."

    if model_too_flat:
        dominant_issue = f"pace-shape + {dominant_issue}"
        mismatch_source = "late current stint / transition"
        explanation += " The observed current-stint pace falls away faster than the model's predicted current-stint shape."

    return {
        "dominant_issue": dominant_issue,
        "mismatch_source": mismatch_source,
        "model_current_stint_too_flat_vs_observed": model_too_flat,
        "observed_current_stint_delta_s": observed_delta,
        "predicted_current_stint_delta_s": predicted_delta,
        "explanation": explanation,
    }


def _real_failure_audit_entries() -> tuple[list[dict], float]:
    """Build detailed pace-shape entries for the remaining real backtest misses."""
    diagnostics = _load_backtest_artifact()
    real_failures = [
        checkpoint
        for checkpoint in diagnostics["checkpoints"]
        if checkpoint["failure_mode"] not in {"exact match", "effectively near-equivalent despite rank miss"}
    ]

    loader = DataLoader(project_root=ROOT)
    miami_raw = loader.load_data("miami_historical")
    train_raw = miami_raw[miami_raw["season"] < 2024].copy()
    holdout_raw = miami_raw[miami_raw["season"] == 2024].copy()

    train_selected = detect_pit_stops(select_relevant_columns(train_raw))
    train_model_df = build_model_df(clean_laps(train_selected))
    train_models = evaluate_all_degradation(train_model_df, use_fuel_correction=True, use_piecewise=True)
    fuel_effects = estimate_fuel_effect(train_model_df)
    holdout_selected = detect_pit_stops(select_relevant_columns(holdout_raw))
    holdout_corrected = apply_fuel_correction(holdout_selected, fuel_effects)
    pit_loss_value = float(np.median(estimate_pit_loss_window(train_selected)))

    entries = []
    for checkpoint in real_failures:
        driver_df = holdout_corrected[holdout_corrected["Driver"] == checkpoint["driver"]].copy()
        model_shape = _plan_pace_shape(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            plan_dict=checkpoint["model_best"],
            laps_remaining=checkpoint["laps_remaining"],
        )
        actual_shape = _plan_pace_shape(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            plan_dict={
                **checkpoint["model_best"],
                "strategy_type": checkpoint["actual_remaining_strategy"]["strategy_type"],
                "next_compound": checkpoint["actual_remaining_strategy"]["next_compound"],
                "final_compound": checkpoint["actual_remaining_strategy"]["final_compound"],
                "pit_lap": checkpoint["actual_remaining_strategy"]["pit_lap"],
                "second_pit_lap": checkpoint["actual_remaining_strategy"]["second_pit_lap"],
                "total_race_time": checkpoint["actual_strategy_estimated_time"],
                "feasible": True,
                "feasibility_reason": "Actual remaining strategy reconstructed from held-out race stints.",
                "explanation": "Actual remaining strategy",
            },
            laps_remaining=checkpoint["laps_remaining"],
        ) if checkpoint["actual_strategy_estimated_time"] is not None else None
        observed_actual_stints = _observed_remaining_stints(
            corrected_driver_df=driver_df,
            checkpoint_lap=checkpoint["checkpoint_lap"],
            current_compound=checkpoint["current_compound"],
            actual_plan=checkpoint["actual_remaining_strategy"],
        )
        issue_summary = _pace_shape_issue_summary(
            checkpoint=checkpoint,
            model_shape=model_shape,
            actual_shape=actual_shape,
            observed_actual_stints=observed_actual_stints,
        )
        entries.append(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "season": checkpoint.get("season"),
                "driver": checkpoint["driver"],
                "checkpoint_lap": checkpoint["checkpoint_lap"],
                "current_state": {
                    "current_compound": checkpoint["current_compound"],
                    "current_tyre_life": checkpoint["current_tyre_life"],
                    "laps_remaining": checkpoint["laps_remaining"],
                },
                "failure_mode": checkpoint["failure_mode"],
                "model_best": checkpoint["model_best"],
                "actual_remaining_strategy": checkpoint["actual_remaining_strategy"],
                "delta_model_best_vs_actual_s": checkpoint["delta_model_best_vs_actual_s"],
                "absolute_first_stop_timing_error_laps": checkpoint["absolute_first_stop_timing_error_laps"],
                "actual_sequence_trace_available": checkpoint["actual_sequence_timing_context"].get("trace_available", True),
                "actual_sequence_trace_error": checkpoint["actual_sequence_timing_context"].get("trace_error"),
                "model_best_pace_shape": model_shape,
                "actual_strategy_pace_shape": actual_shape,
                "observed_actual_stints": observed_actual_stints,
                "issue_summary": issue_summary,
            }
        )

    return entries, pit_loss_value


def _representative_probe_entries() -> tuple[list[dict], float]:
    """Build pace-shape probes for a compact representative validation slice."""
    probe_ids = _load_representative_probe_ids()
    scenarios = {scenario.scenario_id: scenario for scenario in build_representative_scenario_suite()}

    loader = DataLoader(project_root=ROOT)
    pit_raw = loader.load_data("miami_historical")
    deg_result, _ = build_role_based_hybrid_model(project_root=ROOT)
    pit_loss_value = float(np.median(estimate_pit_loss_window(detect_pit_stops(select_relevant_columns(pit_raw)))))

    entries = []
    for scenario_id in probe_ids:
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            continue
        best_plan, ranked_plans = recommend_best_strategy(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=scenario.current_compound,
            current_tyre_life=scenario.current_tyre_life,
            laps_remaining=scenario.laps_remaining,
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            include_two_stop=True,
        )
        runner_up = ranked_plans[1] if len(ranked_plans) > 1 else None
        timing_trace = build_strategy_timing_trace(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=scenario.current_compound,
            current_tyre_life=scenario.current_tyre_life,
            laps_remaining=scenario.laps_remaining,
            next_compound=best_plan.next_compound,
            final_compound=best_plan.final_compound,
        )
        entry = {
            "scenario_id": scenario.scenario_id,
            "current_state": {
                "current_compound": scenario.current_compound,
                "current_tyre_life": scenario.current_tyre_life,
                "laps_remaining": scenario.laps_remaining,
            },
            "rationale": scenario.rationale,
            "best_plan": {
                "strategy_type": best_plan.strategy_type,
                "next_compound": best_plan.next_compound,
                "final_compound": best_plan.final_compound,
                "pit_lap": best_plan.pit_lap,
                "second_pit_lap": best_plan.second_pit_lap,
                "total_race_time": round(float(best_plan.total_race_time), 3),
            },
            "runner_up_gap_s": round(float(runner_up.total_race_time - best_plan.total_race_time), 3) if runner_up else None,
            "timing_trace_summary": {
                "curve_shape": timing_trace["curve_shape"],
                "best_on_window_edge": timing_trace["best_on_window_edge"],
                "near_optimal_band_laps": timing_trace["near_optimal_band_laps"],
            },
            "best_plan_pace_shape": _plan_pace_shape(
                degradation_models=deg_result,
                pit_loss_value=pit_loss_value,
                plan_dict={
                    "strategy_type": best_plan.strategy_type,
                    "current_compound": best_plan.current_compound,
                    "current_tyre_life": best_plan.current_tyre_life,
                    "next_compound": best_plan.next_compound,
                    "final_compound": best_plan.final_compound,
                    "pit_lap": best_plan.pit_lap,
                    "second_pit_lap": best_plan.second_pit_lap,
                    "total_race_time": best_plan.total_race_time,
                    "feasible": best_plan.feasible,
                    "feasibility_reason": best_plan.feasibility_reason,
                    "explanation": best_plan.explanation,
                },
                laps_remaining=scenario.laps_remaining,
            ),
            "runner_up_pace_shape": None,
            "support_summary": {
                "best_next_support": deg_result.get_support_info(best_plan.next_compound),
                "best_final_support": deg_result.get_support_info(best_plan.final_compound) if best_plan.final_compound else None,
            },
        }
        if runner_up is not None:
            entry["runner_up_pace_shape"] = _plan_pace_shape(
                degradation_models=deg_result,
                pit_loss_value=pit_loss_value,
                plan_dict={
                    "strategy_type": runner_up.strategy_type,
                    "current_compound": runner_up.current_compound,
                    "current_tyre_life": runner_up.current_tyre_life,
                    "next_compound": runner_up.next_compound,
                    "final_compound": runner_up.final_compound,
                    "pit_lap": runner_up.pit_lap,
                    "second_pit_lap": runner_up.second_pit_lap,
                    "total_race_time": runner_up.total_race_time,
                    "feasible": runner_up.feasible,
                    "feasibility_reason": runner_up.feasibility_reason,
                    "explanation": runner_up.explanation,
                },
                laps_remaining=scenario.laps_remaining,
            )
        entries.append(entry)

    return entries, pit_loss_value


def main() -> None:
    print("\n" + "=" * 88)
    print("PRE-3 PACE-SHAPE AUDIT")
    print("=" * 88)

    real_failure_entries, backtest_pit_loss_value = _real_failure_audit_entries()
    representative_entries, representative_pit_loss_value = _representative_probe_entries()

    issue_counts = Counter(
        entry["issue_summary"]["dominant_issue"]
        for entry in real_failure_entries
    )
    aggregate = {
        "real_failure_count": len(real_failure_entries),
        "dominant_issue_counts": dict(issue_counts),
        "model_current_stint_too_flat_count": sum(
            1
            for entry in real_failure_entries
            if entry["issue_summary"]["model_current_stint_too_flat_vs_observed"]
        ),
        "actual_sequence_trace_unavailable_count": sum(
            1
            for entry in real_failure_entries
            if not entry["actual_sequence_trace_available"]
        ),
        "long_hard_repeat_case_count": sum(
            1
            for entry in real_failure_entries
            if entry["current_state"]["current_compound"] == "HARD"
        ),
        "representative_probe_count": len(representative_entries),
    }

    artifact = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "pre3",
        "backtest_pit_loss_value": round(backtest_pit_loss_value, 3),
        "representative_pit_loss_value": round(representative_pit_loss_value, 3),
        "aggregate_summary": aggregate,
        "real_backtest_failures": real_failure_entries,
        "representative_probes": representative_entries,
    }

    output_path = ROOT / "data" / "processed" / "pre3_pace_shape_audit.json"
    _json_dump(output_path, artifact)

    print(f"Real failures audited: {aggregate['real_failure_count']}")
    print(f"Model-too-flat current-stint cases: {aggregate['model_current_stint_too_flat_count']}")
    print(f"Actual-sequence traces unavailable in-model: {aggregate['actual_sequence_trace_unavailable_count']}")
    print(f"Repeated long-HARD cases: {aggregate['long_hard_repeat_case_count']}")
    print(f"Representative probes: {aggregate['representative_probe_count']}")
    print(f"Artifact: {output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
