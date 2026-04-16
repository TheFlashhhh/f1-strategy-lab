#!/usr/bin/env python
"""Run Phase 2D broader validation / robustness evaluation.

This script builds the existing hybrid strategy pipeline, runs a compact
representative scenario suite through the Phase 2A engine plus Phase 2C
sensitivity checks, prints a concise summary, and saves inspectable artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import detect_pit_stops, select_relevant_columns
from src.data.loader import DataLoader
from src.features.hybrid_modeling import build_role_based_hybrid_model, load_or_build_hybrid_dataset
from src.simulation.strategy import estimate_pit_loss_window
from src.simulation.strategy_validation import (
    build_representative_scenario_suite,
    run_strategy_validation_suite,
    save_validation_artifacts,
    validation_results_to_frame,
)


def main() -> None:
    """Run the canonical Phase 2D validation workflow."""
    print("\n" + "=" * 88)
    print("PHASE 2D: BROADER VALIDATION / ROBUSTNESS EVALUATION")
    print("=" * 88)

    print("\nLoading hybrid dataset and building the strategy pipeline...")
    df_raw, hybrid_context = load_or_build_hybrid_dataset(project_root=ROOT)
    deg_result, _ = build_role_based_hybrid_model(project_root=ROOT)

    pit_loader = DataLoader(project_root=ROOT)
    pit_raw = pit_loader.load_data(dataset="miami_historical")
    pit_source_df = detect_pit_stops(select_relevant_columns(pit_raw))
    pit_loss_samples = estimate_pit_loss_window(pit_source_df)
    if len(pit_loss_samples) == 0:
        raise RuntimeError("No pit-loss samples were produced for Phase 2D validation.")
    pit_loss_value = float(np.median(pit_loss_samples))

    print(f"  Active pools: {len(hybrid_context.active_pools)}")
    print(f"  Source laps across active pools: {len(df_raw):,}")
    print(f"  Miami pit-loss baseline: {pit_loss_value:.2f}s")

    scenarios = build_representative_scenario_suite()
    print(f"\nRunning representative scenario suite ({len(scenarios)} scenarios)...")
    report = run_strategy_validation_suite(
        degradation_models=deg_result,
        pit_loss_value=pit_loss_value,
        scenarios=scenarios,
    )

    json_path = ROOT / "data" / "processed" / "phase2d_validation_summary.json"
    csv_path = ROOT / "data" / "processed" / "phase2d_validation_summary.csv"
    saved_paths = save_validation_artifacts(report, json_path=json_path, csv_path=csv_path)

    summary = report["aggregate_summary"]
    scenario_df = validation_results_to_frame(report)

    print("\nSummary:")
    print(f"  One-stop recommendations: {summary['strategy_type_counts'].get('one-stop', 0)}")
    print(f"  Two-stop recommendations: {summary['strategy_type_counts'].get('two-stop', 0)}")
    print(f"  Stable: {summary['stability_counts'].get('Stable', 0)}")
    print(f"  Moderately Sensitive: {summary['stability_counts'].get('Moderately Sensitive', 0)}")
    print(f"  Fragile: {summary['stability_counts'].get('Fragile', 0)}")
    print(f"  Best-plan infeasible warnings: {summary['warning_counts'].get('best-plan-infeasible', 0)}")
    print(f"  SOFT support tier: {summary['soft_compound_assessment'].get('support_tier')}")
    print(f"  SOFT weak-data signal: {summary['soft_compound_assessment']['weak_data_signal']}")

    print("\nRepresentative outcomes:")
    display_cols = [
        "scenario_id",
        "current_compound",
        "current_tyre_life",
        "laps_remaining",
        "best_strategy_type",
        "next_tyre",
        "pit_lap",
        "second_pit_lap",
        "estimated_total_time",
        "feasible",
        "stability_label",
    ]
    print(scenario_df[display_cols].to_string(index=False))

    if summary["pathological_cases"]:
        print("\nPathological / warning cases:")
        for case in summary["pathological_cases"]:
            print(
                f"  - {case['scenario_id']}: {case['issue']} | "
                f"{case['strategy_type']} -> {case['next_compound']} @ L{case['pit_lap']}"
            )

    print("\nObservations:")
    for observation in summary["observations"]:
        print(f"  - {observation}")

    print("\nArtifacts:")
    print(f"  - JSON: {saved_paths['json']}")
    print(f"  - CSV:  {saved_paths['csv']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
