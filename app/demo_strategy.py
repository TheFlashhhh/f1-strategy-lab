"""Runnable demo for the complete F1 Strategy Lab with integrated Phase 1 stack.

This demo demonstrates the full Phase 1 pipeline:
- Phase 1A: Data loading (Miami historical + 2026 pre-Miami)
- Phase 1B: Fuel correction (fuel-load confound removal)
- Phase 1C: Degradation modeling (piecewise with cliff detection)

Result: Improved pit-timing strategy with transparent model reporting.

Run as: python app/demo_strategy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as: python app/demo_strategy.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_data
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.evaluate_degradation import evaluate_all_degradation
from src.simulation.strategy import (
    estimate_pit_loss_window,
    find_optimal_pit_lap,
    optimize_pit_window,
    recommend_action,
)


def main() -> None:
    """Run integrated Phase 1 demo."""
    print("\n" + "=" * 80)
    print("F1 STRATEGY LAB - INTEGRATED PHASE 1 PIPELINE")
    print("=" * 80)

    # Phase 1A: Load data
    print("\nPhase 1A: Loading data...")
    df_raw = load_data(dataset="miami_historical", project_root=ROOT)
    print(f"  Loaded {len(df_raw)} raw laps")

    df = select_relevant_columns(df_raw)
    df = detect_pit_stops(df)
    clean_df = clean_laps(df)
    model_df = build_model_df(clean_df)
    print(f"  Filtered to {len(model_df)} model-grade laps")

    # Phases 1B + 1C Integrated: Evaluate degradation with fuel correction + piecewise
    print("\nPhase 1B + 1C: Evaluating degradation (fuel correction + piecewise)...")
    deg_result = evaluate_all_degradation(
        model_df,
        use_fuel_correction=True,
        use_piecewise=True,
    )

    # Display model status
    print("\nModel Status (by compound):")
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        info = deg_result.get_model_info(compound)
        if info["model_type"]:
            print(
                f"  {compound:8s} ({info['samples']:4d} samples): "
                f"{info['model_type']:30s}"
                + (
                    f" [cliff at tyre-life {info['breakpoint_tyre_life']}]"
                    if info["is_piecewise"]
                    else ""
                )
            )

    # Estimate pit loss (uses raw lap times)
    print("\nEstimating pit loss...")
    pit_loss_samples = estimate_pit_loss_window(df)
    if len(pit_loss_samples) == 0:
        raise RuntimeError("No pit-loss samples were produced.")
    pit_loss_value = float(np.median(pit_loss_samples))
    print(f"  Pit-loss samples: {len(pit_loss_samples)}")
    print(f"  Median pit-loss: {pit_loss_value:.2f} s")

    # Optimize strategy
    print("\nOptimizing pit strategy...")
    current_tyre_life = 5
    laps_remaining = 25
    compound = "MEDIUM"
    
    strategy_df = optimize_pit_window(
        degradation_models=deg_result,  # Pass the unified result directly
        pit_loss_value=pit_loss_value,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        compound=compound,
    )
    optimal_pit_lap, optimal_total_time = find_optimal_pit_lap(strategy_df)

    print(f"  Example scenario: {compound} compound, tyre-life {current_tyre_life}, {laps_remaining} laps remaining")
    print(f"    Optimal pit lap: {optimal_pit_lap}")
    print(f"    Minimum total time: {optimal_total_time:.2f} s")

    decision = recommend_action(
        degradation_models=deg_result,
        pit_loss_value=pit_loss_value,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        compound=compound,
    )
    print(f"    Decision: {decision}")

    # Show example predictions
    print("\nExample lap-time predictions (MEDIUM compound, using active models):")
    for tyre_life in [1, 5, 10, 15, 20]:
        predicted = deg_result.predict_lap_time("MEDIUM", tyre_life)
        if predicted:
            print(f"  Tyre-life {tyre_life:2d}: {predicted:.2f} s")

    print("\n" + "=" * 80)
    print("PHASE 1 INTEGRATED DEMO COMPLETE")
    print("=" * 80)
    print("\nKey points:")
    print("  ✓ Phase 1A data loading")
    print("  ✓ Phase 1B fuel correction (automatic)")
    print("  ✓ Phase 1C piecewise degradation (with cliff detection)")
    print("  ✓ Unified prediction interface")
    print("  ✓ Transparent model reporting")


if __name__ == "__main__":
    main()

