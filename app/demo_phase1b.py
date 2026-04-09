"""Phase 1B demo: Fuel correction for lap-time modeling.

This script demonstrates the fuel-correction layer added in Phase 1B,
showing before-vs-after degradation models and evaluating the correction.

Run as: python app/demo_phase1b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as: python app/demo_phase1b.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_data
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.build_features import create_degradation_table, fit_degradation_models, models_to_table
from src.features.fuel_correction import (
    apply_fuel_correction,
    estimate_fuel_effect,
    evaluate_fuel_correction,
    print_fuel_correction_summary,
    save_fuel_correction_summary,
)


def main() -> None:
    """Run Phase 1B fuel correction demonstration."""
    print("\n" + "=" * 70)
    print("PHASE 1B: FUEL CORRECTION DEMO")
    print("=" * 70)

    # Step 1: Load and preprocess data (same as Phase 1A)
    print("\nStep 1: Loading Phase 1A data...")
    df_raw = load_data(dataset="miami_historical", project_root=ROOT)
    print(f"  Loaded {len(df_raw)} total laps")

    df = select_relevant_columns(df_raw)
    df = detect_pit_stops(df)
    clean_df = clean_laps(df)
    model_df = build_model_df(clean_df)
    print(f"  Filtered to {len(model_df)} model-grade laps (pit-excluded, accurate, green-flag)")

    # Step 2: Estimate fuel effect (Phase 1B)
    print("\nStep 2: Estimating fuel-burn effect...")
    fuel_effects = estimate_fuel_effect(model_df)

    if not fuel_effects:
        print("  ERROR: Failed to estimate fuel effects for any compound")
        return

    print(f"  Estimated fuel effects for {len(fuel_effects)} compound(s)")

    # Step 3: Apply fuel correction (Phase 1B)
    print("\nStep 3: Applying fuel correction...")
    corrected_df = apply_fuel_correction(model_df, fuel_effects)
    print(f"  Added FuelCorrectedLapTime column")

    # Step 4: Compare degradation models (before vs after)
    print("\nStep 4: Evaluating fuel correction impact...")
    evaluation = evaluate_fuel_correction(df_raw, model_df, fuel_effects)

    # Step 5: Display results
    print_fuel_correction_summary(evaluation)

    # Step 6: Fit corrected degradation models for reference
    print("\nStep 5: Fitting corrected degradation models...")
    corrected_model_deg_df = create_degradation_table(corrected_df, use_fuel_corrected=True)
    corrected_degradation_models = fit_degradation_models(corrected_model_deg_df)
    print("  Corrected degradation models:")
    print(models_to_table(corrected_degradation_models).to_string(index=False))

    # Step 7: Save summary
    print("\nStep 6: Saving fuel correction summary...")
    output_path = save_fuel_correction_summary(evaluation, output_path=ROOT / "data" / "processed" / "fuel_correction_summary.json")
    print(f"  Saved to {output_path}")

    # Step 8: Example comparison for one compound
    print("\nStep 7: Example lap-time corrections (MEDIUM compound):")
    medium_laps = corrected_df[corrected_df["Compound"] == "MEDIUM"].head(10).copy()
    if len(medium_laps) > 0:
        medium_laps["Correction"] = medium_laps["LapTime"] - medium_laps["FuelCorrectedLapTime"]
        print(
            medium_laps[["Driver", "LapNumber", "LapTime", "FuelCorrectedLapTime", "Correction", "TyreLife"]]
            .to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("PHASE 1B FUEL CORRECTION DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("- Review data/processed/fuel_correction_summary.json for detailed evaluation")
    print("- Run app/demo_strategy.py to see baseline strategy (still works as before)")
    print("- Update degradation models in strategy.py to use FuelCorrectedLapTime if desired")


if __name__ == "__main__":
    main()
