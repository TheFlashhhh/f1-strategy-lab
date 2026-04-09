"""Phase 1C demo: Improved degradation modeling with cliff-aware piecewise fitting.

This script demonstrates Phase 1C's enhanced degradation modeling:
- Fits both linear (baseline) and piecewise (with cliff detection) models
- Uses fuel-corrected lap times by default (from Phase 1B)
- Shows before-vs-after degradation model comparison
- Detects tyre-wear cliffs (transitions to sharp wear)

Run as: python app/demo_phase1c.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as: python app/demo_phase1c.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_data
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.build_features import create_degradation_table, fit_degradation_models, models_to_table
from src.features.fuel_correction import apply_fuel_correction, estimate_fuel_effect
from src.features.degradation_modeling import (
    fit_all_piecewise_models,
    print_degradation_comparison,
    save_degradation_comparison,
)


def main() -> None:
    """Run Phase 1C degradation modeling demonstration."""
    print("\n" + "=" * 80)
    print("PHASE 1C: IMPROVED DEGRADATION MODELING")
    print("=" * 80)

    # Step 1: Load and preprocess data
    print("\nStep 1: Loading Phase 1A data...")
    df_raw = load_data(dataset="miami_historical", project_root=ROOT)
    print(f"  Loaded {len(df_raw)} total laps")

    df = select_relevant_columns(df_raw)
    df = detect_pit_stops(df)
    clean_df = clean_laps(df)
    model_df = build_model_df(clean_df)
    print(f"  Filtered to {len(model_df)} model-grade laps")

    # Step 2: Apply fuel correction (Phase 1B)
    print("\nStep 2: Applying fuel correction (Phase 1B)...")
    fuel_effects = estimate_fuel_effect(model_df)
    if not fuel_effects:
        print("  WARNING: Fuel correction failed, using raw lap times")
        corrected_df = model_df
    else:
        corrected_df = apply_fuel_correction(model_df, fuel_effects)
        print(f"  Applied fuel correction for {len(fuel_effects)} compound(s)")

    # Step 3: Fit baseline linear models
    print("\nStep 3: Fitting baseline linear degradation models...")
    raw_deg_table = create_degradation_table(model_df, use_fuel_corrected=False)
    linear_models_raw = fit_degradation_models(raw_deg_table)
    print("  Linear models (raw lap times):")
    print(models_to_table(linear_models_raw).to_string(index=False))

    # Step 4: Fit improved models using fuel-corrected times
    print("\nStep 4: Fitting improved degradation models (Phase 1C)...")
    print("  Using fuel-corrected lap times (when available)")

    piecewise_models, linear_models_corrected = fit_all_piecewise_models(
        corrected_df, use_fuel_corrected=True
    )

    print(f"  Fitted models for {len(piecewise_models)} compound(s)")

    # Step 5: Display comparison
    print("\nStep 5: Degradation model comparison:")
    print_degradation_comparison(piecewise_models, linear_models_corrected)

    # Step 6: Show detailed results for each compound
    print("\nStep 6: Detailed model results:")
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        if compound in piecewise_models:
            model = piecewise_models[compound]
            print(f"\n  {compound} Compound:")
            print(f"    Total samples: {model.total_samples}")

            if model.fell_back_to_linear:
                print(f"    Model: LINEAR (no clear cliff detected)")
                print(f"      Slope:     {model.pre_cliff_slope:.4f} s/lap")
                print(f"      Intercept: {model.pre_cliff_intercept:.2f} s")
                print(f"      RSS:       {model.rss_piecewise:.1f}")
            else:
                print(f"    Model: PIECEWISE (cliff detected)")
                print(f"      Cliff at tyre-life: {model.breakpoint_tyre_life}")
                print(f"      Pre-cliff sample:  {model.pre_cliff_samples} laps")
                print(f"        Slope:     {model.pre_cliff_slope:.4f} s/lap")
                print(f"        Intercept: {model.pre_cliff_intercept:.2f} s")
                print(f"      Post-cliff samples: {model.post_cliff_samples} laps")
                print(f"        Slope:     {model.post_cliff_slope:.4f} s/lap")
                print(f"        Intercept: {model.post_cliff_intercept:.2f} s")
                print(f"      Fit improvement: {model.improvement_percent:.1f}%")
                print(f"      RSS: Piecewise={model.rss_piecewise:.1f}, Linear={model.rss_linear:.1f}")

    # Step 7: Save comparison artifact
    print("\nStep 7: Saving degradation model comparison...")
    output_path = save_degradation_comparison(
        piecewise_models,
        linear_models_corrected,
        output_path=ROOT / "data" / "processed" / "degradation_model_comparison.json",
    )
    print(f"  Saved to {output_path}")

    # Step 8: Example: Show how piecewise model would be used in strategy
    print("\nStep 8: Example strategy application (MEDIUM compound):")
    if "MEDIUM" in piecewise_models:
        model = piecewise_models["MEDIUM"]
        current_tyre_life = 5

        if model.fell_back_to_linear:
            predicted_time = model.pre_cliff_slope * current_tyre_life + model.pre_cliff_intercept
            print(f"  Tyre life {current_tyre_life}: {predicted_time:.2f} s (linear model)")
        else:
            if current_tyre_life <= model.breakpoint_tyre_life:
                predicted_time = (
                    model.pre_cliff_slope * current_tyre_life + model.pre_cliff_intercept
                )
                print(
                    f"  Tyre life {current_tyre_life}: {predicted_time:.2f} s "
                    f"(pre-cliff, before tyre-life {model.breakpoint_tyre_life})"
                )
            else:
                predicted_time = (
                    model.post_cliff_slope * current_tyre_life + model.post_cliff_intercept
                )
                print(
                    f"  Tyre life {current_tyre_life}: {predicted_time:.2f} s "
                    f"(post-cliff, after tyre-life {model.breakpoint_tyre_life})"
                )

    print("\n" + "=" * 80)
    print("PHASE 1C DEGRADATION MODELING DEMO COMPLETE")
    print("=" * 80)
    print("\nKey outputs:")
    print("- Piecewise degradation models with cliff detection")
    print("- Comparison of linear vs piecewise fit quality")
    print("- Identified tyre-wear cliffs for each compound")
    print("- Saved comparison artifact: data/processed/degradation_model_comparison.json")
    print("\nNext steps:")
    print("- Review degradation_model_comparison.json for detailed results")
    print("- Use piecewise models in strategy optimization for improved predictions")
    print("- Validate results on hold-out race data")


if __name__ == "__main__":
    main()
