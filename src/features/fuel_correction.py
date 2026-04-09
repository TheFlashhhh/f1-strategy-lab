"""Phase 1B: Fuel correction for removing fuel-load confound from lap times.

This module estimates the time advantage from decreasing fuel load as a race
progresses, then applies a correction to normalize lap times to a common fuel
reference point (e.g., full fuel tank at race start).

Assumption: Lap time improves approximately linearly with race progress due to
fuel burn. We fit a simple model: LapTime = base_time + fuel_effect * race_progress.
The fuel effect is estimated separately per compound on representative laps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_fuel_effect(df: pd.DataFrame) -> Dict[str, Tuple[float, int]]:
    """Estimate fuel-burn time effect across a full race.

    Uses model-grade laps (pit-excluded, accurate, green-flag) to estimate how much
    faster each compound becomes as fuel burns. Fits a linear relationship between
    normalized race progress (0–1) and lap time for each compound.

    Assumption: Lap time = base + fuel_effect * normalized_race_progress
    (where normalized_race_progress goes from 0 to 1 across the race)

    Args:
        df: DataFrame with columns [Compound, LapNumber, LapTime, Driver].
            Expected to be pre-filtered to model-grade laps.

    Returns:
        Dict mapping compound → (fuel_effect_s_per_race, sample_count)
        fuel_effect_s_per_race: Seconds of time improvement across the full race.
        Negative value means lap time improves as race progresses (fuel burns).
    """
    fuel_effects = {}

    for compound in df["Compound"].unique():
        compound_df = df[df["Compound"] == compound].copy()

        if len(compound_df) < 10:
            logger.warning(f"Compound {compound}: only {len(compound_df)} samples, skipping")
            continue

        # Normalize lap number to 0–1 range (race progress) per driver
        compound_df["RaceProgress"] = 0.0
        for driver in compound_df["Driver"].unique():
            driver_mask = compound_df["Driver"] == driver
            driver_data = compound_df.loc[driver_mask, "LapNumber"]
            if len(driver_data) > 1:
                min_lap = driver_data.min()
                max_lap = driver_data.max()
                # Normalize to 0-1
                compound_df.loc[driver_mask, "RaceProgress"] = (
                    (compound_df.loc[driver_mask, "LapNumber"] - min_lap) / (max_lap - min_lap)
                )

        # Fit: LapTime ~ RaceProgress (exclude intercept bias by centering)
        valid_mask = compound_df["LapTime"].notna() & compound_df["RaceProgress"].notna()
        valid_data = compound_df.loc[valid_mask].copy()

        if len(valid_data) < 10:
            logger.warning(f"Compound {compound}: insufficient valid samples after filtering")
            continue

        # Fit linear model: slope of LapTime vs. RaceProgress
        # Negative slope = improvement as race progresses (fuel burn effect)
        # The slope is in units of seconds per unit race progress [0-1], i.e., seconds across full race
        try:
            slope, _ = np.polyfit(valid_data["RaceProgress"], valid_data["LapTime"], 1)
            fuel_effects[compound] = (float(slope), len(valid_data))
            logger.info(
                f"Compound {compound}: fuel effect = {slope:.4f} s/race (total improvement over full race), "
                f"{len(valid_data)} laps"
            )
        except Exception as e:
            logger.warning(f"Compound {compound}: failed to fit fuel effect: {e}")

    return fuel_effects


def apply_fuel_correction(
    df: pd.DataFrame,
    fuel_effects: Dict[str, Tuple[float, int]],
) -> pd.DataFrame:
    """Apply fuel correction to create FuelCorrectedLapTime column.

    Normalizes lap times to a reference fuel level (assumed full tank at start,
    normalized_progress = 0). For each lap:

        FuelCorrectedLapTime = LapTime - (fuel_effect * race_progress)

    This removes the time advantage from fuel burn.

    Args:
        df: DataFrame with [Compound, LapNumber, LapTime, Driver].
        fuel_effects: Dict from estimate_fuel_effect().

    Returns:
        DataFrame with new column FuelCorrectedLapTime.
    """
    df = df.copy()
    df["FuelCorrectedLapTime"] = df["LapTime"].copy()  # Default to raw if no correction

    for compound, (fuel_effect, _) in fuel_effects.items():
        compound_mask = df["Compound"] == compound

        # Normalize lap number to race progress per driver
        for driver in df.loc[compound_mask, "Driver"].unique():
            driver_mask = compound_mask & (df["Driver"] == driver)
            driver_laps = df.loc[driver_mask, "LapNumber"]

            if len(driver_laps) > 1:
                min_lap = driver_laps.min()
                max_lap = driver_laps.max()
                race_progress = (df.loc[driver_mask, "LapNumber"] - min_lap) / (max_lap - min_lap)

                # Apply correction
                correction = fuel_effect * race_progress
                df.loc[driver_mask, "FuelCorrectedLapTime"] = (
                    df.loc[driver_mask, "LapTime"] - correction
                )

    return df


def evaluate_fuel_correction(
    raw_laps: pd.DataFrame,
    model_laps: pd.DataFrame,
    fuel_effects: Dict[str, Tuple[float, int]],
) -> Dict:
    """Evaluate before-vs-after degradation models with and without fuel correction.

    Fits linear degradation models on both raw and fuel-corrected lap times,
    then compares the slopes and intercepts.

    Args:
        raw_laps: All laps (for context).
        model_laps: Model-grade laps (pit-excluded, accurate, green-flag).
        fuel_effects: Dict from estimate_fuel_effect().

    Returns:
        Dict with before/after comparison by compound.
    """
    # Apply fuel correction to model laps
    corrected_laps = apply_fuel_correction(model_laps, fuel_effects)

    evaluation = {
        "method": "Linear degradation: LapTime ~ TyreLife",
        "fuel_correction_applied": True,
        "raw_sample_count": len(raw_laps),
        "model_sample_count": len(model_laps),
        "fuel_effects": {
            compound: {
                "coefficient_s_per_full_race": coeff,
                "interpretation": "seconds of total lap-time improvement over the entire race due to fuel burn",
                "sample_count": count,
            }
            for compound, (coeff, count) in fuel_effects.items()
        },
        "degradation_comparison": {},
    }

    for compound in model_laps["Compound"].unique():
        compound_mask_raw = model_laps["Compound"] == compound
        compound_raw = model_laps.loc[compound_mask_raw]
        compound_corrected = corrected_laps.loc[compound_mask_raw]

        if len(compound_raw) < 5:
            logger.warning(f"Compound {compound}: insufficient samples for comparison")
            continue

        # Fit raw degradation
        try:
            raw_slope, raw_intercept = np.polyfit(
                compound_raw["TyreLife"],
                compound_raw["LapTime"],
                1,
            )
        except Exception as e:
            logger.warning(f"Compound {compound}: failed to fit raw degradation: {e}")
            raw_slope, raw_intercept = np.nan, np.nan

        # Fit corrected degradation
        try:
            corrected_slope, corrected_intercept = np.polyfit(
                compound_corrected["TyreLife"],
                compound_corrected["FuelCorrectedLapTime"],
                1,
            )
        except Exception as e:
            logger.warning(f"Compound {compound}: failed to fit corrected degradation: {e}")
            corrected_slope, corrected_intercept = np.nan, np.nan

        # Compare
        slope_change = corrected_slope - raw_slope if not np.isnan(raw_slope) else np.nan
        intercept_change = corrected_intercept - raw_intercept if not np.isnan(raw_intercept) else np.nan

        evaluation["degradation_comparison"][compound] = {
            "raw_slope_s_per_lap": float(raw_slope) if not np.isnan(raw_slope) else None,
            "raw_intercept_s": float(raw_intercept) if not np.isnan(raw_intercept) else None,
            "corrected_slope_s_per_lap": float(corrected_slope) if not np.isnan(corrected_slope) else None,
            "corrected_intercept_s": float(corrected_intercept) if not np.isnan(corrected_intercept) else None,
            "slope_change_s_per_lap": float(slope_change) if not np.isnan(slope_change) else None,
            "intercept_change_s": float(intercept_change) if not np.isnan(intercept_change) else None,
            "slope_change_percent": (
                (slope_change / raw_slope * 100) if not np.isnan(raw_slope) and raw_slope != 0 else None
            ),
            "sample_count": len(compound_raw),
        }

    return evaluation


def save_fuel_correction_summary(evaluation: Dict, output_path: Path | str = "data/processed/fuel_correction_summary.json") -> Path:
    """Save fuel correction evaluation to JSON.

    Args:
        evaluation: Dict from evaluate_fuel_correction().
        output_path: Where to save the summary.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all values are JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        return obj

    evaluation_serializable = make_serializable(evaluation)

    with open(output_path, "w") as f:
        json.dump(evaluation_serializable, f, indent=2)

    logger.info(f"Fuel correction summary saved to {output_path}")
    return output_path


def print_fuel_correction_summary(evaluation: Dict) -> None:
    """Print a human-readable fuel correction summary."""
    print("\n" + "=" * 70)
    print("PHASE 1B: FUEL CORRECTION EVALUATION")
    print("=" * 70)
    print(f"Method: {evaluation['method']}")
    print(f"Total samples: {evaluation['raw_sample_count']}")
    print(f"Model-grade samples: {evaluation['model_sample_count']}")

    print("\nEstimated Fuel Effects (by compound):")
    print("(Negative values = faster as race progresses due to fuel burn)")
    for compound, fuel_data in evaluation["fuel_effects"].items():
        coeff = fuel_data["coefficient_s_per_full_race"]
        count = fuel_data["sample_count"]
        direction = "improvement over race" if coeff < 0 else "penalty over race"
        print(
            f"  {compound:8s}: {coeff:7.4f} s/race ({direction}, {count} laps used for estimation)"
        )

    print("\nDegradation Models Before vs After Fuel Correction:")
    print(f"{'Compound':<10} {'Raw Slope':<15} {'Corrected Slope':<18} {'Slope Change':<15} {'% Change':<10}")
    print("-" * 70)

    # Sort compounds for consistent output (SOFT, MEDIUM, HARD)
    compound_order = ["SOFT", "MEDIUM", "HARD"]
    for compound in compound_order:
        if compound not in evaluation["degradation_comparison"]:
            continue
        comp_data = evaluation["degradation_comparison"][compound]
        raw_slope = comp_data["raw_slope_s_per_lap"]
        corr_slope = comp_data["corrected_slope_s_per_lap"]
        slope_change = comp_data["slope_change_s_per_lap"]
        pct_change = comp_data["slope_change_percent"]

        raw_str = f"{raw_slope:.4f} s/lap" if raw_slope is not None else "N/A"
        corr_str = f"{corr_slope:.4f} s/lap" if corr_slope is not None else "N/A"
        change_str = f"{slope_change:.4f} s/lap" if slope_change is not None else "N/A"
        pct_str = f"{pct_change:.1f}%" if pct_change is not None else "N/A"

        print(f"{compound:<10} {raw_str:<15} {corr_str:<18} {change_str:<15} {pct_str:<10}")

    print("\n" + "=" * 70)
