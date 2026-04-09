"""Feature and model helpers for tyre degradation analysis."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

# Import Phase 1B fuel correction (optional; will be used if available)
try:
    from src.features.fuel_correction import (
        apply_fuel_correction,
        estimate_fuel_effect,
        evaluate_fuel_correction,
        print_fuel_correction_summary,
        save_fuel_correction_summary,
    )
    FUEL_CORRECTION_AVAILABLE = True
except ImportError:
    FUEL_CORRECTION_AVAILABLE = False


def create_degradation_table(df: pd.DataFrame, use_fuel_corrected: bool = False) -> pd.DataFrame:
    """Aggregate lap time by compound and tyre life.

    Returns columns: Compound, TyreLife, LapTime (mean), Count.

    Args:
        df: Model-grade laps with LapTime and optionally FuelCorrectedLapTime.
        use_fuel_corrected: If True, must have FuelCorrectedLapTime column (raises error if missing).
                             If False, uses raw LapTime.

    Returns:
        Aggregated degradation table.
        
    Raises:
        ValueError: If use_fuel_corrected=True but FuelCorrectedLapTime column missing.
    """
    time_col = "LapTime"
    if use_fuel_corrected:
        if "FuelCorrectedLapTime" not in df.columns:
            raise ValueError(
                "use_fuel_corrected=True but FuelCorrectedLapTime column not found. "
                "Ensure apply_fuel_correction() was called before this function."
            )
        time_col = "FuelCorrectedLapTime"

    out = (
        df.groupby(["Compound", "TyreLife"])[time_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    out.columns = ["Compound", "TyreLife", "LapTime", "Count"]
    return out


def fit_degradation_models(
    model_deg_df: pd.DataFrame,
    compounds: Iterable[str] = ("SOFT", "MEDIUM", "HARD"),
) -> Dict[str, Tuple[float, float]]:
    """Fit linear degradation model LapTime = slope * TyreLife + intercept.

    Returns:
        Mapping {compound: (slope, intercept)}.
    """
    models: Dict[str, Tuple[float, float]] = {}
    for compound in compounds:
        compound_data = model_deg_df[model_deg_df["Compound"] == compound]
        if len(compound_data) > 1:
            slope, intercept = np.polyfit(
                compound_data["TyreLife"],
                compound_data["LapTime"],
                1,
            )
            models[compound] = (float(slope), float(intercept))
    return models


def models_to_table(models: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Convert model dict into a readable dataframe sorted by slope."""
    rows = []
    for compound, (slope, intercept) in models.items():
        rows.append(
            {
                "Compound": compound,
                "Slope (s/lap)": slope,
                "Intercept (s)": intercept,
            }
        )
    return pd.DataFrame(rows).sort_values("Slope (s/lap)")
