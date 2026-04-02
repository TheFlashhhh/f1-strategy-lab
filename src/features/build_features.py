"""Feature and model helpers for tyre degradation analysis."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def create_degradation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lap time by compound and tyre life.

    Returns columns: Compound, TyreLife, LapTime (mean), Count.
    """
    out = (
        df.groupby(["Compound", "TyreLife"])["LapTime"]
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
