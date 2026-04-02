"""Reusable preprocessing helpers extracted from the strategy notebook."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_RELEVANT_COLUMNS = [
    "Driver",
    "LapNumber",
    "LapTime",
    "Compound",
    "TyreLife",
    "PitOutTime",
    "PitInTime",
    "Stint",
    "IsAccurate",
    "Deleted",
    "TrackStatus",
]


def select_relevant_columns(
    df: pd.DataFrame,
    relevant_columns: Iterable[str] = DEFAULT_RELEVANT_COLUMNS,
) -> pd.DataFrame:
    """Select only the columns needed for strategy analysis."""
    return df.loc[:, list(relevant_columns)].copy()


def detect_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Detect pit stops by stint changes per driver.

    A pit stop is flagged when a driver's current stint differs from the
    previous lap's stint. The first lap for each driver is forced to 0.
    """
    out = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True).copy()
    out["PrevStint"] = out.groupby("Driver")["Stint"].shift(1)
    out["PitStop"] = (out["Stint"] != out["PrevStint"]).astype(int)
    out.loc[out.groupby("Driver").cumcount() == 0, "PitStop"] = 0
    return out


def clean_laps(df: pd.DataFrame, max_lap_time: float = 150.0) -> pd.DataFrame:
    """Clean lap records by removing invalid lap times and outliers."""
    out = df.dropna(subset=["LapTime"]).copy()
    out["LapTime"] = pd.to_numeric(out["LapTime"], errors="coerce")
    out = out.dropna(subset=["LapTime"])
    out = out[out["LapTime"] <= max_lap_time].copy()
    return out


def build_model_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to model-grade green-flag laps used for regression fitting."""
    return df[
        (df["PitStop"] != 1)
        & (df["TyreLife"] > 2)
        & (df["IsAccurate"] == True)
        & (df["Deleted"] == False)
        & (df["TrackStatus"].isin([1, "1"]))
    ].copy()
