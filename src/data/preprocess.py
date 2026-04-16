"""Reusable preprocessing helpers extracted from the strategy notebook."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


RACE_CONTEXT_COLUMNS = [
    "season",
    "event_name",
    "session_name",
    "data_group",
    "__phase2b_pool_id",
]


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


def get_race_group_columns(df: pd.DataFrame, include_driver: bool = True) -> list[str]:
    """Return the best available grouping columns for race-local operations.

    Hybrid and historical datasets can contain multiple races for the same
    driver. Grouping only by ``Driver`` leaks state across race boundaries,
    which breaks pit-stop detection, fuel-progress normalization, and any
    race-local calibration step.
    """
    group_cols = [col for col in RACE_CONTEXT_COLUMNS if col in df.columns]
    if include_driver and "Driver" in df.columns:
        group_cols.append("Driver")
    return group_cols or (["Driver"] if include_driver and "Driver" in df.columns else [])


def select_relevant_columns(
    df: pd.DataFrame,
    relevant_columns: Iterable[str] = DEFAULT_RELEVANT_COLUMNS,
) -> pd.DataFrame:
    """Select only the columns needed for strategy analysis."""
    selected_columns = list(relevant_columns)
    for col in RACE_CONTEXT_COLUMNS:
        if col in df.columns and col not in selected_columns:
            selected_columns.append(col)
    return df.loc[:, selected_columns].copy()


def detect_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Detect pit stops by stint changes per driver.

    A pit stop is flagged when a driver's current stint differs from the
    previous lap's stint inside the same race context. The first lap for each
    driver/race group is forced to 0.
    """
    group_cols = get_race_group_columns(df, include_driver=True)
    if not group_cols:
        raise ValueError("detect_pit_stops requires a Driver column.")

    out = df.sort_values(group_cols + ["LapNumber"]).reset_index(drop=True).copy()
    out["PrevStint"] = out.groupby(group_cols)["Stint"].shift(1)
    out["PitStop"] = (
        (out["Stint"] != out["PrevStint"]) & out["PrevStint"].notna()
    ).astype(int)
    out.loc[out.groupby(group_cols).cumcount() == 0, "PitStop"] = 0
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
