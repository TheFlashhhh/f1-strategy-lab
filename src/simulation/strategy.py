"""Strategy simulation helpers extracted from the notebook."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.preprocess import get_race_group_columns


def predict_lap_time(
    degradation_models: Union[Dict[str, Tuple[float, float]], object],
    compound: str,
    tyre_life: int,
) -> Optional[float]:
    """Predict lap time for a compound/tyre-life combination.

    Unified prediction helper that works with both:
    - Legacy format: Dict[str, Tuple[slope, intercept]] -> use linear prediction
    - New format: DegradationEvaluationResult -> use its built-in predict method

    Args:
        degradation_models: Either legacy linear dict or DegradationEvaluationResult
        compound: Compound name (e.g., "MEDIUM")
        tyre_life: Tyre-life value

    Returns:
        Predicted lap time in seconds, or None if prediction fails
    """
    if hasattr(degradation_models, "predict_lap_time"):
        return degradation_models.predict_lap_time(compound, tyre_life)

    if isinstance(degradation_models, dict) and compound in degradation_models:
        slope, intercept = degradation_models[compound]
        return slope * tyre_life + intercept

    return None


def _is_valid_prediction(value: Optional[float]) -> bool:
    """Return True when a predicted lap time is finite and usable."""
    return value is not None and np.isfinite(value)


def _row_is_valid_pit_loss_baseline(row: pd.Series) -> bool:
    """Return whether a lap row is representative enough for the baseline window."""
    lap_time = row.get("LapTime")
    if pd.isna(lap_time):
        return False
    if "Deleted" in row.index and bool(row["Deleted"]) is True:
        return False
    if "TrackStatus" in row.index and str(row["TrackStatus"]) != "1":
        return False
    return True


def _row_is_valid_pit_loss_event(row: pd.Series) -> bool:
    """Return whether a lap row can contribute to the actual pit-loss window."""
    lap_time = row.get("LapTime")
    if pd.isna(lap_time):
        return False
    if "Deleted" in row.index and bool(row["Deleted"]) is True:
        return False
    return True


def _ensure_pit_stop_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DataFrame has pit-stop flags using race-context grouping."""
    if "PitStop" in df.columns:
        return df

    group_cols = get_race_group_columns(df, include_driver=True)
    if not group_cols:
        raise ValueError("Pit-loss estimation requires Driver and stint context columns.")

    out = df.sort_values(group_cols + ["LapNumber"]).reset_index(drop=True).copy()
    out["PrevStint"] = out.groupby(group_cols)["Stint"].shift(1)
    out["PitStop"] = (
        (out["Stint"] != out["PrevStint"]) & out["PrevStint"].notna()
    ).astype(int)
    out.loc[out.groupby(group_cols).cumcount() == 0, "PitStop"] = 0
    return out


def estimate_pit_loss_window(df: pd.DataFrame) -> np.ndarray:
    """Estimate pit-loss samples using a race-local pit-window method.

    The estimation is intentionally conservative:
    - group by race context + driver to avoid cross-race leakage
    - require representative laps (accurate, not deleted, green if available)
    - drop non-positive samples, which are not physically credible pit losses
    """
    pit_loss_window_values = []
    prepared_df = _ensure_pit_stop_flags(df)
    group_cols = get_race_group_columns(prepared_df, include_driver=True)

    for _, stint_df in prepared_df.groupby(group_cols):
        driver_df = stint_df.sort_values("LapNumber").reset_index(drop=True)
        pit_indices = driver_df.index[driver_df["PitStop"] == 1].tolist()

        for idx in pit_indices:
            before_vals = []
            after_vals = []
            pit_window_vals = []

            j = idx - 1
            while j >= 0 and len(before_vals) < 2:
                row = driver_df.loc[j]
                if _row_is_valid_pit_loss_baseline(row):
                    before_vals.append(float(row["LapTime"]))
                j -= 1

            k = idx + 1
            while k < len(driver_df) and len(after_vals) < 2:
                row = driver_df.loc[k]
                if _row_is_valid_pit_loss_baseline(row):
                    after_vals.append(float(row["LapTime"]))
                k += 1

            for pit_idx in [idx, idx + 1]:
                if pit_idx < len(driver_df):
                    row = driver_df.loc[pit_idx]
                    if _row_is_valid_pit_loss_event(row):
                        pit_window_vals.append(float(row["LapTime"]))

            baseline_vals = before_vals + after_vals
            if len(baseline_vals) >= 2 and pit_window_vals:
                baseline = float(np.mean(baseline_vals))
                expected_time = baseline * len(pit_window_vals)
                actual_time = float(np.sum(pit_window_vals))
                pit_loss = actual_time - expected_time
                if pit_loss > 0 and np.isfinite(pit_loss):
                    pit_loss_window_values.append(pit_loss)

    return np.array(pit_loss_window_values, dtype=float)


def optimize_pit_window(
    degradation_models: Dict[str, Tuple[float, float]],
    pit_loss_value: float,
    current_tyre_life: int,
    laps_remaining: int,
    compound: str = "MEDIUM",
    post_pit_compound: str = "HARD",
    strict_predictions: bool = False,
) -> pd.DataFrame:
    """Compute total race time for each candidate pit lap.

    Works seamlessly with both legacy linear dicts and new
    DegradationEvaluationResult format.
    """
    results = []

    for pit_lap in range(1, laps_remaining):
        remaining_after_pit = laps_remaining - pit_lap

        stay_time = 0.0
        valid_plan = True
        for lap in range(pit_lap):
            tyre_age = current_tyre_life + lap
            lap_time = predict_lap_time(degradation_models, compound, tyre_age)
            if not _is_valid_prediction(lap_time):
                if strict_predictions:
                    valid_plan = False
                    break
                lap_time = 95.0
            stay_time += float(lap_time)

        pit_time = float(pit_loss_value)
        if valid_plan:
            for lap in range(remaining_after_pit):
                tyre_age = 1 + lap
                lap_time = predict_lap_time(degradation_models, post_pit_compound, tyre_age)
                if not _is_valid_prediction(lap_time):
                    if strict_predictions:
                        valid_plan = False
                        break
                    lap_time = 95.0
                pit_time += float(lap_time)

        total_time = stay_time + pit_time if valid_plan else np.nan
        results.append({"PitLap": pit_lap, "TotalTime": total_time})

    return pd.DataFrame(results)


def find_optimal_pit_lap(strategy_df: pd.DataFrame) -> Tuple[int, float]:
    """Return the pit lap and total time for the minimum-time strategy.

    Handles NaN/invalid cases defensively:
    - If all TotalTime values are NaN, raises clear error
    - If no valid rows exist, raises clear error
    - Otherwise returns the minimum-time row
    """
    if strategy_df.empty:
        raise ValueError(
            "Strategy DataFrame is empty. Cannot find optimal pit lap. "
            "Check degradation models are properly initialized."
        )

    if strategy_df["TotalTime"].isna().all():
        raise ValueError(
            "All TotalTime values are NaN in strategy_df. "
            "This indicates degradation model failure. "
            "Check that models were fitted with sufficient data."
        )

    best_idx = strategy_df["TotalTime"].idxmin()
    if pd.isna(best_idx):
        raise ValueError(
            "Strategy optimization failed: idxmin() returned NaN. "
            "This indicates corrupted strategy data."
        )

    return int(strategy_df.loc[best_idx, "PitLap"]), float(strategy_df.loc[best_idx, "TotalTime"])


def recommend_action(
    degradation_models: Dict[str, Tuple[float, float]],
    pit_loss_value: float,
    current_tyre_life: int,
    laps_remaining: int,
    compound: str = "MEDIUM",
    post_pit_compound: str = "HARD",
) -> str:
    """Recommend PIT now or STAY OUT based on pit-window optimization."""
    strategy_df = optimize_pit_window(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        compound=compound,
        post_pit_compound=post_pit_compound,
    )
    optimal_pit_lap, _ = find_optimal_pit_lap(strategy_df)
    if optimal_pit_lap <= 1:
        return "PIT"
    return f"STAY OUT (pit in {optimal_pit_lap} laps)"


def tyre_life_sensitivity(
    degradation_models: Dict[str, Tuple[float, float]],
    pit_loss_value: float,
    laps_remaining: int,
    tyre_life_values: Iterable[int] = range(1, 21),
    compound: str = "MEDIUM",
    post_pit_compound: str = "HARD",
) -> pd.DataFrame:
    """Compute optimal pit lap for a range of current tyre-life values."""
    rows = []
    for current_tyre_life in tyre_life_values:
        strategy_df = optimize_pit_window(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            compound=compound,
            post_pit_compound=post_pit_compound,
        )
        optimal_pit_lap, total_time = find_optimal_pit_lap(strategy_df)
        rows.append(
            {
                "TyreLife": int(current_tyre_life),
                "OptimalPitLap": int(optimal_pit_lap),
                "TotalTime": float(total_time),
            }
        )
    return pd.DataFrame(rows)
