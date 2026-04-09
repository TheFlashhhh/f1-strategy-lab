"""Strategy simulation helpers extracted from the notebook."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


def predict_lap_time(
    degradation_models: Union[Dict[str, Tuple[float, float]], object],
    compound: str,
    tyre_life: int,
) -> Optional[float]:
    """Predict lap time for a compound/tyre-life combination.
    
    Unified prediction helper that works with both:
    - Legacy format: Dict[str, Tuple[slope, intercept]] → use linear prediction
    - New format: DegradationEvaluationResult → use its built-in predict method
    
    Args:
        degradation_models: Either legacy linear dict or DegradationEvaluationResult
        compound: Compound name (e.g., "MEDIUM")
        tyre_life: Tyre-life value
        
    Returns:
        Predicted lap time in seconds, or None if prediction fails
    """
    # Check if it's the new DegradationEvaluationResult format
    if hasattr(degradation_models, 'predict_lap_time'):
        return degradation_models.predict_lap_time(compound, tyre_life)
    
    # Legacy linear format: Dict[str, Tuple[float, float]]
    if isinstance(degradation_models, dict):
        if compound in degradation_models:
            slope, intercept = degradation_models[compound]
            return slope * tyre_life + intercept
    
    return None


def estimate_pit_loss_window(df: pd.DataFrame) -> np.ndarray:
    """Estimate pit loss samples using the pit-window method from the notebook."""
    pit_loss_window_values = []

    for driver in df["Driver"].dropna().unique():
        driver_df = df[df["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)
        pit_indices = driver_df.index[driver_df["PitStop"] == 1].tolist()

        for idx in pit_indices:
            before_vals = []
            after_vals = []
            pit_window_vals = []

            j = idx - 1
            while j >= 0 and len(before_vals) < 2:
                v = driver_df.at[j, "LapTime"]
                if pd.notna(v):
                    before_vals.append(v)
                j -= 1

            k = idx + 1
            while k < len(driver_df) and len(after_vals) < 2:
                v = driver_df.at[k, "LapTime"]
                if pd.notna(v):
                    after_vals.append(v)
                k += 1

            pit_row_time = driver_df.at[idx, "LapTime"]
            if pd.notna(pit_row_time):
                pit_window_vals.append(pit_row_time)

            if idx + 1 < len(driver_df):
                next_row_time = driver_df.at[idx + 1, "LapTime"]
                if pd.notna(next_row_time):
                    pit_window_vals.append(next_row_time)

            baseline_vals = before_vals + after_vals
            if len(baseline_vals) >= 2 and len(pit_window_vals) >= 1:
                baseline = float(np.mean(baseline_vals))
                expected_time = baseline * len(pit_window_vals)
                actual_time = float(np.sum(pit_window_vals))
                pit_loss_window_values.append(actual_time - expected_time)

    return np.array(pit_loss_window_values, dtype=float)


def optimize_pit_window(
    degradation_models: Dict[str, Tuple[float, float]],
    pit_loss_value: float,
    current_tyre_life: int,
    laps_remaining: int,
    compound: str = "MEDIUM",
    post_pit_compound: str = "HARD",
) -> pd.DataFrame:
    """Compute total race time for each candidate pit lap.
    
    Works seamlessly with both legacy linear dicts and new DegradationEvaluationResult format.
    """
    results = []
    
    for pit_lap in range(1, laps_remaining):
        remaining_after_pit = laps_remaining - pit_lap

        stay_time = 0.0
        for lap in range(pit_lap):
            tyre_age = current_tyre_life + lap
            lap_time = predict_lap_time(degradation_models, compound, tyre_age)
            if lap_time is None:
                lap_time = 95.0  # Fallback default
            stay_time += lap_time

        pit_time = float(pit_loss_value)
        for lap in range(remaining_after_pit):
            tyre_age = 1 + lap
            lap_time = predict_lap_time(degradation_models, post_pit_compound, tyre_age)
            if lap_time is None:
                lap_time = 95.0  # Fallback default
            pit_time += lap_time

        results.append({"PitLap": pit_lap, "TotalTime": stay_time + pit_time})

    return pd.DataFrame(results)


def find_optimal_pit_lap(strategy_df: pd.DataFrame) -> Tuple[int, float]:
    """Return the pit lap and total time for the minimum-time strategy."""
    best_idx = strategy_df["TotalTime"].idxmin()
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
