"""Minimal runnable demo for the F1 strategy modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as: python app/demo_strategy.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.build_features import create_degradation_table, fit_degradation_models, models_to_table
from src.simulation.strategy import (
    estimate_pit_loss_window,
    find_optimal_pit_lap,
    optimize_pit_window,
    recommend_action,
)


def main() -> None:
    data_path = ROOT / "data" / "raw" / "2020_abudhabi_race.csv"
    df_raw = pd.read_csv(data_path)

    df = select_relevant_columns(df_raw)
    df = detect_pit_stops(df)
    clean_df = clean_laps(df)
    model_df = build_model_df(clean_df)

    model_deg_df = create_degradation_table(model_df)
    degradation_models = fit_degradation_models(model_deg_df)

    pit_loss_samples = estimate_pit_loss_window(df)
    if len(pit_loss_samples) == 0:
        raise RuntimeError("No pit-loss samples were produced.")

    pit_loss_value = float(np.median(pit_loss_samples))

    current_tyre_life = 5
    laps_remaining = 25
    compound = "MEDIUM"

    strategy_df = optimize_pit_window(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        compound=compound,
    )
    optimal_pit_lap, optimal_total_time = find_optimal_pit_lap(strategy_df)

    print("=== F1 Strategy Demo ===")
    print(f"Dataset rows: {len(df_raw)}")
    print(f"Model laps: {len(model_df)}")
    print(f"Pit-loss samples: {len(pit_loss_samples)}")
    print(f"Pit-loss median: {pit_loss_value:.2f} s")
    print("\nDegradation models:")
    print(models_to_table(degradation_models).to_string(index=False))

    print("\nOptimization example:")
    print(f"Current tyre life: {current_tyre_life}")
    print(f"Laps remaining: {laps_remaining}")
    print(f"Optimal pit lap: {optimal_pit_lap}")
    print(f"Minimum total time: {optimal_total_time:.2f} s")

    decision = recommend_action(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        compound=compound,
    )
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
