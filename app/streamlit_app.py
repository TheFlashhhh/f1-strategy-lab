"""Minimal Streamlit interface for pit-window strategy decisions."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Support running with: streamlit run app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.build_features import create_degradation_table, fit_degradation_models
from src.simulation.strategy import (
    estimate_pit_loss_window,
    find_optimal_pit_lap,
    optimize_pit_window,
    recommend_action,
)


@st.cache_data
def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load csv, detect pit stops, and prepare cleaned/model datasets."""
    raw_df = pd.read_csv(csv_path)
    selected_df = select_relevant_columns(raw_df)
    pit_df = detect_pit_stops(selected_df)
    clean_df = clean_laps(pit_df)
    model_df = build_model_df(clean_df)
    return pit_df, model_df


@st.cache_data
def build_models_and_pit_loss(pit_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], float, int]:
    """Fit degradation models and estimate median empirical pit-loss."""
    model_deg_df = create_degradation_table(model_df)
    degradation_models = fit_degradation_models(model_deg_df)

    pit_loss_samples = estimate_pit_loss_window(pit_df)
    if len(pit_loss_samples) == 0:
        raise ValueError("No valid pit-loss samples were produced from the dataset.")

    pit_loss_value = float(np.median(pit_loss_samples))
    return degradation_models, pit_loss_value, int(len(pit_loss_samples))


st.set_page_config(page_title="F1 Strategy Lab", layout="wide")
st.title("F1 Strategy Lab")
st.caption("Interactive pit-window decision engine")
st.write(
    "This app uses reusable project modules to estimate pit loss and optimize pit timing "
    "for a selected race state from the 2020 Abu Dhabi lap-level dataset."
)

csv_file = ROOT / "data" / "raw" / "2020_abudhabi_race.csv"

try:
    pit_df, model_df = load_and_prepare_data(str(csv_file))
    degradation_models, pit_loss_value, pit_sample_count = build_models_and_pit_loss(pit_df, model_df)
except Exception as exc:
    st.error(f"Failed to initialize data/model pipeline: {exc}")
    st.stop()

available_compounds = sorted(degradation_models.keys())
if not available_compounds:
    st.error("No degradation models are available from the current dataset.")
    st.stop()

default_current = "MEDIUM" if "MEDIUM" in available_compounds else available_compounds[0]
default_target = "HARD" if "HARD" in available_compounds else available_compounds[0]

st.sidebar.header("Race State")
compound = st.sidebar.selectbox(
    "Current Compound",
    options=available_compounds,
    index=available_compounds.index(default_current),
)
current_tyre_life = st.sidebar.slider("Current Tyre Life (laps)", min_value=1, max_value=40, value=5, step=1)
laps_remaining = st.sidebar.slider("Laps Remaining", min_value=2, max_value=58, value=25, step=1)
target_compound = st.sidebar.selectbox(
    "Target Compound",
    options=available_compounds,
    index=available_compounds.index(default_target),
)

strategy_df = optimize_pit_window(
    degradation_models=degradation_models,
    pit_loss_value=pit_loss_value,
    current_tyre_life=current_tyre_life,
    laps_remaining=laps_remaining,
    compound=compound,
    post_pit_compound=target_compound,
)

optimal_pit_lap, best_total_time = find_optimal_pit_lap(strategy_df)
decision = recommend_action(
    degradation_models=degradation_models,
    pit_loss_value=pit_loss_value,
    current_tyre_life=current_tyre_life,
    laps_remaining=laps_remaining,
    compound=compound,
    post_pit_compound=target_compound,
)

st.subheader("Results")
if decision == "PIT":
    st.markdown(f"### <span style='color: red;'>Recommendation: {decision}</span>", unsafe_allow_html=True)
elif "STAY OUT" in decision:
    st.markdown(f"### <span style='color: green;'>Recommendation: {decision}</span>", unsafe_allow_html=True)
else:
    st.markdown(f"### Recommendation: {decision}")

metric_cols = st.columns(3)
metric_cols[0].metric("Optimal pit lap", f"{optimal_pit_lap}")
metric_cols[1].metric("Estimated pit-loss", f"{pit_loss_value:.2f} s")
metric_cols[2].metric("Best total time", f"{best_total_time:.2f} s")
st.caption("Decision is based on minimizing total race time using compound-specific degradation and empirical pit-loss.")

st.subheader("Pit-Lap Candidate Curve")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(strategy_df["PitLap"], strategy_df["TotalTime"], marker="o", linewidth=2, markersize=4)
ax.axvline(optimal_pit_lap, color="red", linestyle="--", label=f"Optimal: lap {optimal_pit_lap}")
ax.set_xlabel("PitLap")
ax.set_ylabel("TotalTime (s)")
ax.set_title("Total Time vs Candidate Pit Lap")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

st.subheader("Info")
st.markdown(
    f"- Data source: `{csv_file.relative_to(ROOT)}`\n"
    f"- Pit-loss samples used: {pit_sample_count}\n"
    "- This is a deterministic baseline and does not model traffic, safety cars, or opponent actions."
)
