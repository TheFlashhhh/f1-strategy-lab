"""Interactive Streamlit interface for pit-window strategy decisions with integrated Phase 1 stack.

This app uses the complete Phase 1 pipeline:
- Phase 1A: Miami Grand Prix data (2022–2025)
- Phase 1B: Fuel correction (automatic)
- Phase 1C: Piecewise degradation modeling (automatic, with linear fallback)

Run as: streamlit run app/streamlit_app.py
"""

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

from src.data.loader import load_data
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.evaluate_degradation import evaluate_all_degradation
from src.simulation.strategy import (
    estimate_pit_loss_window,
    find_optimal_pit_lap,
    optimize_pit_window,
    recommend_action,
)


@st.cache_data
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1A data, detect pit stops, and prepare cleaned/model datasets."""
    raw_df = load_data(dataset="miami_historical", project_root=ROOT)
    selected_df = select_relevant_columns(raw_df)
    pit_df = detect_pit_stops(selected_df)
    clean_df = clean_laps(pit_df)
    model_df = build_model_df(clean_df)
    return pit_df, model_df


@st.cache_data
def build_integrated_pipeline(pit_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple:
    """Build complete Phase 1 pipeline with fuel correction and piecewise degradation."""
    # Evaluate degradation (Phase 1B + 1C integrated)
    deg_result = evaluate_all_degradation(
        model_df,
        use_fuel_correction=True,  # Phase 1B
        use_piecewise=True,         # Phase 1C
    )

    # Estimate pit loss
    pit_loss_samples = estimate_pit_loss_window(pit_df)
    if len(pit_loss_samples) == 0:
        raise ValueError("No valid pit-loss samples were produced from the dataset.")

    pit_loss_value = float(np.median(pit_loss_samples))
    return deg_result, pit_loss_value, int(len(pit_loss_samples))


st.set_page_config(page_title="F1 Strategy Lab", layout="wide")
st.title("F1 Strategy Lab")
st.caption("Interactive pit-window decision engine (Phase 1: Data + Fuel Correction + Piecewise Degradation)")
st.write(
    "This app uses the integrated Phase 1 pipeline for Miami Grand Prix races (2022–2025):\n"
    "- **Phase 1A**: Race data loading (Parquet)\n"
    "- **Phase 1B**: Fuel correction (automatic)\n"
    "- **Phase 1C**: Piecewise degradation with cliff detection (where data supports it)"
)

try:
    pit_df, model_df = load_and_prepare_data()
    deg_result, pit_loss_value, pit_sample_count = build_integrated_pipeline(pit_df, model_df)
except Exception as exc:
    st.error(f"Failed to initialize Phase 1 pipeline: {exc}")
    st.stop()

# Build compound list from available models
available_compounds = ["SOFT", "MEDIUM", "HARD"]
available_compounds = [c for c in available_compounds if deg_result.get_model_info(c)["model_type"]]

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
    degradation_models=deg_result,  # Pass unified result directly
    pit_loss_value=pit_loss_value,
    current_tyre_life=current_tyre_life,
    laps_remaining=laps_remaining,
    compound=compound,
    post_pit_compound=target_compound,
)

optimal_pit_lap, best_total_time = find_optimal_pit_lap(strategy_df)
decision = recommend_action(
    degradation_models=deg_result,
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

st.subheader("Model Status")
col1, col2, col3 = st.columns(3)
for i, comp in enumerate(["SOFT", "MEDIUM", "HARD"]):
    info = deg_result.get_model_info(comp)
    if info["model_type"]:
        with [col1, col2, col3][i]:
            st.metric(
                f"{comp}",
                f"{info['model_type']}",
                f"{info['samples']} laps"
                + (f" | cliff @{info['breakpoint_tyre_life']}" if info["is_piecewise"] else ""),
            )

st.subheader("Info")
st.markdown(
    f"- **Data source**: Phase 1A Miami Grand Prix Parquet (2022–2025)\n"
    f"- **Pit-loss samples used**: {pit_sample_count}\n"
    f"- **Fuel correction**: Applied automatically (Phase 1B)\n"
    f"- **Degradation model**: Piecewise with cliff detection (Phase 1C) where data supports; linear fallback\n"
    f"- **Baseline**: Deterministic optimization (does not model traffic, safety cars, or opponent actions)"
)

