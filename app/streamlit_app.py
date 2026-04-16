"""Improved interactive Streamlit interface for pit-window strategy decisions.

This app provides a clean, user-focused interface for the F1 Strategy Lab
recommendation engine. It combines Phase 2B hybrid modeling with Phase 1 pipeline
for automated pit strategy recommendations.

Run as: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - import-only fallback for smoke tests
    class _MissingStreamlitShim:
        """Minimal shim so module import succeeds when Streamlit is unavailable."""

        def cache_data(self, func=None, **_kwargs):
            if func is None:
                def decorator(inner):
                    return inner
                return decorator
            return func

        def __getattr__(self, _name):
            raise ModuleNotFoundError(
                "streamlit is not installed in this Python environment. "
                "Install requirements or use the project virtual environment to run the app."
            )

    st = _MissingStreamlitShim()

# Support running with: streamlit run app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.data.loader import DataLoader
from src.features.evaluate_degradation import evaluate_all_degradation
from src.features.hybrid_modeling import load_or_build_hybrid_dataset
from src.simulation.strategy import (
    estimate_pit_loss_window,
    find_optimal_pit_lap,
    optimize_pit_window,
)
from src.simulation.strategy_engine import recommend_best_strategy
from src.simulation.strategy_sensitivity import assess_strategy_stability

# ============================================================================
# CACHED DATA LOADING
# ============================================================================

@st.cache_data
def load_and_prepare_hybrid_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load Phase 2B hybrid dataset and prepare for analysis."""
    try:
        df_raw, hybrid_context = load_or_build_hybrid_dataset(project_root=ROOT)
        context_dict = hybrid_context.to_dict()
    except Exception as e:
        st.warning(
            f"Phase 2B hybrid loading failed ({e}). Falling back to Miami historical."
        )
        from src.data.loader import DataLoader
        loader = DataLoader(project_root=ROOT)
        df_raw = loader.load_data(dataset="miami_historical")
        context_dict = {
            "timestamp": "",
            "weighting_scheme": "fallback",
            "active_pools": [{"pool_id": "miami_historical", "name": "Miami Historical"}],
            "total_samples": len(df_raw),
        }
    
    selected_df = select_relevant_columns(df_raw)
    pit_df = detect_pit_stops(selected_df)
    clean_df = clean_laps(pit_df)
    model_df = build_model_df(clean_df)
    
    return pit_df, model_df, context_dict


@st.cache_data
def build_integrated_pipeline(pit_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple:
    """Build complete Phase 1 pipeline with fuel correction and piecewise degradation."""
    deg_result = evaluate_all_degradation(
        model_df,
        use_fuel_correction=True,
        use_piecewise=True,
    )

    pit_loader = DataLoader(project_root=ROOT)
    pit_raw = pit_loader.load_data(dataset="miami_historical")
    pit_source_df = detect_pit_stops(select_relevant_columns(pit_raw))
    pit_loss_samples = estimate_pit_loss_window(pit_source_df)
    if len(pit_loss_samples) == 0:
        raise ValueError("No valid pit-loss samples were produced from the dataset.")

    pit_loss_value = float(np.median(pit_loss_samples))
    return deg_result, pit_loss_value, int(len(pit_loss_samples))


@st.cache_data
def load_phase2d_validation_summary() -> dict | None:
    """Load the latest Phase 2D summary artifact if it exists."""
    artifact_path = ROOT / "data" / "processed" / "phase2d_validation_summary.json"
    if not artifact_path.exists():
        return None
    try:
        with open(artifact_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


# ============================================================================
# UI SECTIONS
# ============================================================================

def render_page_header():
    """Render page title and overview."""
    st.set_page_config(page_title="F1 Strategy Lab", layout="wide")
    st.title("🏁 F1 Strategy Lab")
    st.markdown("**Real-time pit strategy recommendations using hybrid degradation modeling**")
    
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        This app analyzes tyre degradation and pit-loss data from real F1 races to recommend
        optimal pit timing. It combines:
        
        - **Phase 2B Hybrid Modeling:** 40% Miami historical data + 60% recent 2026 race data
        - **Phase 1B Fuel Correction:** Removes fuel-load confound from lap times
        - **Phase 1C Degradation Modeling:** Detects tyre-wear cliffs for better late-stint predictions
        - **Phase 2A Strategy Engine:** Automatically searches one-stop and two-stop options
        
        **Key limitation:** This model does NOT account for traffic, safety cars, or opponent actions.
        """)


def render_input_sidebar(available_compounds: list[str]) -> tuple[str, int, int, bool, bool]:
    """Render sidebar controls and return user inputs."""
    st.sidebar.header("Race State Input")
    
    default_current = "MEDIUM" if "MEDIUM" in available_compounds else available_compounds[0]
    
    compound = st.sidebar.selectbox(
        "Current Compound",
        options=available_compounds,
        index=available_compounds.index(default_current),
        help="What compound are you currently on?"
    )
    
    current_tyre_life = st.sidebar.slider(
        "Current Tyre Age",
        min_value=1,
        max_value=40,
        value=5,
        help="How many laps has the current tyre been used?"
    )
    
    laps_remaining = st.sidebar.slider(
        "Laps Remaining",
        min_value=2,
        max_value=58,
        value=25,
        help="How many laps until the end of the race?"
    )
    
    st.sidebar.divider()
    st.sidebar.header("Options")
    include_two_stop = st.sidebar.checkbox(
        "Consider two-stop strategies",
        value=True,
        help="Include two-stop pit strategies in the analysis?"
    )
    
    show_advanced = st.sidebar.checkbox(
        "Show advanced analysis",
        value=False,
        help="Show detailed pit-lap curves and model inspection"
    )
    
    return compound, current_tyre_life, laps_remaining, include_two_stop, show_advanced


def render_stint_timeline(best_plan):
    """Render a simple visual stint timeline."""
    # Tyre color mapping with emojis
    tyre_colors = {"SOFT": "🔴", "MEDIUM": "🟡", "HARD": "⚪"}
    
    st.markdown("**Planned Stints:**")
    
    # Stint sequence
    stints_text = f"Current: {tyre_colors.get(best_plan.current_compound, '?')} {best_plan.current_compound} "
    stints_text += f"→ Pit L{best_plan.pit_lap} → {tyre_colors.get(best_plan.next_compound, '?')} {best_plan.next_compound}"
    
    if best_plan.strategy_type.lower() == "two-stop" and best_plan.second_pit_lap:
        stints_text += f" → Pit L{best_plan.second_pit_lap} → {tyre_colors.get(best_plan.final_compound, '?')} {best_plan.final_compound}"
    
    stints_text += " → Finish"
    st.caption(stints_text)


def render_recommendation(best_plan, pit_loss_value: float):
    """Render the main recommendation section with clear call-to-action."""
    st.subheader("🎯 Recommendation")
    
    # Build human-readable summary sentence
    action = "Box" if best_plan.pit_lap <= 1 else f"Box on lap {best_plan.pit_lap}"
    
    if best_plan.strategy_type.lower() == "one-stop":
        summary = f"{action} for {best_plan.next_compound}. Finish the race on this tyre."
    else:
        summary = f"{action} for {best_plan.next_compound}, then lap {best_plan.second_pit_lap} for {best_plan.final_compound}."
    
    # Display summary as prominent card
    with st.container(border=True):
        st.markdown(f"### {summary}")
        st.divider()
    
    # Structured fields below summary
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1.5])
    
    with col1:
        st.metric("Next Action", "PIT" if best_plan.pit_lap <= 1 else "STAY OUT", 
                  delta=f"in {best_plan.pit_lap} laps" if best_plan.pit_lap > 1 else "Now")
    
    with col2:
        st.metric("To Compound", best_plan.next_compound)
    
    with col3:
        st.metric("Type", best_plan.strategy_type.upper())
    
    with col4:
        st.metric("Est. Time", f"{best_plan.total_race_time:.1f}s")
    
    # Feasibility and explanation
    st.write("")
    if best_plan.feasible:
        st.success(f"✓ Feasible — {best_plan.feasibility_reason}")
    else:
        st.warning(f"⚠️ {best_plan.feasibility_reason}")
    
    # Stint timeline visualization
    render_stint_timeline(best_plan)


def render_comparison_table(all_ranked_plans, best_plan):
    """Render alternative strategies comparison."""
    st.subheader("📊 Alternative Strategies")
    
    strategy_data = []
    tyre_colors = {"SOFT": "🔴", "MEDIUM": "🟡", "HARD": "⚪"}
    
    for i, plan in enumerate(all_ranked_plans[:5], 1):
        time_diff = plan.total_race_time - best_plan.total_race_time
        delta_str = "BEST" if abs(time_diff) < 0.01 else f"+{time_diff:.2f}s"
        
        # Build tyre sequence string
        if plan.strategy_type.lower() == "one-stop":
            tyre_seq = f"{tyre_colors.get(plan.next_compound, '?')} {plan.next_compound}"
            pit_laps = f"L{plan.pit_lap}"
        else:
            tyre_seq = f"{tyre_colors.get(plan.next_compound, '?')} {plan.next_compound} → {tyre_colors.get(plan.final_compound, '?')} {plan.final_compound}"
            pit_laps = f"L{plan.pit_lap}, L{plan.second_pit_lap}"
        
        strategy_data.append({
            "Rank": "★" if i == 1 else str(i),
            "Type": "1-Stop" if plan.strategy_type.lower() == "one-stop" else "2-Stop",
            "Tyres": tyre_seq,
            "Pit Laps": pit_laps,
            "Est. Time": f"{plan.total_race_time:.1f}s",
            "Δ vs Best": delta_str,
            "Feasible": "✓" if plan.feasible else "⚠"
        })
    
    df_strats = pd.DataFrame(strategy_data)
    st.dataframe(
        df_strats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn(width="small"),
            "Type": st.column_config.TextColumn(width="small"),
            "Feasible": st.column_config.TextColumn(width="small"),
        }
    )
    
    st.caption("★ = Best strategy | ✓ = Feasible | ⚠ = Marginal feasibility")


def render_model_status(deg_result):
    """Render model quality and availability."""
    st.subheader("🔧 Degradation Models")
    
    col1, col2, col3 = st.columns(3)
    compounds = ["SOFT", "MEDIUM", "HARD"]
    
    for i, comp in enumerate(compounds):
        info = deg_result.get_model_info(comp)
        col = [col1, col2, col3][i]
        
        if info["model_type"]:
            # Build model description
            if info["is_piecewise"] and info.get("breakpoint_tyre_life"):
                model_type_str = f"Piecewise | cliff @ {info['breakpoint_tyre_life']}"
            elif info["is_piecewise"]:
                model_type_str = "Piecewise"
            else:
                model_type_str = "Linear fallback"
            
            # Display as card
            with col:
                st.metric(comp, f"{info['samples']} laps")
                st.caption(model_type_str)
        else:
            col.metric(comp, "N/A", help="Insufficient data")


def render_data_context(hybrid_context):
    """Render data source and blending information."""
    st.subheader("📚 Data & Model Context")
    
    if hybrid_context and hybrid_context.get("active_pools"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Active Pools",
                len(hybrid_context.get("active_pools", [])),
                help="Number of race datasets being blended"
            )
        
        with col2:
            total_laps = hybrid_context.get('total_samples', 0)
            st.metric(
                "Model Laps",
                f"{total_laps:,}",
                help="Laps used to train degradation models"
            )
        
        with col3:
            st.metric(
                "Blend Strategy",
                "Recency 60%",
                help="Current season weighted 60%, historical 40%"
            )
        
        # Show which pools are active with cleaner format
        st.markdown("**Data Pools:**")
        for pool in hybrid_context.get("active_pools", []):
            pool_name = pool['name'].replace(" (2022-2025)", "").replace(" (Pre-Miami)", "")
            pool_weight = pool.get('recency_weight', '?')
            pool_laps = pool.get('sample_count', '?')
            st.caption(
                f"• {pool_name}: {pool_laps:,} laps ({pool_weight})"
            )


def render_phase2d_validation_note():
    """Render a lightweight note about representative validation coverage."""
    summary_artifact = load_phase2d_validation_summary()
    if not summary_artifact:
        return

    metadata = summary_artifact.get("metadata", {})
    aggregate = summary_artifact.get("aggregate_summary", {})
    scenario_count = metadata.get("scenario_count", 0)
    stable = aggregate.get("stability_counts", {}).get("Stable", 0)
    fragile = aggregate.get("stability_counts", {}).get("Fragile", 0)
    one_stop = aggregate.get("strategy_type_counts", {}).get("one-stop", 0)
    two_stop = aggregate.get("strategy_type_counts", {}).get("two-stop", 0)

    st.info(
        "Phase 2D validation artifact loaded: "
        f"{scenario_count} representative scenarios | "
        f"{stable} stable | {fragile} fragile | "
        f"1-stop {one_stop}, 2-stop {two_stop}."
    )


def render_advanced_analysis(available_compounds, deg_result, pit_loss_value, 
                            current_tyre_life, laps_remaining, current_compound):
    """Render advanced features (pit-lap curve, detailed model inspection)."""
    st.subheader("📈 Advanced Analysis")
    
    # Pit-lap curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pit-Timing Sensitivity Curve**")
        target_compound = st.selectbox(
            "Select target compound:",
            options=[c for c in available_compounds if c != current_compound],
            key="advanced_compound"
        )
        
        comparison_df = optimize_pit_window(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            compound=current_compound,
            post_pit_compound=target_compound,
        )
        
        cmp_pit_lap, cmp_total_time = find_optimal_pit_lap(comparison_df)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(comparison_df["PitLap"], comparison_df["TotalTime"], 
               marker="o", linewidth=2, markersize=5, label="Total Race Time")
        ax.axvline(cmp_pit_lap, color="red", linestyle="--", alpha=0.7, 
                  label=f"Optimal pit lap: {cmp_pit_lap}")
        ax.set_xlabel("Pit Lap")
        ax.set_ylabel("Total Race Time (s)")
        ax.set_title(f"One-Stop to {target_compound}: Pit Timing Analysis")
        ax.grid(True, alpha=0.2)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    
    # Lap-time predictions
    with col2:
        st.markdown("**Predicted Lap Times by Tyre Life**")
        
        compounds_to_show = ["SOFT", "MEDIUM", "HARD"]
        compounds_to_show = [c for c in compounds_to_show if deg_result.get_model_info(c)["model_type"]]
        
        prediction_data = []
        for tyre_life in [1, 5, 10, 15, 20]:
            row = {"Tyre Life": tyre_life}
            for comp in compounds_to_show:
                lap_time = deg_result.predict_lap_time(comp, tyre_life)
                row[comp] = f"{lap_time:.2f} s" if lap_time else "N/A"
            prediction_data.append(row)
        
        st.dataframe(pd.DataFrame(prediction_data), use_container_width=True, hide_index=True)


def render_phase2c_sensitivity(
    best_plan,
    pit_loss_value: float,
    deg_result,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
):
    """Render Phase 2C sensitivity analysis section."""
    st.subheader("🔍 Recommendation Stability (Phase 2C)")
    
    try:
        # Perform sensitivity analysis
        stability_assessment = assess_strategy_stability(
            baseline_plan=best_plan,
            pit_loss_value=pit_loss_value,
            degradation_models=deg_result,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
        )
        
        # Display stability label with color coding
        stability_colors = {
            "Stable": "green",
            "Moderately Sensitive": "orange",
            "Fragile": "red",
        }
        color = stability_colors.get(stability_assessment.stability_label, "blue")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.metric("Stability", stability_assessment.stability_label)
        with col2:
            pit_changes = sum(1 for s in stability_assessment.pit_loss_sensitivity.scenarios if s.recommendation_changed)
            st.metric("Pit-Loss Sensitive", "Yes" if pit_changes > 0 else "No")
        with col3:
            deg_changes = sum(1 for s in stability_assessment.degradation_sensitivity.scenarios if s.recommendation_changed)
            st.metric("Degradation Sensitive", "Yes" if deg_changes > 0 else "No")
        
        st.divider()
        
        # Pit-loss sensitivity table
        st.markdown("**Pit-Loss Sensitivity** (testing ±1-2 seconds)")
        pit_loss_data = []
        for scenario in stability_assessment.pit_loss_sensitivity.scenarios:
            pit_loss_data.append({
                "Pit Loss": f"{scenario.pit_loss_value:.1f}s",
                "Next Compound": scenario.best_plan.next_compound,
                "Pit Lap": scenario.best_plan.pit_lap,
                "Time": f"{scenario.best_plan.total_race_time:.1f}s",
                "Changed": "Yes" if scenario.recommendation_changed else "No",
            })
        
        if pit_loss_data:
            st.dataframe(
                pd.DataFrame(pit_loss_data),
                use_container_width=True,
                hide_index=True,
            )
        
        st.divider()
        
        # Degradation sensitivity table
        st.markdown("**Degradation Sensitivity** (testing optimistic/pessimistic wear)")
        deg_data = []
        for scenario in stability_assessment.degradation_sensitivity.scenarios:
            scenario_label = "Optimistic" if "optimistic" in scenario.scenario_name else "Pessimistic"
            deg_data.append({
                "Scenario": scenario_label,
                "Scale Factor": f"{scenario.degradation_scale_factor:.1f}x",
                "Next Compound": scenario.best_plan.next_compound,
                "Pit Lap": scenario.best_plan.pit_lap,
                "Time": f"{scenario.best_plan.total_race_time:.1f}s",
                "Changed": "Yes" if scenario.recommendation_changed else "No",
            })
        
        if deg_data:
            st.dataframe(
                pd.DataFrame(deg_data),
                use_container_width=True,
                hide_index=True,
            )
        
        # Flip conditions
        if stability_assessment.flip_conditions:
            st.divider()
            st.markdown("**⚠️ Flip Conditions**")
            for condition in stability_assessment.flip_conditions:
                st.warning(f"• {condition}")
        
    except Exception as e:
        st.error(f"Sensitivity analysis failed: {e}")


def render_assumptions_footer():
    """Render assumptions and limitations."""
    with st.expander("⚠️ Assumptions & Limitations", expanded=False):
        st.markdown("""
        **Model Assumptions:**
        - Linear or piecewise-linear tyre degradation
        - Steady Miami pit-loss baseline calibrated from historical race windows
        - Constant fuel burn effect across the race
        - Green-flag conditions throughout
        
        **Limitations:**
        - No traffic or overtaking model
        - No adaptive strategy (e.g., responding to safety cars)
        - Based on historical Miami + 2026 race data only
        - SOFT compound has limited data (~33 laps); use with caution
        - Two-stop strategies are evaluated naively; real strategy depends on position/gaps
        
        **What this model IS:**
        - A deterministic pit-window optimizer for isolated decision-making
        - A tool to understand tyre degradation trade-offs
        - A reference point for strategy discussion
        
        **What this model is NOT:**
        - A real-time race simulator
        - A multi-option traffic optimizer
        - A substitute for real F1 team analysis
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    render_page_header()
    
    try:
        pit_df, model_df, hybrid_context = load_and_prepare_hybrid_data()
        deg_result, pit_loss_value, pit_sample_count = build_integrated_pipeline(pit_df, model_df)
    except Exception as exc:
        st.error(f"❌ Failed to initialize pipeline: {exc}")
        st.stop()

    # Get available compounds
    available_compounds = ["SOFT", "MEDIUM", "HARD"]
    available_compounds = [c for c in available_compounds if deg_result.get_model_info(c)["model_type"]]

    if not available_compounds:
        st.error("❌ No degradation models available. Check data integrity.")
        st.stop()

    # Get user input
    compound, current_tyre_life, laps_remaining, include_two_stop, show_advanced = render_input_sidebar(available_compounds)

    # Generate recommendations
    best_plan, all_ranked_plans = recommend_best_strategy(
        degradation_models=deg_result,
        pit_loss_value=pit_loss_value,
        current_compound=compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        candidate_compounds=available_compounds,
        include_two_stop=include_two_stop,
    )

    # === STRATEGY SECTION ===
    render_recommendation(best_plan, pit_loss_value)
    st.divider()
    
    # === ALTERNATIVES SECTION ===
    render_comparison_table(all_ranked_plans, best_plan)
    st.divider()
    
    # === MODEL & DATA CONTEXT SECTION ===
    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_model_status(deg_result)
    with col_right:
        render_data_context(hybrid_context)
    render_phase2d_validation_note()
    
    # === PHASE 2C SENSITIVITY SECTION ===
    st.divider()
    render_phase2c_sensitivity(
        best_plan=best_plan,
        pit_loss_value=pit_loss_value,
        deg_result=deg_result,
        current_compound=compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
    )
    
    # === ADVANCED ANALYSIS SECTION ===
    if show_advanced:
        st.divider()
        render_advanced_analysis(
            available_compounds, deg_result, pit_loss_value,
            current_tyre_life, laps_remaining, compound
        )
    
    # === ASSUMPTIONS & LIMITATIONS ===
    st.divider()
    render_assumptions_footer()


if __name__ == "__main__":
    main()
