"""Runnable demo for the calibrated F1 Strategy Lab strategy pipeline.

This demo demonstrates the current strategy stack:
- Phase 1A: Data loading (Miami historical + 2026 pre-Miami blended)
- Phase 1B: Fuel correction (fuel-load confound removal)
- Phase 1C: Degradation modeling (piecewise with cliff detection)
- Phase 2B / Pre-3: Miami-anchor role-based hybrid modeling
- Phase 2E: Strategy refinement / calibration (pit-loss + search cleanup)

Result: Improved pit-timing strategy with transparent hybrid-model reporting.

Run as: python app/demo_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow running as: python app/demo_strategy.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import detect_pit_stops, select_relevant_columns
from src.data.loader import DataLoader
from src.features.hybrid_modeling import (
    build_role_based_hybrid_model,
    load_or_build_hybrid_dataset,
    summarize_hybrid_context,
)
from src.simulation.strategy import (
    estimate_pit_loss_window,
    find_optimal_pit_lap,
    optimize_pit_window,
    recommend_action,
)
from src.simulation.strategy_engine import recommend_best_strategy
from src.simulation.strategy_sensitivity import assess_strategy_stability


def main() -> None:
    """Run integrated Phase 2B hybrid modeling demo."""
    print("\n" + "=" * 80)
    print("F1 STRATEGY LAB - CALIBRATED STRATEGY PIPELINE")
    print("=" * 80)

    # Phase 2B: Load hybrid dataset
    print("\nPhase 2B: Loading hybrid dataset (Miami historical + 2026 pre-Miami)...")
    try:
        df_raw, hybrid_context = load_or_build_hybrid_dataset(project_root=ROOT)
        print(f"  [OK] Hybrid context loaded: {len(df_raw)} total source laps")
        print(f"  [OK] Active pools: {len(hybrid_context.active_pools)}")
        for pool in hybrid_context.active_pools:
            print(f"    - {pool.name} ({pool.sample_count} laps, role: {pool.circuit_role})")
    except Exception as e:
        print(f"  [ERROR] Hybrid loading failed: {e}")
        print("  Falling back to Miami historical only...")
        from src.data.loader import load_data
        df_raw = load_data(dataset="miami_historical", project_root=ROOT)
        hybrid_context = None
    
    print("\nPhase 2B / Pre-3: Building role-based hybrid degradation model...")
    deg_result, hybrid_context = build_role_based_hybrid_model(project_root=ROOT)

    # Display model status
    print("\nModel Status (by compound):")
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        info = deg_result.get_model_info(compound)
        if info["model_type"]:
            print(
                f"  {compound:8s} ({info['samples']:4d} total model laps): "
                f"{info['model_type']:22s} "
                f"| support: {info.get('support_tier', 'n/a')}"
            )
            print(
                f"    Miami anchor: {info.get('miami_model_type')} "
                f"({info.get('miami_model_laps', 0)} laps) | "
                f"2026 recency: {info.get('recency_model_type')} "
                f"({info.get('recency_model_laps', 0)} laps)"
            )
            if info.get("support_reason"):
                print(f"    Support note: {info['support_reason']}")

    # Estimate pit loss from Miami-only race context (circuit-specific calibration)
    print("\nEstimating Miami pit loss baseline...")
    pit_loader = DataLoader(project_root=ROOT)
    pit_raw = pit_loader.load_data(dataset="miami_historical")
    pit_source_df = detect_pit_stops(select_relevant_columns(pit_raw))
    pit_loss_samples = estimate_pit_loss_window(pit_source_df)
    if len(pit_loss_samples) == 0:
        raise RuntimeError("No pit-loss samples were produced.")
    pit_loss_value = float(np.median(pit_loss_samples))
    print(f"  Pit-loss samples: {len(pit_loss_samples)}")
    print(f"  Miami median pit-loss: {pit_loss_value:.2f} s")

    # Report hybrid modeling context (Phase 2B)
    if hybrid_context:
        print("\n" + "-" * 80)
        print("PHASE 2B HYBRID MODELING CONTEXT")
        print("-" * 80)
        
        summary = summarize_hybrid_context(hybrid_context)
        
        print("\nData Pool Composition:")
        for pool_info in summary["data_grouping"]:
            print(f"\n  {pool_info['name']}")
            print(f"    Role: {pool_info['role']}")
            print(f"    Target: {pool_info['target_race_context']}")
            print(f"    Raw weight: {pool_info['recency_weight']}")
            normalized = pool_info["normalized_weight"]
            print(f"    Normalized: {normalized:.1%}" if normalized is not None else "    Normalized: role-based only")
            print(f"    Laps: {pool_info['sample_counts']['total_laps']}")
            if pool_info["sample_counts"]["by_compound"]:
                for compound, count in pool_info["sample_counts"]["by_compound"].items():
                    print(f"      {compound}: {count}")

        print("\nBlending Strategy:")
        print(f"  Method: {summary['blending_strategy']['method']}")
        print(f"  Total laps across source pools: {summary['total_laps']}")
        print("\n  Philosophy:")
        print("    - Miami pool: Circuit-specific anchor")
        print("    - 2026 pool: Bounded recency adjustment/support")
        print("    - Non-Miami 2026 races are not presented as direct Miami degradation truth")
        
        # Save summary
        summary_path = ROOT / "data" / "processed" / "phase2b_data_summary.json"
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n  [OK] Summary saved to {summary_path}")
        except Exception as e:
            print(f"\n  [ERROR] Failed to save summary: {e}")
        
        print("-" * 80)

    # Optimize strategy
    print("\nOptimizing pit strategy...")
    current_tyre_life = 5
    laps_remaining = 25
    compound = "MEDIUM"
    
    # Single-scenario optimization
    print("\n  [Single Scenario Analysis]")
    strategy_df = optimize_pit_window(
        degradation_models=deg_result,  # Pass the unified result directly
        pit_loss_value=pit_loss_value,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        compound=compound,
    )
    optimal_pit_lap, optimal_total_time = find_optimal_pit_lap(strategy_df)

    print(f"  Example scenario: {compound} compound, tyre-life {current_tyre_life}, {laps_remaining} laps remaining")
    print(f"    Optimal pit window: pit in {optimal_pit_lap} laps")
    print(f"    Minimum total time: {optimal_total_time:.2f} s")


    decision = recommend_action(
        degradation_models=deg_result,
        pit_loss_value=pit_loss_value,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        compound=compound,
    )
    print(f"    Decision: {decision}")

    # Phase 2A: Automatic strategy recommendation
    print("\n  [Phase 2A - Automatic Strategy Search]")
    best_plan, all_ranked_plans = recommend_best_strategy(
        degradation_models=deg_result,
        pit_loss_value=pit_loss_value,
        current_compound=compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        candidate_compounds=["SOFT", "MEDIUM", "HARD"],
        include_two_stop=True,
    )

    print(f"\n  Top Recommendation:")
    print(f"    Type: {best_plan.strategy_type.upper()}")
    print(f"    Next Tyre: {best_plan.next_compound}")
    print(f"    First pit: in {best_plan.pit_lap} laps")
    if best_plan.second_pit_lap:
        print(f"    Second pit: {best_plan.second_pit_lap - best_plan.pit_lap} laps after the first stop")
        print(f"    Final Tyre: {best_plan.final_compound}")
    print(f"    Total Time: {best_plan.total_race_time:.2f} s")
    print(f"    Feasible: {'[OK] Yes' if best_plan.feasible else '[!] Check'}")
    print(f"    Rationale: {best_plan.explanation}")
    next_support = deg_result.get_support_info(best_plan.next_compound)
    print(
        f"    Next-tyre support: {next_support.get('support_tier', 'n/a')} "
        f"({next_support.get('prediction_health', 'unknown')})"
    )
    if best_plan.final_compound:
        final_support = deg_result.get_support_info(best_plan.final_compound)
        print(
            f"    Final-tyre support: {final_support.get('support_tier', 'n/a')} "
            f"({final_support.get('prediction_health', 'unknown')})"
        )

    print(f"\n  Top 5 Strategy Options (ranked by time):")
    for i, plan in enumerate(all_ranked_plans[:5], 1):
        feasible_mark = "[OK]" if plan.feasible else "[!]"
        time_diff = plan.total_race_time - best_plan.total_race_time
        diff_str = f"{time_diff:+.2f}s" if time_diff != 0 else "BEST"
        
        if plan.strategy_type == "one-stop":
            plan_support = deg_result.get_support_info(plan.next_compound)
            print(
                f"    {i}. {feasible_mark} ONE-STOP  | "
                f"{plan.current_compound}->{plan.next_compound} @ in {plan.pit_lap} lap{'s' if plan.pit_lap != 1 else ''} | "
                f"Time: {plan.total_race_time:.2f}s ({diff_str}) | "
                f"support: {plan_support.get('support_tier', 'n/a')}"
            )
        else:
            next_support = deg_result.get_support_info(plan.next_compound)
            final_support = deg_result.get_support_info(plan.final_compound)
            print(
                f"    {i}. {feasible_mark} TWO-STOP  | "
                f"{plan.current_compound}->{plan.next_compound}->{plan.final_compound} @ "
                f"in {plan.pit_lap} lap{'s' if plan.pit_lap != 1 else ''}, "
                f"then {plan.second_pit_lap - plan.pit_lap} later | "
                f"Time: {plan.total_race_time:.2f}s ({diff_str}) | "
                f"support: {next_support.get('support_tier', 'n/a')}/{final_support.get('support_tier', 'n/a')}"
            )

    # Show example predictions
    print("\nExample lap-time predictions (MEDIUM compound, using active models):")
    for tyre_life in [1, 5, 10, 15, 20]:
        predicted = deg_result.predict_lap_time("MEDIUM", tyre_life)
        if predicted:
            print(f"  Tyre-life {tyre_life:2d}: {predicted:.2f} s")

    # Phase 2C: Strategy Sensitivity Analysis
    print("\n" + "=" * 80)
    print("PHASE 2C: STRATEGY SENSITIVITY & UNCERTAINTY ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing stability of recommendation under reasonable assumptions...")
    
    try:
        stability_assessment = assess_strategy_stability(
            baseline_plan=best_plan,
            pit_loss_value=pit_loss_value,
            degradation_models=deg_result,
            current_compound=compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
        )
        
        print(f"\n[Baseline Recommendation]")
        print(f"  Strategy: {best_plan.strategy_type.upper()}")
        print(f"  Next Tyre: {best_plan.next_compound}")
        print(f"  First pit: in {best_plan.pit_lap} laps")
        if best_plan.second_pit_lap:
            print(f"  Second pit: {best_plan.second_pit_lap - best_plan.pit_lap} laps after the first stop")
        print(f"  Total Time: {best_plan.total_race_time:.2f}s")
        
        print(f"\n[Stability Assessment]")
        print(f"  Label: {stability_assessment.stability_label}")
        
        print(f"\n[Pit-Loss Sensitivity] (baseline: {pit_loss_value:.2f}s)")
        if stability_assessment.pit_loss_sensitive:
            print(f"  Status: SENSITIVE - recommendation changes under ±1-2s variation")
            for scenario in stability_assessment.pit_loss_sensitivity.scenarios:
                if scenario.recommendation_changed:
                    delta_str = f"{scenario.pit_loss_value - pit_loss_value:+.1f}s"
                    print(
                        f"    ○ At {delta_str}: {scenario.best_plan.next_compound} "
                        f"in {scenario.best_plan.pit_lap} lap{'s' if scenario.best_plan.pit_lap != 1 else ''}"
                    )
        else:
            print(f"  Status: STABLE - recommendation unchanged under ±1-2s variation")
        
        print(f"\n[Degradation Sensitivity] (baseline model)")
        if stability_assessment.degradation_sensitive:
            print(f"  Status: SENSITIVE - recommendation changes under assumption variation")
            for scenario in stability_assessment.degradation_sensitivity.scenarios:
                if scenario.recommendation_changed:
                    scenario_label = "optimistic" if "optimistic" in scenario.scenario_name else "pessimistic"
                    print(
                        f"    ○ {scenario_label.upper()}: {scenario.best_plan.next_compound} "
                        f"in {scenario.best_plan.pit_lap} lap{'s' if scenario.best_plan.pit_lap != 1 else ''}"
                    )
        else:
            print(f"  Status: STABLE - recommendation unchanged across degradation scenarios")
        
        if stability_assessment.flip_conditions:
            print(f"\n[Flip Conditions]")
            for condition in stability_assessment.flip_conditions:
                print(f"  • {condition}")
        else:
            print(f"\n[Flip Conditions]")
            print(f"  • None identified - recommendation is robust")
        
        # Save sensitivity artifact
        artifact_path = ROOT / "data" / "processed" / "phase2c_sensitivity_summary.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(artifact_path, "w") as f:
            json.dump(stability_assessment.to_dict(), f, indent=2)
        print(f"\n  [OK] Sensitivity artifact saved to {artifact_path}")
        
    except Exception as e:
        print(f"\n  [ERROR] Phase 2C analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("PHASE 1 + PHASE 2B + PHASE 2A + PHASE 2C DEMO COMPLETE")
    print("=" * 80)
    print("\nKey points:")
    print("  [OK] Phase 1A data loading (Parquet-first)")
    print("  [OK] Phase 2B hybrid modeling (Miami-specific + 2026 recency blend)")
    print("  [OK] Pre-3 support surfacing (explicit compound support tiers)")
    print("  [OK] Phase 1B fuel correction (automatic)")
    print("  [OK] Phase 1C piecewise degradation (with cliff detection)")
    print("  [OK] Unified prediction interface")
    print("  [OK] Transparent model reporting")
    print("  [OK] Phase 2A automatic strategy search")
    print("  [OK] One-stop vs two-stop evaluation")
    print("  [OK] Ranked strategy recommendations")
    print("  [OK] Phase 2C sensitivity analysis (pit-loss & degradation)")
    print("  [OK] Phase 2D validation available via scripts/run_phase2d_validation.py")
    print("  [OK] Phase 2E calibration (race-local pit loss + cleaner search)")
    print("\nPhase 2B enables:")
    print("  - Miami-specific circuit anchor for degradation and pit loss")
    print("  - Current-season recency as bounded support/adjustment")
    print("  - Explicit support tiers per compound")
    print("  - Less overclaiming than a flat cross-circuit truth blend")
    print("\nPhase 2C enables:")
    print("  - Scenario-based sensitivity analysis (NOT probabilistic)")
    print("  - Pit-loss assumption testing (±1-2 seconds)")
    print("  - Degradation assumption testing (optimistic/pessimistic)")
    print("  - Stability classification (Stable / Moderately Sensitive / Fragile)")
    print("  - Identification of flip conditions")
    print("\nFor broader validation across representative race states:")
    print("  - Run: python scripts/run_phase2d_validation.py")
    print("  - Review: data/processed/phase2d_validation_summary.json")



if __name__ == "__main__":
    main()

