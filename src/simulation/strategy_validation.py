"""Phase 2D: broader validation / robustness evaluation helpers.

This module defines a compact representative scenario suite and utilities to
run the existing strategy engine plus Phase 2C stability analysis across it.
The goal is to evaluate recommendation behavior across a broader set of race
states without changing the core optimizer.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.simulation.strategy import predict_lap_time
from src.simulation.strategy_engine import StrategyPlan, recommend_best_strategy
from src.simulation.strategy_sensitivity import StrategyStabilityAssessment, assess_strategy_stability


@dataclass(frozen=True)
class ValidationScenario:
    """A representative strategy-validation race state."""

    scenario_id: str
    current_compound: str
    current_tyre_life: int
    tyre_age_bucket: str
    laps_remaining: int
    laps_remaining_bucket: str
    rationale: str


@dataclass(frozen=True)
class BacktestCheckpoint:
    """A held-out Miami checkpoint used in the Pre-3 backtest."""

    checkpoint_id: str
    season: int
    driver: str
    checkpoint_lap: int
    rationale: str


def build_representative_scenario_suite() -> List[ValidationScenario]:
    """Return a compact, representative Phase 2D scenario suite.

    The suite covers all current compounds and mixes low/medium/high tyre age
    with short/medium/long remaining-race states without enumerating every
    possible combination.
    """
    return [
        ValidationScenario(
            scenario_id="soft_low_short",
            current_compound="SOFT",
            current_tyre_life=3,
            tyre_age_bucket="low",
            laps_remaining=8,
            laps_remaining_bucket="short",
            rationale="Fresh soft near the end of the race.",
        ),
        ValidationScenario(
            scenario_id="soft_medium_medium",
            current_compound="SOFT",
            current_tyre_life=8,
            tyre_age_bucket="medium",
            laps_remaining=18,
            laps_remaining_bucket="medium",
            rationale="Mid-life soft with a standard remaining stint length.",
        ),
        ValidationScenario(
            scenario_id="soft_high_long",
            current_compound="SOFT",
            current_tyre_life=14,
            tyre_age_bucket="high",
            laps_remaining=32,
            laps_remaining_bucket="long",
            rationale="Stress case: aged soft asked to cover a long remaining race.",
        ),
        ValidationScenario(
            scenario_id="soft_high_short",
            current_compound="SOFT",
            current_tyre_life=15,
            tyre_age_bucket="high",
            laps_remaining=10,
            laps_remaining_bucket="short",
            rationale="Late-stint soft where immediate pit pressure should be visible.",
        ),
        ValidationScenario(
            scenario_id="medium_low_medium",
            current_compound="MEDIUM",
            current_tyre_life=4,
            tyre_age_bucket="low",
            laps_remaining=18,
            laps_remaining_bucket="medium",
            rationale="Fresh-medium baseline race state.",
        ),
        ValidationScenario(
            scenario_id="medium_medium_long",
            current_compound="MEDIUM",
            current_tyre_life=10,
            tyre_age_bucket="medium",
            laps_remaining=32,
            laps_remaining_bucket="long",
            rationale="Typical one-stop vs two-stop trade-off zone.",
        ),
        ValidationScenario(
            scenario_id="medium_high_short",
            current_compound="MEDIUM",
            current_tyre_life=18,
            tyre_age_bucket="high",
            laps_remaining=8,
            laps_remaining_bucket="short",
            rationale="Worn medium with little race left.",
        ),
        ValidationScenario(
            scenario_id="medium_high_medium",
            current_compound="MEDIUM",
            current_tyre_life=18,
            tyre_age_bucket="high",
            laps_remaining=18,
            laps_remaining_bucket="medium",
            rationale="Aged medium with enough distance left to expose instability.",
        ),
        ValidationScenario(
            scenario_id="hard_low_long",
            current_compound="HARD",
            current_tyre_life=4,
            tyre_age_bucket="low",
            laps_remaining=32,
            laps_remaining_bucket="long",
            rationale="Fresh hard asked to carry a long closing stint.",
        ),
        ValidationScenario(
            scenario_id="hard_medium_short",
            current_compound="HARD",
            current_tyre_life=10,
            tyre_age_bucket="medium",
            laps_remaining=8,
            laps_remaining_bucket="short",
            rationale="Hard tyre with short sprint remaining.",
        ),
        ValidationScenario(
            scenario_id="hard_high_medium",
            current_compound="HARD",
            current_tyre_life=22,
            tyre_age_bucket="high",
            laps_remaining=18,
            laps_remaining_bucket="medium",
            rationale="Used hard in a medium-distance finish window.",
        ),
        ValidationScenario(
            scenario_id="hard_high_long",
            current_compound="HARD",
            current_tyre_life=26,
            tyre_age_bucket="high",
            laps_remaining=32,
            laps_remaining_bucket="long",
            rationale="Stress case: very used hard with long distance left.",
        ),
        ValidationScenario(
            scenario_id="soft_extreme_long",
            current_compound="SOFT",
            current_tyre_life=18,
            tyre_age_bucket="high",
            laps_remaining=54,
            laps_remaining_bucket="long",
            rationale="Extreme long-distance stress case that should force the two-stop path to be exercised.",
        ),
        ValidationScenario(
            scenario_id="hard_extreme_long",
            current_compound="HARD",
            current_tyre_life=30,
            tyre_age_bucket="high",
            laps_remaining=55,
            laps_remaining_bucket="long",
            rationale="Extreme hard-tyre stress case added to check whether two-stop search is reachable at all.",
        ),
    ]


def build_pre3_backtest_checkpoint_suite() -> List[BacktestCheckpoint]:
    """Return the canonical held-out checkpoint suite for Pre-3 backtesting.

    The suite stays inside Miami 2024 because the local Miami 2025 race file
    still has large early-race compound gaps (`nan` compounds through laps
    23-24 for many drivers), which makes first-stint checkpoint selection much
    less trustworthy for a deterministic strategy audit.
    """
    return [
        BacktestCheckpoint(
            checkpoint_id="bot_soft_early",
            season=2024,
            driver="BOT",
            checkpoint_lap=5,
            rationale="Early first-stint SOFT checkpoint in a real two-stop race.",
        ),
        BacktestCheckpoint(
            checkpoint_id="lec_medium_mid",
            season=2024,
            driver="LEC",
            checkpoint_lap=12,
            rationale="Mid first-stint MEDIUM checkpoint in a one-stop race.",
        ),
        BacktestCheckpoint(
            checkpoint_id="ham_hard_mid",
            season=2024,
            driver="HAM",
            checkpoint_lap=15,
            rationale="HARD-compound checkpoint before a medium switch.",
        ),
        BacktestCheckpoint(
            checkpoint_id="nor_medium_late",
            season=2024,
            driver="NOR",
            checkpoint_lap=24,
            rationale="Late first-stint MEDIUM checkpoint close to the actual stop.",
        ),
        BacktestCheckpoint(
            checkpoint_id="alo_hard_mid",
            season=2024,
            driver="ALO",
            checkpoint_lap=15,
            rationale="Second HARD-to-MEDIUM one-stop case to test whether the long-HARD miss pattern repeats.",
        ),
        BacktestCheckpoint(
            checkpoint_id="gas_medium_mid",
            season=2024,
            driver="GAS",
            checkpoint_lap=8,
            rationale="Medium-to-HARD one-stop checkpoint expected to behave like a timing-heavy near-equivalent case.",
        ),
        BacktestCheckpoint(
            checkpoint_id="per_medium_mid",
            season=2024,
            driver="PER",
            checkpoint_lap=12,
            rationale="Medium-start two-stop checkpoint with the same first future compound as the model, but an extra real stop later.",
        ),
        BacktestCheckpoint(
            checkpoint_id="pia_medium_late",
            season=2024,
            driver="PIA",
            checkpoint_lap=19,
            rationale="Late first-stint MEDIUM checkpoint inside a real two-stop race.",
        ),
        BacktestCheckpoint(
            checkpoint_id="ver_medium_mid",
            season=2024,
            driver="VER",
            checkpoint_lap=16,
            rationale="Another late first-stint MEDIUM-to-HARD one-stop checkpoint for timing-band validation.",
        ),
        BacktestCheckpoint(
            checkpoint_id="zho_medium_late_soft",
            season=2024,
            driver="ZHO",
            checkpoint_lap=20,
            rationale="Late first-stint MEDIUM checkpoint that finished on SOFT in the real race, included to test soft-finish compound selection.",
        ),
    ]


def _plan_to_dict(plan: StrategyPlan) -> dict:
    """Serialize a StrategyPlan to a JSON-friendly dict."""
    return {
        "strategy_type": plan.strategy_type,
        "current_compound": plan.current_compound,
        "current_tyre_life": plan.current_tyre_life,
        "next_compound": plan.next_compound,
        "final_compound": plan.final_compound,
        "pit_lap": plan.pit_lap,
        "second_pit_lap": plan.second_pit_lap,
        "total_race_time": round(plan.total_race_time, 3),
        "feasible": plan.feasible,
        "feasibility_reason": plan.feasibility_reason,
        "one_stop_estimate": round(plan.one_stop_estimate, 3) if plan.one_stop_estimate is not None else None,
        "explanation": plan.explanation,
    }


def _strategy_stint_spec(plan: StrategyPlan) -> List[dict]:
    """Return ordered stint specifications for a strategy plan."""
    stints = [
        {
            "label": "current_stint",
            "compound": plan.current_compound,
            "starting_tyre_life": plan.current_tyre_life,
            "laps": plan.pit_lap,
        }
    ]
    if plan.strategy_type == "one-stop":
        stints.append(
            {
                "label": "finish_stint",
                "compound": plan.next_compound,
                "starting_tyre_life": 1,
                "laps": max(0, plan.total_laps_remaining - plan.pit_lap) if hasattr(plan, "total_laps_remaining") else None,
            }
        )
        return stints

    stints.append(
        {
            "label": "middle_stint",
            "compound": plan.next_compound,
            "starting_tyre_life": 1,
            "laps": plan.second_pit_lap - plan.pit_lap,
        }
    )
    stints.append(
        {
            "label": "finish_stint",
            "compound": plan.final_compound,
            "starting_tyre_life": 1,
            "laps": None,
        }
    )
    return stints


def decompose_strategy_plan(
    degradation_models: object,
    pit_loss_value: float,
    plan: StrategyPlan,
    laps_remaining: int,
) -> dict:
    """Return a stint-by-stint time decomposition for a strategy plan."""
    remaining_after_first_pit = laps_remaining - plan.pit_lap
    stints = [
        {
            "label": "current_stint",
            "compound": plan.current_compound,
            "starting_tyre_life": plan.current_tyre_life,
            "laps": plan.pit_lap,
        }
    ]

    if plan.strategy_type == "one-stop":
        stints.append(
            {
                "label": "finish_stint",
                "compound": plan.next_compound,
                "starting_tyre_life": 1,
                "laps": remaining_after_first_pit,
            }
        )
    else:
        second_stint_laps = plan.second_pit_lap - plan.pit_lap
        final_stint_laps = laps_remaining - plan.second_pit_lap
        stints.extend(
            [
                {
                    "label": "middle_stint",
                    "compound": plan.next_compound,
                    "starting_tyre_life": 1,
                    "laps": second_stint_laps,
                },
                {
                    "label": "finish_stint",
                    "compound": plan.final_compound,
                    "starting_tyre_life": 1,
                    "laps": final_stint_laps,
                },
            ]
        )

    stint_rows = []
    total_stint_time = 0.0
    soft_stint_time = 0.0
    for stint in stints:
        stint_time = 0.0
        for lap in range(stint["laps"]):
            tyre_life = stint["starting_tyre_life"] + lap
            lap_time = predict_lap_time(
                degradation_models,
                stint["compound"],
                tyre_life,
            )
            if lap_time is None:
                raise ValueError(
                    f"Could not decompose {plan.strategy_type} plan because "
                    f"{stint['compound']} returned no prediction at tyre life {tyre_life}."
                )
            stint_time += float(lap_time)
        if stint["compound"] == "SOFT":
            soft_stint_time += stint_time
        total_stint_time += stint_time
        stint_rows.append(
            {
                **stint,
                "stint_time_s": round(stint_time, 3),
            }
        )

    pit_stop_count = 1 if plan.strategy_type == "one-stop" else 2
    pit_loss_total = float(pit_loss_value) * pit_stop_count
    return {
        "strategy_type": plan.strategy_type,
        "pit_stop_count": pit_stop_count,
        "pit_loss_total_s": round(pit_loss_total, 3),
        "stint_time_total_s": round(total_stint_time, 3),
        "soft_stint_time_total_s": round(soft_stint_time, 3),
        "stints": stint_rows,
        "reconstructed_total_s": round(total_stint_time + pit_loss_total, 3),
    }


def _first_non_soft_alternative(ranked_plans: List[StrategyPlan]) -> Optional[StrategyPlan]:
    """Return the best-ranked plan that does not use SOFT in a future stint."""
    for plan in ranked_plans:
        if plan.next_compound != "SOFT" and plan.final_compound != "SOFT":
            return plan
    return None


def _soft_prediction_signal(degradation_models: object) -> dict:
    """Summarize the active SOFT prediction shape and recency adjustment."""
    lap_probe_points = [1, 5, 10, 15, 20]
    active_curve = {
        str(lap): round(float(degradation_models.predict_lap_time("SOFT", lap)), 3)
        for lap in lap_probe_points
    }
    signal = {
        "active_soft_curve_s": active_curve,
        "soft_prediction_improves_with_tyre_life": active_curve["20"] < active_curve["1"],
    }

    if hasattr(degradation_models, "miami_models") and hasattr(degradation_models, "recency_models"):
        miami_curve = {
            str(lap): round(float(degradation_models.miami_models.predict_lap_time("SOFT", lap)), 3)
            for lap in lap_probe_points
        }
        recency_curve = {
            str(lap): round(float(degradation_models.recency_models.predict_lap_time("SOFT", lap)), 3)
            for lap in lap_probe_points
        }
        support_info = degradation_models.get_support_info("SOFT")
        weight = float(support_info.get("hybrid_adjustment_weight", 0.0) or 0.0)
        cap = float(degradation_models.get_model_info("SOFT").get("hybrid_adjustment_cap", 0.0) or 0.0)
        weighted_delta = {
            str(lap): round(weight * (recency_curve[str(lap)] - miami_curve[str(lap)]), 3)
            for lap in lap_probe_points
        }
        signal.update(
            {
                "miami_soft_curve_s": miami_curve,
                "recency_soft_curve_s": recency_curve,
                "weighted_soft_recency_delta_s": weighted_delta,
                "soft_recency_adjustment_cap_s": cap,
                "soft_recency_adjustment_cap_hit": any(abs(delta) >= cap for delta in weighted_delta.values()) if cap > 0 else False,
            }
        )
    return signal


def _soft_inspection_verdict(
    margin_s: float,
    support_tier: Optional[str],
    uses_long_soft_finish: bool,
) -> str:
    """Return a simple human-readable verdict for a SOFT-winning plan."""
    if margin_s < 1.0:
        return "Borderline: SOFT only wins by a sub-second margin."
    if support_tier in {"Low", "Moderate"} and uses_long_soft_finish:
        return "Cautious but defensible: SOFT wins by pace, but the finish stint leans on weaker-support data."
    if support_tier in {"Low", "Moderate"}:
        return "Mixed: SOFT wins on projected time, but support remains below the HARD/MEDIUM standard."
    return "Defensible: SOFT wins with enough margin and support to justify the recommendation."


def build_soft_preference_diagnostic(
    degradation_models: object,
    pit_loss_value: float,
    scenarios: Optional[Iterable[ValidationScenario]] = None,
    candidate_compounds: Optional[List[str]] = None,
    include_two_stop: bool = True,
) -> dict:
    """Build a focused diagnostic summary for states where SOFT is selected."""
    if scenarios is None:
        scenarios = build_representative_scenario_suite()
    scenario_list = list(scenarios)
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    soft_cases: List[dict] = []
    for scenario in scenario_list:
        best_plan, ranked_plans = recommend_best_strategy(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_compound=scenario.current_compound,
            current_tyre_life=scenario.current_tyre_life,
            laps_remaining=scenario.laps_remaining,
            candidate_compounds=candidate_compounds,
            include_two_stop=include_two_stop,
        )
        if best_plan.next_compound != "SOFT" and best_plan.final_compound != "SOFT":
            continue

        alt_plan = _first_non_soft_alternative(ranked_plans)
        if alt_plan is None:
            continue

        best_breakdown = decompose_strategy_plan(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            plan=best_plan,
            laps_remaining=scenario.laps_remaining,
        )
        alt_breakdown = decompose_strategy_plan(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            plan=alt_plan,
            laps_remaining=scenario.laps_remaining,
        )
        margin_s = float(alt_plan.total_race_time - best_plan.total_race_time)
        pace_advantage_s = round(
            alt_breakdown["stint_time_total_s"] - best_breakdown["stint_time_total_s"],
            3,
        )
        pit_loss_delta_s = round(
            best_breakdown["pit_loss_total_s"] - alt_breakdown["pit_loss_total_s"],
            3,
        )
        soft_support = degradation_models.get_support_info("SOFT")
        uses_long_soft_finish = any(
            stint["compound"] == "SOFT" and stint["laps"] >= 18
            for stint in best_breakdown["stints"]
            if stint["label"] != "current_stint"
        )
        if pit_loss_delta_s == 0:
            why_soft_won = "SOFT won on projected stint pace with pit loss held equal."
        elif pace_advantage_s > pit_loss_delta_s:
            why_soft_won = "SOFT pace advantage outweighed the extra pit-loss burden of the alternative structure."
        else:
            why_soft_won = "SOFT still won after accounting for extra pit loss, but the margin is mostly structure-driven."

        soft_cases.append(
            {
                "scenario_id": scenario.scenario_id,
                "current_state": {
                    "current_compound": scenario.current_compound,
                    "current_tyre_life": scenario.current_tyre_life,
                    "laps_remaining": scenario.laps_remaining,
                    "tyre_age_bucket": scenario.tyre_age_bucket,
                    "laps_remaining_bucket": scenario.laps_remaining_bucket,
                },
                "best_soft_plan": _plan_to_dict(best_plan),
                "nearby_non_soft_alternative": _plan_to_dict(alt_plan),
                "margin_vs_nearest_non_soft_s": round(margin_s, 3),
                "time_decomposition": {
                    "best_soft_plan": best_breakdown,
                    "nearby_non_soft_alternative": alt_breakdown,
                    "net_advantage_s": round(margin_s, 3),
                    "stint_time_advantage_s": pace_advantage_s,
                    "pit_loss_delta_s": pit_loss_delta_s,
                },
                "support_context": {
                    "soft_support_tier": soft_support.get("support_tier"),
                    "soft_support_reason": soft_support.get("support_reason"),
                    "best_plan_supports": {
                        "next": degradation_models.get_support_info(best_plan.next_compound),
                        "final": degradation_models.get_support_info(best_plan.final_compound) if best_plan.final_compound else None,
                    },
                    "alternative_supports": {
                        "next": degradation_models.get_support_info(alt_plan.next_compound),
                        "final": degradation_models.get_support_info(alt_plan.final_compound) if alt_plan.final_compound else None,
                    },
                },
                "why_soft_won": why_soft_won,
                "inspection_verdict": _soft_inspection_verdict(
                    margin_s=margin_s,
                    support_tier=soft_support.get("support_tier"),
                    uses_long_soft_finish=uses_long_soft_finish,
                ),
            }
        )

    model_info = degradation_models.get_model_info("SOFT")
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "pre3",
        "pit_loss_value": round(float(pit_loss_value), 3),
        "soft_model_info": model_info,
        "soft_prediction_signal": _soft_prediction_signal(degradation_models),
        "soft_selected_case_count": len(soft_cases),
        "average_margin_vs_nearest_non_soft_s": round(
            float(pd.Series([case["margin_vs_nearest_non_soft_s"] for case in soft_cases]).mean()),
            3,
        ) if soft_cases else None,
        "cases": soft_cases,
    }
    return summary


def save_soft_preference_diagnostic(
    diagnostic_report: dict,
    output_path: Path | str,
) -> Path:
    """Save the SOFT preference diagnostic artifact."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(diagnostic_report, handle, indent=2)
    return output


def _identify_scenario_warnings(
    scenario: ValidationScenario,
    best_plan: StrategyPlan,
    ranked_plans: List[StrategyPlan],
    stability: Optional[StrategyStabilityAssessment],
    model_info: dict,
    next_support_info: Optional[dict],
    final_support_info: Optional[dict],
) -> List[str]:
    """Return scenario-level warnings worth surfacing in artifacts."""
    warnings: List[str] = []
    if not best_plan.feasible:
        warnings.append("best-plan-infeasible")
    if stability and stability.stability_label == "Fragile":
        warnings.append("fragile-recommendation")
    if len(ranked_plans) > 1:
        margin = ranked_plans[1].total_race_time - best_plan.total_race_time
        if margin < 1.0:
            warnings.append("small-margin-to-runner-up")
    if scenario.current_compound == "SOFT":
        samples = model_info.get("samples", 0)
        model_type = (model_info.get("model_type") or "").upper()
        if samples < 100 or "FALLBACK" in model_type:
            warnings.append("soft-weak-data-signal")
    for support_info in [next_support_info, final_support_info]:
        if not support_info:
            continue
        tier = support_info.get("support_tier")
        compound = support_info.get("compound")
        if tier == "Low":
            warnings.append(f"{compound.lower()}-low-support-plan")
        elif tier == "Moderate":
            warnings.append(f"{compound.lower()}-moderate-support-plan")
    return warnings


def run_strategy_validation_suite(
    degradation_models: object,
    pit_loss_value: float,
    scenarios: Optional[Iterable[ValidationScenario]] = None,
    candidate_compounds: Optional[List[str]] = None,
    include_two_stop: bool = True,
    include_stability_analysis: bool = True,
) -> dict:
    """Run the strategy engine across a representative scenario suite."""
    if scenarios is None:
        scenarios = build_representative_scenario_suite()
    scenario_list = list(scenarios)
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    scenario_results = []

    for scenario in scenario_list:
        best_plan, ranked_plans = recommend_best_strategy(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_compound=scenario.current_compound,
            current_tyre_life=scenario.current_tyre_life,
            laps_remaining=scenario.laps_remaining,
            candidate_compounds=candidate_compounds,
            include_two_stop=include_two_stop,
        )

        stability = None
        if include_stability_analysis:
            stability = assess_strategy_stability(
                baseline_plan=best_plan,
                pit_loss_value=pit_loss_value,
                degradation_models=degradation_models,
                current_compound=scenario.current_compound,
                current_tyre_life=scenario.current_tyre_life,
                laps_remaining=scenario.laps_remaining,
            )

        current_model_info = degradation_models.get_model_info(scenario.current_compound)
        next_support_info = (
            degradation_models.get_support_info(best_plan.next_compound)
            if hasattr(degradation_models, "get_support_info")
            else None
        )
        final_support_info = (
            degradation_models.get_support_info(best_plan.final_compound)
            if best_plan.final_compound and hasattr(degradation_models, "get_support_info")
            else None
        )
        runner_up_gap = None
        if len(ranked_plans) > 1:
            runner_up_gap = float(ranked_plans[1].total_race_time - best_plan.total_race_time)
        best_one_stop_plan = next((plan for plan in ranked_plans if plan.strategy_type == "one-stop"), None)
        best_two_stop_plan = next((plan for plan in ranked_plans if plan.strategy_type == "two-stop"), None)

        warnings = _identify_scenario_warnings(
            scenario=scenario,
            best_plan=best_plan,
            ranked_plans=ranked_plans,
            stability=stability,
            model_info=current_model_info,
            next_support_info=next_support_info,
            final_support_info=final_support_info,
        )

        scenario_results.append(
            {
                "scenario_id": scenario.scenario_id,
                "current_compound": scenario.current_compound,
                "current_tyre_life": scenario.current_tyre_life,
                "tyre_age_bucket": scenario.tyre_age_bucket,
                "laps_remaining": scenario.laps_remaining,
                "laps_remaining_bucket": scenario.laps_remaining_bucket,
                "rationale": scenario.rationale,
                "best_plan": _plan_to_dict(best_plan),
                "runner_up_gap_s": round(runner_up_gap, 3) if runner_up_gap is not None else None,
                "best_one_stop_plan": _plan_to_dict(best_one_stop_plan) if best_one_stop_plan else None,
                "best_two_stop_plan": _plan_to_dict(best_two_stop_plan) if best_two_stop_plan else None,
                "best_two_stop_gap_s": (
                    round(float(best_two_stop_plan.total_race_time - best_plan.total_race_time), 3)
                    if best_two_stop_plan is not None
                    else None
                ),
                "top_3_plans": [_plan_to_dict(plan) for plan in ranked_plans[:3]],
                "stability_label": stability.stability_label if stability else None,
                "pit_loss_sensitive": stability.pit_loss_sensitive if stability else None,
                "degradation_sensitive": stability.degradation_sensitive if stability else None,
                "flip_conditions": stability.flip_conditions if stability else [],
                "warnings": warnings,
                "current_compound_model_info": current_model_info,
                "current_compound_support_info": (
                    degradation_models.get_support_info(scenario.current_compound)
                    if hasattr(degradation_models, "get_support_info")
                    else None
                ),
                "next_compound_support_info": next_support_info,
                "final_compound_support_info": final_support_info,
            }
        )

    summary = summarize_validation_results(
        scenario_results=scenario_results,
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
    )

    return {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "phase": "2D",
            "scenario_count": len(scenario_results),
            "pit_loss_value": round(float(pit_loss_value), 3),
            "candidate_compounds": list(candidate_compounds),
            "include_two_stop": include_two_stop,
            "include_stability_analysis": include_stability_analysis,
        },
        "scenarios": scenario_results,
        "aggregate_summary": summary,
    }


def summarize_validation_results(
    scenario_results: List[dict],
    degradation_models: object,
    pit_loss_value: float,
) -> dict:
    """Build aggregate robustness summary for the Phase 2D artifact."""
    strategy_type_counts: Counter = Counter()
    next_tyre_counts: Counter = Counter()
    stability_counts: Counter = Counter()
    fragile_by_compound: Counter = Counter()
    fragile_by_laps_bucket: Counter = Counter()
    fragile_by_age_bucket: Counter = Counter()
    warning_counts: Counter = Counter()
    unstable_states: defaultdict[str, List[str]] = defaultdict(list)
    pathological_cases: List[dict] = []
    low_confidence_counts: Counter = Counter()
    two_stop_available_count = 0
    two_stop_runner_up_count = 0
    two_stop_within_2s_count = 0
    two_stop_within_5s_count = 0
    two_stop_selected_ids: List[str] = []

    for result in scenario_results:
        best_plan = result["best_plan"]
        strategy_type_counts[best_plan["strategy_type"]] += 1
        next_tyre_counts[best_plan["next_compound"]] += 1
        best_two_stop_gap = result.get("best_two_stop_gap_s")
        best_two_stop_plan = result.get("best_two_stop_plan")
        if best_two_stop_plan is not None:
            two_stop_available_count += 1
            if len(result.get("top_3_plans", [])) > 1 and result["top_3_plans"][1]["strategy_type"] == "two-stop":
                two_stop_runner_up_count += 1
            if best_two_stop_gap is not None and best_two_stop_gap <= 2.0:
                two_stop_within_2s_count += 1
            if best_two_stop_gap is not None and best_two_stop_gap <= 5.0:
                two_stop_within_5s_count += 1
        if best_plan["strategy_type"] == "two-stop":
            two_stop_selected_ids.append(result["scenario_id"])

        stability_label = result.get("stability_label") or "Not Assessed"
        stability_counts[stability_label] += 1

        for warning in result.get("warnings", []):
            warning_counts[warning] += 1
            if warning.endswith("-low-support-plan") or warning.endswith("-moderate-support-plan"):
                low_confidence_counts[warning] += 1

        if stability_label in {"Moderately Sensitive", "Fragile"}:
            unstable_states[result["current_compound"]].append(result["scenario_id"])

        if stability_label == "Fragile":
            fragile_by_compound[result["current_compound"]] += 1
            fragile_by_laps_bucket[result["laps_remaining_bucket"]] += 1
            fragile_by_age_bucket[result["tyre_age_bucket"]] += 1

        runner_up_gap = result.get("runner_up_gap_s")
        if (not best_plan["feasible"]) or (runner_up_gap is not None and runner_up_gap < 0.5):
            pathological_cases.append(
                {
                    "scenario_id": result["scenario_id"],
                    "issue": "infeasible-best-plan" if not best_plan["feasible"] else "runner-up-within-0.5s",
                    "strategy_type": best_plan["strategy_type"],
                    "next_compound": best_plan["next_compound"],
                    "pit_lap": best_plan["pit_lap"],
                    "warnings": result.get("warnings", []),
                }
            )

    model_info = {
        compound: degradation_models.get_model_info(compound)
        for compound in ["SOFT", "MEDIUM", "HARD"]
    }
    prediction_health = {
        compound: {
            "lap_1": degradation_models.predict_lap_time(compound, 1),
            "lap_5": degradation_models.predict_lap_time(compound, 5),
            "lap_10": degradation_models.predict_lap_time(compound, 10),
        }
        for compound in ["SOFT", "MEDIUM", "HARD"]
    }
    soft_info = model_info["SOFT"]
    soft_support_info = (
        degradation_models.get_support_info("SOFT")
        if hasattr(degradation_models, "get_support_info")
        else {}
    )
    soft_predictions = prediction_health["SOFT"]
    soft_prediction_invalid = any(value is None for value in soft_predictions.values())
    soft_weak_data_signal = (
        soft_support_info.get("support_tier") == "Low"
        or soft_info.get("samples", 0) < 100
        or "FALLBACK" in str(soft_info.get("miami_model_type") or "").upper()
        or warning_counts.get("soft-weak-data-signal", 0) > 0
        or soft_prediction_invalid
    )

    observations: List[str] = []
    if strategy_type_counts:
        dominant_type, dominant_count = strategy_type_counts.most_common(1)[0]
        observations.append(
            f"{dominant_type} recommendations dominated {dominant_count}/{len(scenario_results)} representative scenarios."
        )
    if stability_counts.get("Fragile", 0):
        observations.append(
            "Fragility concentrated in "
            f"{', '.join(sorted(k for k, v in fragile_by_compound.items() if v > 0))} "
            "and tended to appear in older-tyre or longer-remaining race states."
        )
    else:
        observations.append("No scenarios were labeled Fragile under the current Phase 2C sensitivity checks.")
    if unstable_states:
        unstable_desc = ", ".join(
            f"{compound} ({len(ids)})" for compound, ids in sorted(unstable_states.items())
        )
        observations.append(f"Unstable recommendations appeared for current compounds: {unstable_desc}.")
    observations.append(
        "SOFT weak-data check: "
        + (
            "warning signals remain present."
            if soft_weak_data_signal
            else "no obvious weak-data warning remains in the active model/reporting path."
        )
    )
    if two_stop_selected_ids:
        observations.append(
            "Two-stop recommendations now appear in explicit long-distance stress states: "
            + ", ".join(two_stop_selected_ids)
            + "."
        )
    if two_stop_available_count and two_stop_within_5s_count == 0:
        observations.append(
            "Even when two-stop plans are feasible, they rarely come within 5s of the one-stop optimum under the current pit-loss and degradation assumptions."
        )
    elif two_stop_within_5s_count:
        observations.append(
            f"Two-stop plans came within 5s of the best plan in {two_stop_within_5s_count}/{len(scenario_results)} scenarios."
        )

    return {
        "pit_loss_value": round(float(pit_loss_value), 3),
        "strategy_type_counts": dict(strategy_type_counts),
        "next_tyre_counts": dict(next_tyre_counts),
        "stability_counts": dict(stability_counts),
        "fragile_by_current_compound": dict(fragile_by_compound),
        "fragile_by_laps_remaining_bucket": dict(fragile_by_laps_bucket),
        "fragile_by_tyre_age_bucket": dict(fragile_by_age_bucket),
        "warning_counts": dict(warning_counts),
        "low_confidence_plan_counts": dict(low_confidence_counts),
        "unstable_states_by_current_compound": dict(unstable_states),
        "pathological_cases": pathological_cases,
        "strategy_mix_analysis": {
            "two_stop_available_count": two_stop_available_count,
            "two_stop_runner_up_count": two_stop_runner_up_count,
            "two_stop_within_2s_count": two_stop_within_2s_count,
            "two_stop_within_5s_count": two_stop_within_5s_count,
            "two_stop_selected_scenarios": two_stop_selected_ids,
        },
        "active_model_info": model_info,
        "prediction_health": prediction_health,
        "soft_compound_assessment": {
            "samples": soft_info.get("samples", 0),
            "model_type": soft_info.get("model_type"),
            "support_tier": soft_support_info.get("support_tier"),
            "support_reason": soft_support_info.get("support_reason"),
            "weak_data_signal": soft_weak_data_signal,
            "scenario_warning_count": warning_counts.get("soft-weak-data-signal", 0),
            "prediction_invalid": soft_prediction_invalid,
        },
        "observations": observations,
    }


def validation_results_to_frame(validation_report: dict) -> pd.DataFrame:
    """Flatten scenario results to a DataFrame for CSV export."""
    rows = []
    for result in validation_report["scenarios"]:
        best_plan = result["best_plan"]
        rows.append(
            {
                "scenario_id": result["scenario_id"],
                "current_compound": result["current_compound"],
                "current_tyre_life": result["current_tyre_life"],
                "tyre_age_bucket": result["tyre_age_bucket"],
                "laps_remaining": result["laps_remaining"],
                "laps_remaining_bucket": result["laps_remaining_bucket"],
                "best_strategy_type": best_plan["strategy_type"],
                "next_tyre": best_plan["next_compound"],
                "final_tyre": best_plan["final_compound"],
                "pit_lap": best_plan["pit_lap"],
                "second_pit_lap": best_plan["second_pit_lap"],
                "estimated_total_time": best_plan["total_race_time"],
                "feasible": best_plan["feasible"],
                "feasibility_reason": best_plan["feasibility_reason"],
                "stability_label": result["stability_label"],
                "pit_loss_sensitive": result["pit_loss_sensitive"],
                "degradation_sensitive": result["degradation_sensitive"],
                "runner_up_gap_s": result["runner_up_gap_s"],
                "warnings": "|".join(result["warnings"]),
            }
        )
    return pd.DataFrame(rows)


def save_validation_artifacts(
    validation_report: dict,
    json_path: Path | str,
    csv_path: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Save JSON and optional CSV validation artifacts."""
    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as handle:
        json.dump(validation_report, handle, indent=2)

    saved_paths = {"json": json_output}
    if csv_path is not None:
        csv_output = Path(csv_path)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        validation_results_to_frame(validation_report).to_csv(csv_output, index=False)
        saved_paths["csv"] = csv_output

    return saved_paths
