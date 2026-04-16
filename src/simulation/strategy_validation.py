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


def _identify_scenario_warnings(
    scenario: ValidationScenario,
    best_plan: StrategyPlan,
    ranked_plans: List[StrategyPlan],
    stability: Optional[StrategyStabilityAssessment],
    model_info: dict,
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
        runner_up_gap = None
        if len(ranked_plans) > 1:
            runner_up_gap = float(ranked_plans[1].total_race_time - best_plan.total_race_time)

        warnings = _identify_scenario_warnings(
            scenario=scenario,
            best_plan=best_plan,
            ranked_plans=ranked_plans,
            stability=stability,
            model_info=current_model_info,
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
                "top_3_plans": [_plan_to_dict(plan) for plan in ranked_plans[:3]],
                "stability_label": stability.stability_label if stability else None,
                "pit_loss_sensitive": stability.pit_loss_sensitive if stability else None,
                "degradation_sensitive": stability.degradation_sensitive if stability else None,
                "flip_conditions": stability.flip_conditions if stability else [],
                "warnings": warnings,
                "current_compound_model_info": current_model_info,
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

    for result in scenario_results:
        best_plan = result["best_plan"]
        strategy_type_counts[best_plan["strategy_type"]] += 1
        next_tyre_counts[best_plan["next_compound"]] += 1

        stability_label = result.get("stability_label") or "Not Assessed"
        stability_counts[stability_label] += 1

        for warning in result.get("warnings", []):
            warning_counts[warning] += 1

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
    soft_predictions = prediction_health["SOFT"]
    soft_prediction_invalid = any(value is None for value in soft_predictions.values())
    soft_weak_data_signal = (
        soft_info.get("samples", 0) < 100
        or "FALLBACK" in (soft_info.get("model_type") or "").upper()
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

    return {
        "pit_loss_value": round(float(pit_loss_value), 3),
        "strategy_type_counts": dict(strategy_type_counts),
        "next_tyre_counts": dict(next_tyre_counts),
        "stability_counts": dict(stability_counts),
        "fragile_by_current_compound": dict(fragile_by_compound),
        "fragile_by_laps_remaining_bucket": dict(fragile_by_laps_bucket),
        "fragile_by_tyre_age_bucket": dict(fragile_by_age_bucket),
        "warning_counts": dict(warning_counts),
        "unstable_states_by_current_compound": dict(unstable_states),
        "pathological_cases": pathological_cases,
        "active_model_info": model_info,
        "prediction_health": prediction_health,
        "soft_compound_assessment": {
            "samples": soft_info.get("samples", 0),
            "model_type": soft_info.get("model_type"),
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
