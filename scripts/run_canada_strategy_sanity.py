"""Run Canada 2026 strategy sanity cases and write race-weekend artifacts.

This is a focused verification/playbook script. It uses the canonical race
configuration layer and strategy engine; it does not create a second optimizer.
Generated outputs live under data/processed/race_packs/canada_2026/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.race_config import (
    RaceStrategyContext,
    build_race_strategy_context,
    confidence_label_for_context,
    race_pack_dir,
)
from src.simulation.strategy_engine import (
    StrategyPlan,
    build_strategy_timing_trace,
    recommend_best_strategy,
)
from src.simulation.strategy_sensitivity import assess_strategy_stability


SCRIPT_VERSION = "canada-strategy-sanity.v1"
SUPPORT_ORDER = {"Low": 0, "Moderate": 1, "High": 2}


@dataclass(frozen=True)
class SanityCase:
    case_id: str
    label: str
    current_lap: int
    current_compound: str
    tyre_age: int
    purpose: str


CASES = (
    SanityCase(
        case_id="early_medium_stint",
        label="Early MEDIUM stint",
        current_lap=10,
        current_compound="MEDIUM",
        tyre_age=8,
        purpose="Opening-stint check before the first expected stop window.",
    ),
    SanityCase(
        case_id="mid_medium_stint",
        label="Mid MEDIUM stint",
        current_lap=24,
        current_compound="MEDIUM",
        tyre_age=12,
        purpose="Mid-opening-stint check near the race-pack baseline.",
    ),
    SanityCase(
        case_id="late_medium_stint",
        label="Late MEDIUM stint",
        current_lap=38,
        current_compound="MEDIUM",
        tyre_age=26,
        purpose="Late MEDIUM-stint stress case approaching model stint limits.",
    ),
    SanityCase(
        case_id="early_hard_stint",
        label="Early HARD stint",
        current_lap=12,
        current_compound="HARD",
        tyre_age=7,
        purpose="Early hard-runner check for long first-stint behavior.",
    ),
    SanityCase(
        case_id="mid_hard_stint",
        label="Mid HARD stint",
        current_lap=36,
        current_compound="HARD",
        tyre_age=22,
        purpose="Midrace hard-stint check for one-stop timing.",
    ),
    SanityCase(
        case_id="short_soft_finish",
        label="Short SOFT finish scenario",
        current_lap=60,
        current_compound="SOFT",
        tyre_age=4,
        purpose="Short final stint where SOFT may be plausible but support remains Low.",
    ),
    SanityCase(
        case_id="high_tyre_age",
        label="High tyre-age scenario",
        current_lap=45,
        current_compound="MEDIUM",
        tyre_age=34,
        purpose="High tyre-age stress case for immediate-stop behavior.",
    ),
    SanityCase(
        case_id="low_confidence_soft",
        label="Low-confidence SOFT scenario",
        current_lap=42,
        current_compound="SOFT",
        tyre_age=6,
        purpose="Force SOFT support caveats into the sanity ledger.",
    ),
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _plan_to_dict(plan: Optional[StrategyPlan]) -> Optional[dict[str, Any]]:
    if plan is None:
        return None
    return {
        "strategy_type": plan.strategy_type,
        "current_compound": plan.current_compound,
        "current_tyre_life": plan.current_tyre_life,
        "next_compound": plan.next_compound,
        "final_compound": plan.final_compound,
        "pit_lap": plan.pit_lap,
        "second_pit_lap": plan.second_pit_lap,
        "total_race_time": round(float(plan.total_race_time), 3),
        "feasible": bool(plan.feasible),
        "feasibility_reason": plan.feasibility_reason,
        "explanation": plan.explanation,
    }


def _strategy_label(plan: Optional[StrategyPlan]) -> str:
    if plan is None:
        return "n/a"
    label = f"{plan.strategy_type}: {plan.current_compound}->{plan.next_compound}"
    if plan.final_compound:
        label += f"->{plan.final_compound}"
    label += f" @ +{plan.pit_lap}"
    if plan.second_pit_lap is not None:
        label += f", +{plan.second_pit_lap}"
    return label


def _pit_window(trace: dict[str, Any]) -> dict[str, Any]:
    band = trace.get("near_optimal_band_laps") or [trace.get("best_first_pit_lap")]
    band = sorted(int(value) for value in band if value is not None)
    return {
        "start_in_laps": min(band) if band else None,
        "end_in_laps": max(band) if band else None,
        "width_laps": len(band),
        "shape": trace.get("curve_shape"),
        "tolerance_s": trace.get("near_optimal_tolerance_s"),
    }


def _recommended_action(window: dict[str, Any], plan: StrategyPlan) -> str:
    start = window.get("start_in_laps")
    end = window.get("end_in_laps")
    if start is None or end is None:
        return f"Review manually; model selected {_strategy_label(plan)}"
    if start <= 1:
        return f"Pit now or this lap window; target {plan.next_compound}"
    if start <= 3:
        return f"Prepare to pit in +{start} to +{end} laps for {plan.next_compound}"
    return f"Stay out; first model window +{start} to +{end} laps for {plan.next_compound}"


def _support_tier_for_plan(context: RaceStrategyContext, plan: StrategyPlan) -> str:
    compounds = [plan.current_compound, plan.next_compound]
    if plan.final_compound:
        compounds.append(plan.final_compound)
    tiers = [
        context.compound_support.get(compound, {}).get("support_tier", "Low")
        for compound in compounds
    ]
    return min(tiers, key=lambda tier: SUPPORT_ORDER.get(tier, 0))


def _support_notes(context: RaceStrategyContext, plan: StrategyPlan) -> list[str]:
    compounds = [plan.current_compound, plan.next_compound]
    if plan.final_compound:
        compounds.append(plan.final_compound)
    notes = []
    for compound in dict.fromkeys(compounds):
        support = context.compound_support.get(compound, {})
        tier = support.get("support_tier", "Low")
        if tier != "High":
            notes.append(f"{compound} support is {tier}: {support.get('support_reason', 'no reason supplied')}")
    return notes


def _sensitivity_notes(stability: Any, context: RaceStrategyContext) -> list[str]:
    notes = [
        f"Stability: {stability.stability_label}",
        f"Pit-loss sensitivity: {'sensitive' if stability.pit_loss_sensitive else 'stable'}",
        f"Degradation sensitivity: {'sensitive' if stability.degradation_sensitive else 'stable'}",
        f"Pit-loss uncertainty: {context.pit_loss.uncertainty}",
    ]
    notes.extend(stability.flip_conditions[:3])
    return notes


def _flag_case(
    case: SanityCase,
    context: RaceStrategyContext,
    plan: StrategyPlan,
    ranked_plans: list[StrategyPlan],
    window: dict[str, Any],
    confidence: str,
    support_tier: str,
    caveats: list[str],
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    laps_remaining = context.race_config.total_laps - case.current_lap
    start = window.get("start_in_laps")
    end = window.get("end_in_laps")

    if not plan.feasible:
        blockers.append("best-plan-not-feasible")
    if plan.pit_lap < 1 or plan.pit_lap >= laps_remaining:
        blockers.append("best-plan-pit-lap-outside-race-bounds")
    if plan.second_pit_lap is not None and (
        plan.second_pit_lap <= plan.pit_lap or plan.second_pit_lap >= laps_remaining
    ):
        blockers.append("second-stop-outside-race-bounds")
    if start is None or end is None:
        blockers.append("missing-pit-window")
    elif start < 1 or end > laps_remaining or start > end:
        blockers.append("pit-window-outside-race-bounds")
    if plan.next_compound == "SOFT" or plan.final_compound == "SOFT":
        soft_support = context.compound_support.get("SOFT", {}).get("support_tier", "Low")
        if soft_support == "Low" and not confidence.startswith("Cautious"):
            blockers.append("soft-low-support-without-cautious-confidence")
    if support_tier == "Low" and not any("SOFT support is Low" in note for note in caveats):
        blockers.append("low-support-case-missing-support-caveat")
    if context.pit_loss.uncertainty != "Low" and not any("Pit-loss uncertainty" in note for note in caveats):
        blockers.append("pit-loss-uncertainty-not-surfaced")
    if not any("manual snapshot mode does not ingest live timing" in note.lower() for note in caveats):
        blockers.append("canada-live-caveat-not-preserved")

    if window.get("width_laps") == 1 and confidence in {"Stable", "High"}:
        warnings.append("single-lap-window-should-not-be-overread")
    if plan.strategy_type == "two-stop":
        one_stop = next((candidate for candidate in ranked_plans if candidate.strategy_type == "one-stop"), None)
        if one_stop is not None:
            margin = float(one_stop.total_race_time - plan.total_race_time)
            if margin <= 5.0 or context.pit_loss.uncertainty != "Low":
                warnings.append("two-stop-recommendation-needs-manual-review")
    if window.get("shape") == "flat":
        warnings.append("flat-window-use-range-not-exact-lap")
    if support_tier == "Low":
        warnings.append("low-compound-support")

    return blockers, warnings


def _evaluate_case(context: RaceStrategyContext, case: SanityCase) -> dict[str, Any]:
    model = context.degradation_model
    pit_loss_value = context.pit_loss.value_s
    laps_remaining = context.race_config.total_laps - case.current_lap
    candidate_compounds = list(context.race_config.compounds)

    best_plan, ranked_plans = recommend_best_strategy(
        degradation_models=model,
        pit_loss_value=pit_loss_value,
        current_compound=case.current_compound,
        current_tyre_life=case.tyre_age,
        laps_remaining=laps_remaining,
        candidate_compounds=candidate_compounds,
        include_two_stop=True,
    )
    trace = build_strategy_timing_trace(
        degradation_models=model,
        pit_loss_value=pit_loss_value,
        current_compound=case.current_compound,
        current_tyre_life=case.tyre_age,
        laps_remaining=laps_remaining,
        next_compound=best_plan.next_compound,
        final_compound=best_plan.final_compound,
    )
    stability = assess_strategy_stability(
        baseline_plan=best_plan,
        pit_loss_value=pit_loss_value,
        degradation_models=model,
        current_compound=case.current_compound,
        current_tyre_life=case.tyre_age,
        laps_remaining=laps_remaining,
    )
    compounds_for_confidence = [case.current_compound, best_plan.next_compound]
    if best_plan.final_compound:
        compounds_for_confidence.append(best_plan.final_compound)
    confidence = confidence_label_for_context(
        stability.stability_label,
        context,
        compounds_for_confidence,
    )
    window = _pit_window(trace)
    support_tier = _support_tier_for_plan(context, best_plan)
    top_alternative = ranked_plans[1] if len(ranked_plans) > 1 else None
    caveats = []
    caveats.extend(_support_notes(context, best_plan))
    caveats.extend(
        [
            f"Pit-loss uncertainty is {context.pit_loss.uncertainty}: {context.pit_loss.method}.",
            *context.caveats,
        ]
    )
    caveats = list(dict.fromkeys(str(item) for item in caveats if item))
    blockers, warnings = _flag_case(
        case=case,
        context=context,
        plan=best_plan,
        ranked_plans=ranked_plans,
        window=window,
        confidence=confidence,
        support_tier=support_tier,
        caveats=caveats,
    )
    sensitivity_notes = _sensitivity_notes(stability, context)

    return {
        "case_id": case.case_id,
        "label": case.label,
        "purpose": case.purpose,
        "current_lap": case.current_lap,
        "laps_remaining": laps_remaining,
        "current_compound": case.current_compound,
        "tyre_age": case.tyre_age,
        "recommended_action": _recommended_action(window, best_plan),
        "strategy_type": best_plan.strategy_type,
        "pit_window": window,
        "pit_window_label": (
            f"+{window['start_in_laps']} to +{window['end_in_laps']}"
            if window.get("start_in_laps") is not None
            else "n/a"
        ),
        "next_compound": best_plan.next_compound,
        "final_compound": best_plan.final_compound,
        "confidence": confidence,
        "support_tier": support_tier,
        "key_caveats": caveats[:6],
        "top_alternative_strategy": _strategy_label(top_alternative),
        "top_alternative": _plan_to_dict(top_alternative),
        "sensitivity_notes": sensitivity_notes,
        "warnings": warnings,
        "blockers": blockers,
        "best_plan": _plan_to_dict(best_plan),
        "timing_trace_summary": {
            "curve_shape": trace.get("curve_shape"),
            "near_optimal_band_laps": trace.get("near_optimal_band_laps"),
            "best_on_window_edge": trace.get("best_on_window_edge"),
            "local_minima_count": trace.get("local_minima_count"),
        },
        "stability": stability.to_dict(),
    }


def _csv_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result["case_id"],
        "label": result["label"],
        "current_lap": result["current_lap"],
        "laps_remaining": result["laps_remaining"],
        "current_compound": result["current_compound"],
        "tyre_age": result["tyre_age"],
        "recommended_action": result["recommended_action"],
        "strategy_type": result["strategy_type"],
        "pit_window": result["pit_window_label"],
        "next_compound": result["next_compound"],
        "final_compound": result["final_compound"],
        "confidence": result["confidence"],
        "support_tier": result["support_tier"],
        "key_caveats": " | ".join(result["key_caveats"]),
        "top_alternative_strategy": result["top_alternative_strategy"],
        "sensitivity_notes": " | ".join(result["sensitivity_notes"]),
        "warnings": " | ".join(result["warnings"]),
        "blockers": " | ".join(result["blockers"]),
    }


def _write_playbook(
    output_path: Path,
    context: RaceStrategyContext,
    results: list[dict[str, Any]],
    status: str,
) -> Path:
    pit = context.pit_loss
    support = context.compound_support
    lines = [
        "# Canada 2026 Race-Weekend Playbook",
        "",
        f"Generated at: {datetime.utcnow().isoformat()}",
        f"Sanity status: {status}",
        "",
        "## Pre-App Refresh",
        "",
        "Run these from the project root before opening the dashboard:",
        "",
        "```bash",
        "python scripts/build_race_pack.py --race canada_2026",
        "python scripts/run_canada_strategy_sanity.py",
        "```",
        "",
        "If Streamlit appears stale after rebuilding, use the app menu to clear cache and rerun, or restart the Streamlit process.",
        "",
        "## Launch",
        "",
        "1. From the project root, run `streamlit run app/streamlit_app.py`.",
        "2. In the race selector, choose `Canada 2026`.",
        "3. Use `Manual live snapshot` mode for race-weekend checks.",
        "",
        "## Manual Fields To Update",
        "",
        "- Current lap.",
        "- Selected driver.",
        "- Current compound.",
        "- Tyre age.",
        "- Position, gap ahead, and gap behind when available.",
        "- Race flags / track status as context. The model does not optimize SC/VSC strategy.",
        "",
        "## Data Support",
        "",
        f"- Pit loss used by strategy checks: {pit.value_s:.3f}s.",
        f"- Pit-loss method: {pit.method}.",
        f"- Pit-loss support/uncertainty: {pit.support_tier} / {pit.uncertainty}.",
        f"- Raw legacy effective pit-loss median: {pit.raw_value_s:.2f}s, IQR {pit.raw_iqr_s:.2f}s.",
        f"- SOFT support: {support['SOFT']['support_tier']} ({support['SOFT']['race_model_laps']} model laps).",
        f"- MEDIUM support: {support['MEDIUM']['support_tier']} ({support['MEDIUM']['race_model_laps']} model laps).",
        f"- HARD support: {support['HARD']['support_tier']} ({support['HARD']['race_model_laps']} model laps).",
        "",
        "## Race Checkpoints",
        "",
        "- Lap 10: Record compound, tyre age, position, gap ahead/behind, flags, recommendation, pit window, confidence, and caveats.",
        "- Lap 20: Re-run the manual snapshot; compare whether the first-stop window has narrowed or moved.",
        "- Lap 30: Record whether the model is still on a one-stop path and whether degradation sensitivity changed.",
        "- Lap 40: Check hard-runner and medium-runner cases; note any flat pit-window ranges instead of treating one lap as exact.",
        "- Lap 50: Confirm finish-compound support. Treat any SOFT use as Low-support unless new data says otherwise.",
        "- Lap 60: Use the short-finish case; save the ledger entry if the call could matter post-race.",
        "",
        "## When To Save A Ledger Entry",
        "",
        "- Save when the recommendation changes from the previous checkpoint.",
        "- Save before any real pit-call discussion.",
        "- Save when flags, weather, traffic, or gaps make the recommendation questionable.",
        "- Save at laps 10, 20, 30, 40, 50, and 60 for post-race review even if the recommendation is unchanged.",
        "- Ledger entries are local JSONL records under `data/processed/recommendation_ledger/canada_2026.jsonl`.",
        "",
        "## Interpreting The Output",
        "",
        "- Use the pit window as a range, not a single exact lap.",
        "- `Stable` means the selected action survives the scripted sensitivity checks; it is not a live-race guarantee.",
        "- `Moderately Sensitive` or `Fragile` means the call depends on degradation or pit-loss assumptions.",
        "- `Cautious` confidence means at least one support tier is weak or race-local certainty is limited.",
        "- SOFT is Low support for Canada in this pack. Do not describe a SOFT recommendation as high confidence.",
        "",
        "## Do Not Overclaim",
        "",
        "- No live timing ingestion.",
        "- No weather prediction.",
        "- No SC/VSC strategy optimization.",
        "- No traffic or competitor-aware optimization.",
        "- Pit loss is a filtered direct pit-in/out transit proxy, not full sector-adjusted strategic loss.",
        "",
        "## Sanity Case Summary",
        "",
    ]

    for result in results:
        warning_label = ", ".join(result["warnings"]) if result["warnings"] else "none"
        blocker_label = ", ".join(result["blockers"]) if result["blockers"] else "none"
        lines.append(
            f"- {result['label']}: {result['recommended_action']} | "
            f"confidence {result['confidence']} | support {result['support_tier']} | "
            f"warnings {warning_label} | blockers {blocker_label}"
        )

    lines.extend(
        [
            "",
            "## Post-Race Review Checklist",
            "",
            "- Compare saved ledger calls with actual pit laps and compounds.",
            "- Mark whether race flags, rain, traffic, or safety-car periods invalidated the deterministic call.",
            "- Compare actual pit-lane losses with the 23.705s strategy proxy.",
            "- Review whether SOFT gained enough race-local evidence to remain Low or be revisited later.",
            "- Update the race pack only from data, not from one-off impressions.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_sanity() -> dict[str, Path]:
    context = build_race_strategy_context(project_root=ROOT, race_key="canada_2026")
    output_dir = race_pack_dir(ROOT, "canada_2026")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [_evaluate_case(context, case) for case in CASES]
    blocker_count = sum(len(result["blockers"]) for result in results)
    warning_count = sum(len(result["warnings"]) for result in results)
    status = "pass" if blocker_count == 0 else "blocked"
    if status == "pass" and warning_count:
        status = "pass_with_warnings"

    csv_path = output_dir / "strategy_sanity_cases.csv"
    pd.DataFrame([_csv_payload(result) for result in results]).to_csv(csv_path, index=False)

    summary_path = _write_json(
        output_dir / "strategy_sanity_summary.json",
        {
            "generated_at": datetime.utcnow().isoformat(),
            "script_version": SCRIPT_VERSION,
            "race_key": context.race_config.key,
            "race_label": context.race_config.label,
            "status": status,
            "case_count": len(results),
            "warning_count": warning_count,
            "blocker_count": blocker_count,
            "pit_loss": context.pit_loss.to_dict(),
            "compound_support": context.compound_support,
            "cases": results,
        },
    )
    playbook_path = _write_playbook(
        output_dir / "race_weekend_playbook.md",
        context=context,
        results=results,
        status=status,
    )
    return {
        "strategy_sanity_cases": csv_path,
        "strategy_sanity_summary": summary_path,
        "race_weekend_playbook": playbook_path,
    }


def main() -> None:
    paths = run_sanity()
    summary = json.loads(paths["strategy_sanity_summary"].read_text(encoding="utf-8"))
    print("\nCanada strategy sanity complete:")
    print(f"  status: {summary['status']}")
    print(f"  cases: {summary['case_count']}")
    print(f"  warnings: {summary['warning_count']}")
    print(f"  blockers: {summary['blocker_count']}")
    for label, path in paths.items():
        print(f"  {label}: {path}")

    if summary["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
