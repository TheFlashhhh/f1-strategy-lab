"""Build processed race-pack artifacts for a configured race.

Run examples:
    python scripts/build_race_pack.py --race canada_2026
    python scripts/build_race_pack.py --race canada_2026 --fetch-historical

Generated files stay under data/processed/race_packs/<race_key>/, which is
ignored by Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.race_config import (
    RaceConfig,
    build_race_strategy_context,
    confidence_label_for_context,
    get_race_config,
    list_race_configs,
    race_pack_dir,
)
from src.simulation.strategy_engine import build_strategy_timing_trace, recommend_best_strategy
from src.simulation.strategy_sensitivity import assess_strategy_stability


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _plan_to_dict(plan: Any) -> dict[str, Any]:
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


def _pit_window_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    band = trace.get("near_optimal_band_laps") or [trace.get("best_first_pit_lap")]
    band = [int(value) for value in band if value is not None]
    return {
        "start_in_laps": min(band) if band else None,
        "end_in_laps": max(band) if band else None,
        "shape": trace.get("curve_shape"),
        "tolerance_s": trace.get("near_optimal_tolerance_s"),
    }


def _build_baseline_strategies(context: Any) -> dict[str, Any]:
    race_config = context.race_config
    model = context.degradation_model
    pit_loss_value = context.pit_loss.value_s
    available_compounds = [
        compound
        for compound in race_config.compounds
        if model.get_model_info(compound).get("model_type")
    ]

    scenario_results = []
    for scenario in race_config.baseline_scenarios:
        try:
            best_plan, ranked_plans = recommend_best_strategy(
                degradation_models=model,
                pit_loss_value=pit_loss_value,
                current_compound=scenario.current_compound,
                current_tyre_life=scenario.current_tyre_life,
                laps_remaining=scenario.laps_remaining,
                candidate_compounds=available_compounds,
                include_two_stop=True,
            )
            timing_trace = build_strategy_timing_trace(
                degradation_models=model,
                pit_loss_value=pit_loss_value,
                current_compound=scenario.current_compound,
                current_tyre_life=scenario.current_tyre_life,
                laps_remaining=scenario.laps_remaining,
                next_compound=best_plan.next_compound,
                final_compound=best_plan.final_compound,
            )
            stability = assess_strategy_stability(
                baseline_plan=best_plan,
                pit_loss_value=pit_loss_value,
                degradation_models=model,
                current_compound=scenario.current_compound,
                current_tyre_life=scenario.current_tyre_life,
                laps_remaining=scenario.laps_remaining,
            )
            future_compounds = [
                compound
                for compound in [best_plan.next_compound, best_plan.final_compound]
                if compound
            ]
            scenario_results.append(
                {
                    "scenario": scenario.to_dict(),
                    "status": "ok",
                    "best_plan": _plan_to_dict(best_plan),
                    "pit_window": _pit_window_from_trace(timing_trace),
                    "confidence": confidence_label_for_context(
                        stability.stability_label,
                        context,
                        future_compounds,
                    ),
                    "stability": stability.to_dict(),
                    "alternatives": [_plan_to_dict(plan) for plan in ranked_plans[:5]],
                    "timing_trace_summary": {
                        "curve_shape": timing_trace.get("curve_shape"),
                        "near_optimal_band_laps": timing_trace.get("near_optimal_band_laps"),
                        "best_on_window_edge": timing_trace.get("best_on_window_edge"),
                        "local_minima_count": timing_trace.get("local_minima_count"),
                    },
                }
            )
        except Exception as exc:
            scenario_results.append(
                {
                    "scenario": scenario.to_dict(),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "race_key": race_config.key,
        "race_label": race_config.label,
        "pit_loss_value_s": round(float(pit_loss_value), 3),
        "pit_loss_method": context.pit_loss.method,
        "pit_loss_uncertainty": context.pit_loss.uncertainty,
        "pit_loss_raw_value_s": (
            round(float(context.pit_loss.raw_value_s), 3)
            if context.pit_loss.raw_value_s is not None
            else None
        ),
        "pit_loss_strategy_sample_count": context.pit_loss.strategy_sample_count,
        "available_compounds": available_compounds,
        "scenarios": scenario_results,
    }


def _readiness_status(context: Any, baseline_report: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    data_summary = context.data_summary

    if not data_summary.get("historical_dataset_available"):
        warnings.append(
            f"Race-local dataset {data_summary['historical_dataset']} is not available."
        )
    if context.pit_loss.support_tier != "High":
        warnings.append(f"Pit-loss support tier is {context.pit_loss.support_tier}.")
    if context.pit_loss.uncertainty in {"Moderate", "High", "Unknown"}:
        warnings.append(f"Pit-loss uncertainty is {context.pit_loss.uncertainty}.")
    low_compounds = [
        compound
        for compound, support in context.compound_support.items()
        if support.get("support_tier") == "Low"
    ]
    if low_compounds:
        warnings.append("Low race-specific compound support: " + ", ".join(low_compounds))

    failed = [
        item["scenario"]["scenario_id"]
        for item in baseline_report.get("scenarios", [])
        if item.get("status") != "ok"
    ]
    if failed:
        blockers.append("Baseline strategy scenarios failed: " + ", ".join(failed))

    if blockers:
        return "blocked", blockers + warnings
    if warnings:
        return "cautious", warnings
    return "ready"


def _write_readiness_report(
    output_path: Path,
    context: Any,
    baseline_report: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> Path:
    status, status_notes = _readiness_status(context, baseline_report)
    lines = [
        f"# {context.race_config.label} Race-Pack Readiness",
        "",
        f"Generated at: {datetime.utcnow().isoformat()}",
        f"Status: {status}",
        "",
        "## Data Support",
        "",
        f"- Historical dataset: {context.data_summary['historical_dataset']}",
        f"- Historical raw laps: {context.data_summary['historical_raw_laps']}",
        f"- Model source: {context.data_summary['model_source']}",
        f"- Pit loss strategy estimate: {context.pit_loss.value_s:.2f}s "
        f"({context.pit_loss.support_tier}, {context.pit_loss.uncertainty} uncertainty, "
        f"{context.pit_loss.method})",
    ]
    if context.pit_loss.raw_value_s is not None:
        raw_iqr = context.pit_loss.raw_iqr_s if context.pit_loss.raw_iqr_s is not None else 0.0
        strategy_iqr = (
            context.pit_loss.strategy_iqr_s
            if context.pit_loss.strategy_iqr_s is not None
            else 0.0
        )
        lines.extend(
            [
                f"- Raw effective pit-loss median: {context.pit_loss.raw_value_s:.2f}s "
                f"(n={context.pit_loss.raw_sample_count}, IQR={raw_iqr:.2f}s)",
                f"- Filtered strategy sample: n={context.pit_loss.strategy_sample_count}, "
                f"IQR={strategy_iqr:.2f}s",
            ]
        )
    if context.pit_loss.filter_notes:
        lines.append("- Pit-loss filter notes: " + " ".join(context.pit_loss.filter_notes))
    lines.extend(
        [
            "",
            "## Compound Support",
            "",
        ]
    )
    for compound, support in context.compound_support.items():
        lines.append(
            f"- {compound}: {support.get('support_tier')} | "
            f"{support.get('race_model_laps', 0)} race-local model laps | "
            f"{support.get('support_reason')}"
        )

    lines.extend(["", "## Baseline Strategy Checks", ""])
    for item in baseline_report.get("scenarios", []):
        scenario = item["scenario"]
        if item.get("status") != "ok":
            lines.append(f"- {scenario['scenario_id']}: FAILED - {item.get('error')}")
            continue
        plan = item["best_plan"]
        window = item["pit_window"]
        lines.append(
            f"- {scenario['scenario_id']}: {plan['strategy_type']} | "
            f"pit +{window['start_in_laps']}-+{window['end_in_laps']} | "
            f"next {plan['next_compound']} | confidence {item['confidence']}"
        )

    if status_notes:
        lines.extend(["", "## Readiness Notes", ""])
        for note in status_notes:
            lines.append(f"- {note}")

    if context.caveats:
        lines.extend(["", "## Caveats", ""])
        for caveat in context.caveats:
            lines.append(f"- {caveat}")

    lines.extend(["", "## Artifacts", ""])
    for label, path in artifact_paths.items():
        lines.append(f"- {label}: `{path.as_posix()}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _fetch_historical_data(project_root: Path, race_config: RaceConfig, output_dir: Path) -> Path:
    """Optionally ingest race-local historical data through the existing FastF1 path."""
    from src.data.ingest_phase1 import DataIngestPipeline, SessionRecord

    pipeline = DataIngestPipeline(cache_dir=project_root / "data" / "raw" / "fastf1_cache")
    datasets = []
    for year in race_config.historical_years:
        try:
            laps = pipeline.load_session(year, race_config.event_name, "R")
            if laps is None:
                pipeline.manifest.append(
                    SessionRecord(
                        season=year,
                        event_name=race_config.event_name,
                        session_name="Race",
                        data_group=race_config.historical_dataset,
                        regulation_era="2022_2026",
                        target_race_context=race_config.circuit_id,
                        success=False,
                        error_message="No laps found / data unavailable",
                    )
                )
                continue
            std_df, missing_fields = pipeline.standardize_schema(
                laps,
                year=year,
                event_name=race_config.event_name,
                circuit_name=race_config.circuit_name,
                session_name="Race",
                data_group=race_config.historical_dataset,
                regulation_era="2022_2026",
                target_race_context=race_config.circuit_id,
            )
            datasets.append(std_df)
            pipeline.manifest.append(
                SessionRecord(
                    season=year,
                    event_name=race_config.event_name,
                    session_name="Race",
                    data_group=race_config.historical_dataset,
                    regulation_era="2022_2026",
                    target_race_context=race_config.circuit_id,
                    success=True,
                    row_count=len(std_df),
                    missing_fields=missing_fields,
                )
            )
        except Exception as exc:
            pipeline.manifest.append(
                SessionRecord(
                    season=year,
                    event_name=race_config.event_name,
                    session_name="Race",
                    data_group=race_config.historical_dataset,
                    regulation_era="2022_2026",
                    target_race_context=race_config.circuit_id,
                    success=False,
                    error_message=str(exc),
                )
            )

    if datasets:
        pipeline.save_datasets(
            datasets,
            group_name=race_config.historical_dataset,
            output_dir=project_root / "data" / "raw",
        )

    manifest_path = output_dir / "historical_ingest_manifest.json"
    pipeline.save_manifest(manifest_path)
    return manifest_path


def build_race_pack(race: str, fetch_historical: bool = False) -> dict[str, Path]:
    race_config = get_race_config(race)
    output_dir = race_pack_dir(ROOT, race_config.key)
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_manifest = None
    if fetch_historical:
        fetch_manifest = _fetch_historical_data(ROOT, race_config, output_dir)

    context = build_race_strategy_context(project_root=ROOT, race_key=race_config.key)
    baseline_report = _build_baseline_strategies(context)

    artifact_paths = {
        "manifest": _write_json(
            output_dir / "manifest.json",
            {
                "generated_at": datetime.utcnow().isoformat(),
                "race_config": context.race_config.to_dict(),
                "context": context.to_summary(),
                "fetch_historical_manifest": str(fetch_manifest) if fetch_manifest else None,
            },
        ),
        "compound_support": _write_json(
            output_dir / "compound_support_summary.json",
            {
                "generated_at": datetime.utcnow().isoformat(),
                "race_key": race_config.key,
                "compound_support": context.compound_support,
            },
        ),
        "pit_loss": _write_json(
            output_dir / "pit_loss_estimate.json",
            {
                "generated_at": datetime.utcnow().isoformat(),
                "race_key": race_config.key,
                "pit_loss": context.pit_loss.to_dict(),
            },
        ),
        "baseline_strategies": _write_json(
            output_dir / "baseline_strategies.json",
            baseline_report,
        ),
    }
    if context.pit_loss.audit_path:
        audit_path = Path(context.pit_loss.audit_path)
        if audit_path.exists():
            artifact_paths["pit_loss_audit"] = audit_path
    artifact_paths["readiness_report"] = _write_readiness_report(
        output_dir / "readiness_report.md",
        context=context,
        baseline_report=baseline_report,
        artifact_paths=artifact_paths,
    )
    if fetch_manifest:
        artifact_paths["historical_ingest_manifest"] = fetch_manifest
    return artifact_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a race activation pack.")
    parser.add_argument(
        "--race",
        default="canada_2026",
        help="Race key or alias. Examples: canada_2026, canada, miami_2026.",
    )
    parser.add_argument(
        "--fetch-historical",
        action="store_true",
        help="Attempt to ingest configured historical race-local data with FastF1 before building the pack.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured races and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for config in list_race_configs():
            print(f"{config.key}: {config.label} ({config.total_laps} laps)")
        return

    artifact_paths = build_race_pack(
        race=args.race,
        fetch_historical=bool(args.fetch_historical),
    )
    print("\nRace pack built:")
    for label, path in artifact_paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
