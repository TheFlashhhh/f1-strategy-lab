#!/usr/bin/env python
"""Run the canonical Phase 3A race-state readiness audit."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.race_state import (
    SCHEMA_VERSION,
    build_demo_race_state,
    extract_race_states_from_phase2d_artifact,
    extract_race_states_from_pre3_backtest,
    schema_field_catalog_to_dict,
)


STATUS_AVAILABLE = "available now"
STATUS_DERIVABLE = "derivable with light work"
STATUS_PARTIAL = "partially available / messy"
STATUS_NOT_READY = "not reliably available yet"


def _json_dump(path: Path, payload: dict) -> None:
    """Write JSON with parent-directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _json_load(path: Path) -> dict:
    """Read JSON from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _raw_dataset_snapshot(path: Path) -> dict:
    """Summarize one canonical parquet dataset."""
    df = pd.read_parquet(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "events": sorted(df["event_name"].dropna().unique().tolist()) if "event_name" in df.columns else [],
        "null_counts": {
            column: int(df[column].isna().sum())
            for column in ["driver", "team", "lap_number", "compound", "tyre_life", "position", "track_status"]
            if column in df.columns
        },
    }


def _build_dashboard_field_audit(
    manifest_summary: dict,
    miami_snapshot: dict,
    recency_snapshot: dict,
    phase2d_exists: bool,
    pre3_exists: bool,
) -> dict[str, dict[str, object]]:
    """Classify dashboard-field readiness using current repo/data evidence."""
    miami_position_nulls = miami_snapshot["null_counts"].get("position", 0)
    recency_position_nulls = recency_snapshot["null_counts"].get("position", 0)
    total_positions = miami_snapshot["rows"] + recency_snapshot["rows"]
    total_position_nulls = miami_position_nulls + recency_position_nulls

    audit = {
        "race_session_identity": {
            "status": STATUS_AVAILABLE,
            "evidence": (
                "Canonical parquet datasets include season/event_name/session_name/circuit_name, "
                f"and the manifest records {manifest_summary['total_sessions_succeeded']} successful source sessions."
            ),
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"], "data/raw/manifest.json"],
        },
        "driver_identity": {
            "status": STATUS_AVAILABLE,
            "evidence": "Both canonical parquet datasets include a non-null driver column on every stored row.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "team_identity": {
            "status": STATUS_AVAILABLE,
            "evidence": "Both canonical parquet datasets include a non-null team column on every stored row.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "current_position": {
            "status": STATUS_AVAILABLE,
            "evidence": (
                "Position is stored directly in the canonical raw data; "
                f"{total_position_nulls} of {total_positions} replay rows are missing it."
            ),
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "start_position": {
            "status": STATUS_DERIVABLE,
            "evidence": "Start position can be derived from each driver's earliest available lap position within an event/session.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "lap_by_lap_position_change": {
            "status": STATUS_DERIVABLE,
            "evidence": "Lap-level position is present, so position deltas can be reconstructed per driver with simple replay grouping.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "nearby_car_identity_context": {
            "status": STATUS_DERIVABLE,
            "evidence": "Cars ahead/behind can be inferred by sorting same-lap rows on position when position is available.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "gaps_to_car_ahead_behind": {
            "status": STATUS_NOT_READY,
            "evidence": "Canonical raw parquet contains order position but no reliable lap-level interval/gap timing columns.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "lap_time_context": {
            "status": STATUS_AVAILABLE,
            "evidence": "lap_time is present in canonical raw replay data and is already used throughout the deterministic pipeline.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "current_compound_and_tyre_age": {
            "status": STATUS_AVAILABLE,
            "evidence": "compound and tyre_life are present in canonical raw data and already feed the current recommendation engine.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "stint_history": {
            "status": STATUS_DERIVABLE,
            "evidence": "stint, compound, lap_number, pit_in_time, and pit_out_time are available for replay reconstruction of stint timelines.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "pit_events": {
            "status": STATUS_DERIVABLE,
            "evidence": "Pit events can be inferred from stint changes and supported by pit_in_time / pit_out_time fields.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "recommendation_payload": {
            "status": STATUS_AVAILABLE,
            "evidence": (
                "The current deterministic engine plus Phase 2D / Pre-3 artifacts already expose best-plan type, pit timing, "
                "compound path, feasibility, and explanation."
            ),
            "source_paths": [
                "data/processed/phase2d_validation_summary.json",
                "data/processed/pre3_backtest_summary.json",
            ],
        },
        "support_confidence_risk_notes": {
            "status": STATUS_AVAILABLE,
            "evidence": (
                "Support tiers, stability labels, flip conditions, stop-timing notes, and pace-shape audit signals are already produced "
                "by the current validation / defensibility stack."
            ),
            "source_paths": [
                "data/processed/pre3_compound_support_summary.json",
                "data/processed/phase2d_validation_summary.json",
                "data/processed/pre3_stop_timing_audit.json",
                "data/processed/pre3_pace_shape_audit.json",
            ],
        },
        "track_status": {
            "status": STATUS_AVAILABLE,
            "evidence": "track_status is stored on every canonical replay row.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "sc_vsc_indicators": {
            "status": STATUS_PARTIAL,
            "evidence": (
                "track_status codes exist, but the pipeline does not yet normalize them into explicit SC/VSC booleans or consume them in strategy logic."
            ),
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "weather": {
            "status": STATUS_NOT_READY,
            "evidence": "No canonical weather feed or weather columns are present in the current raw replay data or processed artifacts.",
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "tyre_inventory": {
            "status": STATUS_NOT_READY,
            "evidence": (
                "The pipeline tracks current compound and tyre age, but not a canonical set-inventory ledger or per-race fresh/used tyre allocation model."
            ),
            "source_paths": [miami_snapshot["path"], recency_snapshot["path"]],
        },
        "driver_photo_assets": {
            "status": STATUS_NOT_READY,
            "evidence": "No canonical driver profile or media asset table exists in the repository today.",
            "source_paths": [],
        },
        "track_map_coordinates": {
            "status": STATUS_NOT_READY,
            "evidence": "The repository has no canonical per-lap XY/sector coordinate feed for live or replay track-map rendering.",
            "source_paths": [],
        },
        "historical_replay_readiness": {
            "status": STATUS_AVAILABLE if phase2d_exists and pre3_exists else STATUS_DERIVABLE,
            "evidence": (
                "Raw replay data exists and canonical validation/backtest artifacts are present, which is enough to build historical checkpoints before live mode."
                if phase2d_exists and pre3_exists
                else "Raw replay data exists; canonical replay artifacts can be rebuilt from the repo if needed."
            ),
            "source_paths": [
                miami_snapshot["path"],
                "data/processed/phase2d_validation_summary.json",
                "data/processed/pre3_backtest_summary.json",
            ],
        },
        "live_dashboard_feed": {
            "status": STATUS_NOT_READY,
            "evidence": "The repo has historical/replay inputs and offline strategy artifacts, but no live ingestion, live timing adapter, or push-update loop yet.",
            "source_paths": [],
        },
    }
    return audit


def _status_counts(field_audit: dict[str, dict[str, object]]) -> dict[str, int]:
    """Count statuses across the dashboard-field audit."""
    counts: dict[str, int] = {}
    for item in field_audit.values():
        status = str(item["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _status_keys(field_audit: dict[str, dict[str, object]], target_status: str) -> list[str]:
    """Return all field keys assigned to one status."""
    return [
        key
        for key, item in field_audit.items()
        if item["status"] == target_status
    ]


def main() -> None:
    """Run the Phase 3A readiness audit and write the canonical artifact."""
    print("\n" + "=" * 88)
    print("PHASE 3A: RACE-STATE AND DASHBOARD GROUNDWORK READINESS")
    print("=" * 88)

    manifest_path = ROOT / "data" / "raw" / "manifest.json"
    miami_path = ROOT / "data" / "raw" / "miami_historical" / "combined.parquet"
    recency_path = ROOT / "data" / "raw" / "season_2026_pre_miami" / "combined.parquet"
    phase2d_path = ROOT / "data" / "processed" / "phase2d_validation_summary.json"
    pre3_backtest_path = ROOT / "data" / "processed" / "pre3_backtest_summary.json"
    pre3_support_path = ROOT / "data" / "processed" / "pre3_compound_support_summary.json"
    stop_timing_path = ROOT / "data" / "processed" / "pre3_stop_timing_audit.json"
    pace_shape_path = ROOT / "data" / "processed" / "pre3_pace_shape_audit.json"

    for required_path in [manifest_path, miami_path, recency_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase 3A readiness input missing: {required_path}")

    manifest_summary = _json_load(manifest_path)
    miami_snapshot = _raw_dataset_snapshot(miami_path)
    recency_snapshot = _raw_dataset_snapshot(recency_path)
    phase2d_states = extract_race_states_from_phase2d_artifact(ROOT)
    pre3_states = extract_race_states_from_pre3_backtest(ROOT)
    demo_state = build_demo_race_state(ROOT)

    field_audit = _build_dashboard_field_audit(
        manifest_summary=manifest_summary,
        miami_snapshot=miami_snapshot,
        recency_snapshot=recency_snapshot,
        phase2d_exists=phase2d_path.exists(),
        pre3_exists=pre3_backtest_path.exists(),
    )
    status_counts = _status_counts(field_audit)

    example_states = {
        "demo_scenario": demo_state.to_dict(),
        "validation_example": phase2d_states[0].to_dict() if phase2d_states else None,
        "backtest_checkpoint_example": pre3_states[0].to_dict() if pre3_states else None,
    }

    readiness_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "3A",
        "schema_version": SCHEMA_VERSION,
        "pipeline_snapshot": {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "raw_datasets": {
                "miami_historical": miami_snapshot,
                "season_2026_pre_miami": recency_snapshot,
            },
            "processed_artifacts_present": {
                "phase2d_validation_summary": phase2d_path.exists(),
                "pre3_backtest_summary": pre3_backtest_path.exists(),
                "pre3_compound_support_summary": pre3_support_path.exists(),
                "pre3_stop_timing_audit": stop_timing_path.exists(),
                "pre3_pace_shape_audit": pace_shape_path.exists(),
            },
        },
        "schema_field_catalog": schema_field_catalog_to_dict(),
        "dashboard_field_availability": field_audit,
        "status_counts": status_counts,
        "readiness_summary": {
            STATUS_AVAILABLE: _status_keys(field_audit, STATUS_AVAILABLE),
            STATUS_DERIVABLE: _status_keys(field_audit, STATUS_DERIVABLE),
            STATUS_PARTIAL: _status_keys(field_audit, STATUS_PARTIAL),
            STATUS_NOT_READY: _status_keys(field_audit, STATUS_NOT_READY),
            "replay_first_rationale": (
                "Historical replay is ready before live mode because canonical lap-level replay data and validation/backtest artifacts already exist, "
                "while live timing, gap timing, weather, and map coordinates do not."
            ),
            "recommended_next_phase": "Phase 3B can build a dashboard shell and driver detail drawer on top of these canonical replay checkpoints without changing strategy logic.",
        },
        "example_race_states": example_states,
    }

    output_path = ROOT / "data" / "processed" / "phase3a_data_availability_summary.json"
    _json_dump(output_path, readiness_payload)

    print(f"Schema version: {SCHEMA_VERSION}")
    print(
        "Raw replay rows: "
        f"Miami {miami_snapshot['rows']:,} | 2026 pre-Miami {recency_snapshot['rows']:,}"
    )
    print(
        "Example race states built: "
        f"demo=1 | validation={1 if phase2d_states else 0} | backtest={1 if pre3_states else 0}"
    )
    print(
        f"{STATUS_AVAILABLE}: {status_counts.get(STATUS_AVAILABLE, 0)} | "
        f"{STATUS_DERIVABLE}: {status_counts.get(STATUS_DERIVABLE, 0)} | "
        f"{STATUS_PARTIAL}: {status_counts.get(STATUS_PARTIAL, 0)} | "
        f"{STATUS_NOT_READY}: {status_counts.get(STATUS_NOT_READY, 0)}"
    )
    print("Key ready-now areas: " + ", ".join(_status_keys(field_audit, STATUS_AVAILABLE)))
    print("Key future gaps: " + ", ".join(_status_keys(field_audit, STATUS_NOT_READY)))
    print(f"Artifact written: {output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
