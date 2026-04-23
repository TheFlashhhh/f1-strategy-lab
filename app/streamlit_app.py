"""Replay-first Streamlit dashboard shell for F1 Strategy Lab.

Phase 3B reshapes the app around the Phase 3A race-state groundwork:

- main race-control shell with a schematic circuit panel
- timing/order panel built from canonical race-state objects
- selected-driver detail workflow for replay checkpoints
- advanced analysis moved into secondary tabs/expanders

The strategy engine remains the existing deterministic single-car model.

Run as: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import re
import sys
from typing import Any, Optional

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.race_control_board import (
    race_control_board_available,
    render_race_control_board,
)
from src.data.loader import DataLoader
from src.data.preprocess import detect_pit_stops, select_relevant_columns
from src.features.hybrid_modeling import (
    build_role_based_hybrid_model,
    load_or_build_hybrid_dataset,
    summarize_hybrid_context,
)
from src.simulation.race_state import (
    RaceState,
    build_demo_race_state,
    extract_historical_replay_snapshot,
    extract_race_states_from_phase2d_artifact,
)
from src.simulation.strategy import estimate_pit_loss_window, optimize_pit_window
from src.simulation.strategy_engine import build_strategy_timing_trace, recommend_best_strategy
from src.simulation.strategy_sensitivity import assess_strategy_stability


COMPOUND_COLORS = {
    "SOFT": "#ef4444",
    "MEDIUM": "#f59e0b",
    "HARD": "#e5e7eb",
    "UNKNOWN": "#94a3b8",
}

COMPOUND_SHORT = {
    "SOFT": "S",
    "MEDIUM": "M",
    "HARD": "H",
    "UNKNOWN": "?",
}

TEAM_SHORT_NAMES = {
    "Aston Martin": "Aston",
    "Alpine": "Alpine",
    "Ferrari": "Ferrari",
    "Haas F1 Team": "Haas",
    "Kick Sauber": "Sauber",
    "McLaren": "McLaren",
    "Mercedes": "Mercedes",
    "RB": "RB",
    "Red Bull Racing": "Red Bull",
    "Williams": "Williams",
}

FALLBACK_TEAM_COLORS = {
    "Alpine": "#0090FF",
    "Aston Martin": "#229971",
    "Ferrari": "#DC0000",
    "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252",
    "McLaren": "#FF8700",
    "Mercedes": "#27F4D2",
    "RB": "#6692FF",
    "Red Bull Racing": "#1E5BC6",
    "Williams": "#64C4FF",
}

TEAM_BADGE_TONES = {
    "Alpine": "blue",
    "Aston Martin": "green",
    "Ferrari": "red",
    "Haas F1 Team": "gray",
    "Kick Sauber": "green",
    "McLaren": "orange",
    "Mercedes": "green",
    "RB": "blue",
    "Red Bull Racing": "blue",
    "Williams": "blue",
}

COMPOUND_BADGE_TONES = {
    "SOFT": "red",
    "MEDIUM": "orange",
    "HARD": "gray",
    "UNKNOWN": "blue",
}

ASSET_MANIFEST_PATH = ROOT / "assets" / "asset_manifest.json"
HTML_TAG_RE = re.compile(r"<[^>]+>")


@st.cache_data
def load_hybrid_context() -> dict:
    """Load the role-based hybrid context used for reporting."""
    try:
        _, hybrid_context = load_or_build_hybrid_dataset(project_root=ROOT)
        return summarize_hybrid_context(hybrid_context)
    except Exception as exc:
        st.warning(f"Hybrid context loading failed ({exc}). Using a minimal fallback summary.")
        return {
            "metadata": {"weighting_scheme": "fallback", "total_active_pools": 1},
            "data_grouping": [],
            "blending_strategy": {
                "method": "miami_only_fallback",
                "rationale": "Hybrid pools unavailable; app is showing a reduced context summary.",
            },
            "total_laps": 0,
        }


@st.cache_data
def build_integrated_pipeline() -> tuple:
    """Build the canonical role-based hybrid model plus Miami pit-loss baseline."""
    deg_result, _ = build_role_based_hybrid_model(project_root=ROOT)
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
    with open(artifact_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_pre3_backtest_summary() -> dict | None:
    """Load the canonical Pre-3 backtest artifact."""
    artifact_path = ROOT / "data" / "processed" / "pre3_backtest_summary.json"
    if not artifact_path.exists():
        return None
    with open(artifact_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_phase3a_availability_summary() -> dict | None:
    """Load the canonical Phase 3A data-availability artifact when present."""
    artifact_path = ROOT / "data" / "processed" / "phase3a_data_availability_summary.json"
    if not artifact_path.exists():
        return None
    with open(artifact_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_phase2d_race_state_payloads() -> list[dict[str, Any]]:
    """Load representative validation race states as JSON-friendly payloads."""
    return [state.to_dict() for state in extract_race_states_from_phase2d_artifact(ROOT)]


@st.cache_data
def load_asset_manifest() -> dict[str, Any]:
    """Load optional local-first board asset metadata."""
    if not ASSET_MANIFEST_PATH.exists():
        return {"teams": {}, "drivers": {}}
    try:
        with open(ASSET_MANIFEST_PATH, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"teams": {}, "drivers": {}}
    return {
        "teams": manifest.get("teams", {}),
        "drivers": manifest.get("drivers", {}),
    }


@st.cache_data
def load_historical_replay_snapshot_payload(checkpoint_id: str) -> dict[str, Any]:
    """Build one historical replay field snapshot for the dashboard shell."""
    backtest_summary = load_pre3_backtest_summary() or {}
    checkpoint = next(
        (
            item
            for item in backtest_summary.get("checkpoints", [])
            if item.get("checkpoint_id") == checkpoint_id
        ),
        None,
    )
    if checkpoint is None:
        raise ValueError(f"Unknown replay checkpoint: {checkpoint_id}")

    states = extract_historical_replay_snapshot(
        ROOT,
        season=int(checkpoint["season"]),
        event_name="Miami Grand Prix",
        session_name="Race",
        lap_number=int(checkpoint["checkpoint_lap"]),
        include_extended_recommendation_context=False,
    )
    return {
        "states": [state.to_dict() for state in states],
        "meta": {
            "snapshot_key": f"historical:{checkpoint_id}",
            "source_label": "Historical Replay Snapshot",
            "source_mode": "historical_replay",
            "headline": f"Miami 2024 replay checkpoint: lap {checkpoint['checkpoint_lap']}",
            "subheadline": checkpoint["rationale"],
            "mode_note": (
                "This is a replay-first historical shell. Driver order, tyres, and stint state come from lap-level replay data; "
                "the recommendation overlay comes from the current deterministic strategy engine."
            ),
            "field_scope_note": (
                "Exact interval gaps, live coordinates, and tyre inventory are still missing; optional local logos/photos only appear when you provide them through the asset manifest."
            ),
            "default_driver_code": checkpoint["driver"],
            "is_full_field": True,
            "reference_path": "data/processed/pre3_backtest_summary.json",
        },
    }


def _phase2d_scenario_options() -> list[str]:
    """Return representative validation scenario ids."""
    payloads = load_phase2d_race_state_payloads()
    return [payload["state_id"].replace("phase2d:", "") for payload in payloads]


def _pre3_checkpoint_options() -> list[str]:
    """Return replay checkpoint ids."""
    backtest_summary = load_pre3_backtest_summary() or {}
    return [item["checkpoint_id"] for item in backtest_summary.get("checkpoints", [])]


def _driver_label(state: RaceState) -> str:
    """Return a compact driver label."""
    return state.selected_driver.display_name or state.selected_driver.driver_code or "Unknown"


def _driver_key(state: RaceState) -> str:
    """Return a stable driver selection key."""
    return state.selected_driver.driver_code or state.state_id


def _format_position(value: Optional[int]) -> str:
    """Format a race position."""
    return f"P{value}" if value is not None else "P?"


def _compound_badge_text(compound: str) -> str:
    """Return a compact compound label."""
    return compound or "UNKNOWN"


def _compound_short(compound: str | None) -> str:
    """Return one-letter compound shorthand."""
    return COMPOUND_SHORT.get(compound or "UNKNOWN", "?")


def _team_short(team_name: str | None) -> str:
    """Return a compact team label."""
    if not team_name:
        return "Team n/a"
    return TEAM_SHORT_NAMES.get(team_name, team_name)


def _hex_to_rgb(color: str | None) -> tuple[int, int, int]:
    """Convert a hex colour string to RGB, falling back to the board cyan."""
    if not color:
        return (56, 189, 248)
    clean = color.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(part * 2 for part in clean)
    if len(clean) != 6:
        return (56, 189, 248)
    try:
        return tuple(int(clean[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return (56, 189, 248)


def _rgba(color: str | None, alpha: float) -> str:
    """Return an rgba() string for inline styling."""
    red, green, blue = _hex_to_rgb(color)
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"


def _contrast_text_color(color: str | None) -> str:
    """Return a high-contrast text colour for a solid background."""
    red, green, blue = _hex_to_rgb(color)
    luminance = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    return "#07101a" if luminance >= 150 else "#f8fafc"


def _team_badge_tone(team_name: str | None) -> str:
    """Return the closest Streamlit badge tone for a team accent."""
    return TEAM_BADGE_TONES.get(team_name or "", "blue")


def _compound_badge_tone(compound: str | None) -> str:
    """Return the Streamlit badge tone for a compound accent."""
    return COMPOUND_BADGE_TONES.get(compound or "UNKNOWN", "blue")


def _markdown_badge(label: str, tone: str) -> str:
    """Return a small native Markdown badge."""
    safe_label = _plain_text(label, "")
    safe_label = safe_label.replace("[", "\\[").replace("]", "\\]")
    return f":{tone}-badge[{safe_label}]"


@st.cache_data
def _inline_local_asset(reference: str) -> str | None:
    """Inline a local asset as a data URI when it exists inside the workspace."""
    asset_path = Path(reference)
    if not asset_path.is_absolute():
        asset_path = ROOT / asset_path
    try:
        asset_path = asset_path.resolve()
        asset_path.relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return None
    if not asset_path.exists() or not asset_path.is_file():
        return None
    mime_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _asset_source(reference: str | None) -> str | None:
    """Resolve an asset reference to a browser-safe source string."""
    if not reference:
        return None
    text = str(reference).strip()
    if not text:
        return None
    if text.startswith(("https://", "http://", "data:")):
        return text
    return _inline_local_asset(text)


def _team_visuals(team_name: str | None) -> dict[str, str | None]:
    """Return logo and accent data for a team."""
    manifest = load_asset_manifest()
    entry = manifest.get("teams", {}).get(team_name or "", {})
    primary_color = entry.get("primary_color") or FALLBACK_TEAM_COLORS.get(team_name or "", "#38bdf8")
    return {
        "primary_color": primary_color,
        "primary_soft": _rgba(primary_color, 0.14),
        "primary_line": _rgba(primary_color, 0.42),
        "primary_glow": _rgba(primary_color, 0.26),
        "primary_fill": _rgba(primary_color, 0.18),
        "logo_src": _asset_source(entry.get("logo")),
    }


def _driver_visuals(driver_code: str | None) -> dict[str, str | None]:
    """Return optional driver-photo metadata."""
    manifest = load_asset_manifest()
    entry = manifest.get("drivers", {}).get(driver_code or "", {})
    return {
        "photo_src": _asset_source(entry.get("photo")),
        "team_name": entry.get("team"),
    }


def _driver_asset_bundle(selected_state: RaceState) -> dict[str, str | None]:
    """Return combined driver/team visual metadata for one selected state."""
    driver = selected_state.selected_driver
    team_name = driver.team_name
    visuals = _team_visuals(team_name)
    visuals.update(_driver_visuals(driver.driver_code))
    visuals["team_name"] = team_name or visuals.get("team_name") or "Team n/a"
    return visuals


def _team_style_vars(color: str | None) -> str:
    """Return inline CSS custom properties for selected-team accents."""
    return (
        f"--team-accent:{color or '#38bdf8'};"
        f"--team-accent-soft:{_rgba(color, 0.14)};"
        f"--team-accent-line:{_rgba(color, 0.42)};"
        f"--team-accent-glow:{_rgba(color, 0.22)};"
    )


def _plain_text(value: Any, fallback: str = "n/a") -> str:
    """Return a plain display string with any tag-like fragments removed."""
    if value is None:
        return fallback
    text = HTML_TAG_RE.sub(" ", str(value))
    text = " ".join(text.split())
    return text or fallback


def _html_escape(value: Any) -> str:
    """Escape a value for small inline HTML snippets."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _compound_badge_html(compound: str | None, compact: bool = False) -> str:
    """Return compact HTML for a tyre compound badge."""
    compound_name = compound or "UNKNOWN"
    label = _compound_short(compound_name) if compact else compound_name
    color = COMPOUND_COLORS.get(compound_name, COMPOUND_COLORS["UNKNOWN"])
    text_color = "#070b12" if compound_name in {"MEDIUM", "HARD"} else "#f8fafc"
    return (
        f"<span class='compound-badge' style='background:{color}; color:{text_color};'>"
        f"{_html_escape(label)}</span>"
    )


def _compact_recommendation_summary(recommendation: Any) -> str:
    """Return a compact recommendation summary for order rows."""
    if recommendation is None:
        return "Overlay unavailable"
    target = recommendation.next_compound or "TBD"
    if recommendation.pit_in_laps is None:
        return f"Target {target}"
    if recommendation.pit_in_laps <= 1:
        return f"Pit now -> {target}"
    return f"Pit +{recommendation.pit_in_laps} -> {target}"


def _selection_defaults(states: list[RaceState], snapshot_meta: dict[str, Any]) -> str:
    """Keep selected-driver state in sync with the active snapshot."""
    available_keys = [_driver_key(state) for state in states]
    default_key = snapshot_meta.get("default_driver_code") or available_keys[0]
    previous_snapshot = st.session_state.get("selected_snapshot_key")
    if previous_snapshot != snapshot_meta["snapshot_key"]:
        st.session_state["selected_snapshot_key"] = snapshot_meta["snapshot_key"]
        st.session_state["selected_driver_code"] = default_key
    if st.session_state.get("selected_driver_code") not in available_keys:
        st.session_state["selected_driver_code"] = default_key
    return st.session_state["selected_driver_code"]


def _format_call_title(recommendation: Any) -> str:
    """Return a product-facing current-call title."""
    if recommendation is None:
        return "Overlay unavailable"
    target = recommendation.next_compound or "TBD"
    if recommendation.pit_in_laps is None:
        return f"Target {target}"
    if recommendation.pit_in_laps <= 1:
        return f"Pit now -> {target}"
    return f"Pit in {recommendation.pit_in_laps} laps -> {target}"


def _format_call_subtitle(recommendation: Any) -> str:
    """Return a compact strategy subtitle for the current call."""
    if recommendation is None:
        return "Deterministic overlay unavailable"
    segments = [recommendation.strategy_type or "strategy"]
    if recommendation.next_compound:
        target = f"target {recommendation.next_compound}"
        if recommendation.final_compound:
            target += f", then {recommendation.final_compound}"
        segments.append(target)
    if recommendation.estimated_total_race_time_s is not None:
        segments.append(f"{recommendation.estimated_total_race_time_s:.1f}s model time")
    return " | ".join(segments)


def _format_call_window(recommendation: Any) -> str | None:
    """Return a tight-margin timing-window note when available."""
    if recommendation is None or not recommendation.near_optimal_pit_window:
        return None
    band_start = min(recommendation.near_optimal_pit_window)
    band_end = max(recommendation.near_optimal_pit_window)
    if band_start == band_end:
        return f"Window edge: lap {band_start}"
    return f"Near-optimal window: laps {band_start}-{band_end}"


def _track_status_label(track_status: str | None) -> str:
    """Return a compact track-status label."""
    if not track_status or str(track_status) == "1":
        return "Green"
    return f"TS {track_status}"


def _schematic_track_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return the stable schematic track geometry used for the replay shell."""
    track_theta = np.linspace(0, 2 * np.pi, 500)
    track_x = 132 * np.cos(track_theta) + 36 * np.cos(2 * track_theta)
    track_y = 90 * np.sin(track_theta) - 18 * np.sin(3 * track_theta)
    return track_x, track_y


def _build_race_control_payload(
    states: list[RaceState],
    selected_driver_code: str,
    snapshot_meta: dict[str, Any],
    pit_sample_count: int,
) -> dict[str, Any]:
    """Build the Python-to-component payload contract for the custom board."""
    selected_state = next(
        (state for state in states if _driver_key(state) == selected_driver_code),
        states[0],
    )
    ordered_states = sorted(
        states,
        key=lambda state: (
            state.selected_driver.current_position is None,
            state.selected_driver.current_position if state.selected_driver.current_position is not None else 999,
            _driver_key(state),
        ),
    )

    track_x, track_y = _schematic_track_geometry()
    marker_theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(ordered_states), endpoint=False)
    markers = []
    for state, theta in zip(ordered_states, marker_theta):
        team_visuals = _team_visuals(state.selected_driver.team_name)
        marker_x = 132 * np.cos(theta) + 36 * np.cos(2 * theta)
        marker_y = 90 * np.sin(theta) - 18 * np.sin(3 * theta)
        markers.append(
            {
                "driver_code": _driver_key(state),
                "label": _driver_label(state),
                "position": state.selected_driver.current_position,
                "selected": _driver_key(state) == selected_driver_code,
                "x": round(float(marker_x), 2),
                "y": round(float(marker_y), 2),
                "color": team_visuals["primary_color"],
                "text_color": _contrast_text_color(team_visuals["primary_color"]),
                "team_color": team_visuals["primary_color"],
                "team_color_glow": team_visuals["primary_glow"],
            }
        )

    timing_rows = []
    for state in ordered_states:
        driver = state.selected_driver
        team_visuals = _team_visuals(driver.team_name)
        compound_name = driver.current_compound or "UNKNOWN"
        recommendation = state.recommendation
        timing_rows.append(
            {
                "driver_code": _driver_key(state),
                "name": _driver_label(state),
                "team": driver.team_name or "Team n/a",
                "team_short": _team_short(driver.team_name),
                "position": driver.current_position,
                "position_label": _format_position(driver.current_position),
                "start_position": driver.start_position,
                "compound": compound_name,
                "compound_short": _compound_short(compound_name),
                "compound_color": COMPOUND_COLORS.get(compound_name, COMPOUND_COLORS["UNKNOWN"]),
                "tyre_age": driver.tyre_age_laps,
                "tyre_age_label": str(driver.tyre_age_laps),
                "call_snippet": _compact_recommendation_summary(recommendation),
                "selected": _driver_key(state) == selected_driver_code,
                "team_logo_src": team_visuals["logo_src"],
                "team_color": team_visuals["primary_color"],
                "team_color_soft": team_visuals["primary_soft"],
                "team_color_line": team_visuals["primary_line"],
            }
        )

    selected_driver = selected_state.selected_driver
    asset_bundle = _driver_asset_bundle(selected_state)
    recommendation = selected_state.recommendation
    selected_stints = []
    for stint in selected_driver.stint_history:
        compound_name = stint.compound or "UNKNOWN"
        if stint.start_lap is not None and stint.end_lap is not None:
            lap_label = f"L{stint.start_lap}-{stint.end_lap}"
        elif stint.laps_completed is not None:
            lap_label = f"{stint.laps_completed}L"
        else:
            lap_label = "laps n/a"
        pos_label = f"{_format_position(stint.opening_position)}->{_format_position(stint.closing_position)}"
        selected_stints.append(
            {
                "label": f"{_compound_short(compound_name)}{int(stint.stint_number)}",
                "compound": compound_name,
                "summary": f"{lap_label} | {pos_label}",
                "color": COMPOUND_COLORS.get(compound_name, COMPOUND_COLORS["UNKNOWN"]),
            }
        )

    note_segments = [
        snapshot_meta["source_label"],
        f"lap {selected_state.lap_number}" if selected_state.lap_number is not None else "lap n/a",
        selected_state.event_name or selected_state.session_name or "synthetic checkpoint",
    ]
    if selected_state.notes:
        note_segments.append(selected_state.notes[0])

    return {
        "meta": {
            "source_mode": snapshot_meta["source_mode"],
            "title": "F1 Strategy Lab",
            "subtitle": snapshot_meta["headline"],
            "tags": [
                snapshot_meta["source_label"],
                snapshot_meta["headline"],
                f"Pit baseline: {pit_sample_count} samples",
            ],
        },
        "circuit": {
            "mode": "schematic",
            "name": selected_state.circuit_name or "Circuit replay map",
            "start_finish_label": "START",
            "start_finish": {
                "x": -150,
                "y": -84,
                "marker_x": -120,
                "marker_y": -84,
            },
            "path_points": [
                {"x": round(float(x), 2), "y": round(float(y), 2)}
                for x, y in zip(track_x, track_y)
            ],
            "markers": markers,
        },
        "timing": {
            "rows": timing_rows,
            "note": (
                "Single-state shell source: order panel only shows the available checkpoint."
                if not snapshot_meta["is_full_field"]
                else None
            ),
        },
        "selected_driver": {
            "driver": _driver_label(selected_state),
            "team": selected_driver.team_name or "Team n/a",
            "compound": selected_driver.current_compound or "UNKNOWN",
            "compound_color": COMPOUND_COLORS.get(
                selected_driver.current_compound or "UNKNOWN",
                COMPOUND_COLORS["UNKNOWN"],
            ),
            "start_position": selected_driver.start_position,
            "start_position_label": _format_position(selected_driver.start_position),
            "current_position": selected_driver.current_position,
            "current_position_label": _format_position(selected_driver.current_position),
            "laps_left": selected_state.laps_remaining,
            "laps_left_label": str(selected_state.laps_remaining),
            "tyre_age": selected_driver.tyre_age_laps,
            "tyre_age_label": str(selected_driver.tyre_age_laps),
            "stint": selected_driver.current_stint_number,
            "stint_label": str(selected_driver.current_stint_number or "n/a"),
            "track_status": selected_state.event_status.track_status,
            "track_status_label": _track_status_label(selected_state.event_status.track_status),
            "call_title": _format_call_title(recommendation),
            "call_subtitle": _format_call_subtitle(recommendation),
            "call_window": _format_call_window(recommendation),
            "support": recommendation.support_tier if recommendation else "n/a",
            "confidence": recommendation.confidence_label if recommendation else "Pending",
            "risk_notes": len(recommendation.risk_notes or []) if recommendation else 0,
            "risk_notes_label": str(len(recommendation.risk_notes or []) if recommendation else 0),
            "ahead": selected_state.competitor_context.ahead_driver_code or "n/a",
            "behind": selected_state.competitor_context.behind_driver_code or "n/a",
            "stints": selected_stints,
            "note": " | ".join(segment for segment in note_segments if segment),
            "team_logo_src": asset_bundle["logo_src"],
            "driver_photo_src": asset_bundle["photo_src"],
            "team_color": asset_bundle["primary_color"],
            "team_color_soft": asset_bundle["primary_soft"],
            "team_color_line": asset_bundle["primary_line"],
            "team_color_glow": asset_bundle["primary_glow"],
        },
    }


def apply_race_control_theme() -> None:
    """Apply compact race-control styling to the Streamlit shell."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #05070b;
            color: #e5e7eb;
        }
        [data-testid="stHeader"] {
            background: rgba(5, 7, 11, 0.96);
        }
        [data-testid="stSidebar"] {
            background: #06080d;
            border-right: 1px solid #111827;
        }
        [data-testid="stSidebar"] * {
            color: #b6c2d1;
        }
        [data-testid="stSidebar"] section {
            padding-top: 0.35rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 0.82rem !important;
            margin-bottom: 0.35rem !important;
        }
        [data-testid="stSidebar"] label {
            font-size: 0.68rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-size: 0.68rem;
            line-height: 1.2;
        }
        [data-testid="stSidebar"] [data-testid="stForm"],
        [data-testid="stSidebar"] div[data-baseweb="select"],
        [data-testid="stSidebar"] div[data-testid="stSlider"] {
            opacity: 0.82;
        }
        .block-container {
            max-width: 1560px;
            padding-top: 0.26rem;
            padding-bottom: 0.56rem;
        }
        h1 {
            font-size: 1.25rem !important;
            line-height: 1.1 !important;
            margin-bottom: 0.15rem !important;
            letter-spacing: 0 !important;
        }
        h2, h3, h4 {
            letter-spacing: 0 !important;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.18rem;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.22rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #060b13;
            border-color: #1a2536;
            padding: 0.36rem;
        }
        div[data-testid="stMetric"] {
            background: #0a101b;
            border: 1px solid #1b2638;
            border-radius: 4px;
            padding: 0.24rem 0.34rem;
        }
        div[data-testid="stMetric"] label {
            color: #94a3b8;
            font-size: 0.6rem;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 0.88rem;
        }
        div[data-testid="stCaptionContainer"] {
            color: #94a3b8;
        }
        .stButton > button {
            min-height: 1.16rem;
            height: 1.16rem;
            padding: 0 0.18rem;
            border-radius: 4px;
            border: 1px solid #1f2a3d;
            background: #0b1220;
            color: #e5e7eb;
            font-size: 0.56rem;
            font-weight: 800;
        }
        .stButton > button:hover {
            border-color: #38bdf8;
            color: #f8fafc;
            background: #0f172a;
        }
        .race-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 0.8rem;
            padding: 0.12rem 0 0.18rem 0;
            border-bottom: 1px solid #2a3a54;
            background: linear-gradient(90deg, rgba(14, 20, 32, 0.92), rgba(5, 7, 11, 0.2));
        }
        .race-title {
            color: #ffffff;
            font-size: 1.22rem;
            font-weight: 800;
            line-height: 1.05;
            text-shadow: 0 0 14px rgba(56, 189, 248, 0.22);
        }
        .race-subtitle {
            color: #cbd5e1;
            font-size: 0.66rem;
            margin-top: 0.02rem;
        }
        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.22rem;
            justify-content: flex-end;
        }
        .status-pill, .compound-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 1.05rem;
            padding: 0.05rem 0.38rem;
            border-radius: 4px;
            font-size: 0.58rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
            white-space: nowrap;
        }
        .status-pill {
            color: #e5eefb;
            background: #111827;
            border: 1px solid #32425c;
            box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.06);
        }
        .panel-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #f8fafc;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.06rem;
        }
        .panel-kicker {
            color: #94a3b8;
            font-size: 0.58rem;
            text-transform: uppercase;
            font-weight: 700;
        }
        .timing-head, .timing-row {
            display: grid;
            grid-template-columns: 2rem 2.75rem 3.95rem 1.55rem 2rem minmax(4.4rem, 1fr);
            align-items: center;
            gap: 0.18rem;
        }
        .timing-head {
            color: #64748b;
            font-size: 0.54rem;
            font-weight: 800;
            text-transform: uppercase;
            padding: 0 0.08rem 0.08rem 0.08rem;
        }
        .timing-row {
            min-height: 1.2rem;
            padding: 0.055rem 0.12rem;
            border-top: 1px solid #162033;
            font-size: 0.59rem;
            background: rgba(7, 12, 21, 0.82);
        }
        .timing-row.selected {
            background: linear-gradient(90deg, var(--team-accent-soft, rgba(56, 189, 248, 0.22)), rgba(15, 23, 42, 0.82));
            border-left: 2px solid var(--team-accent, #38bdf8);
            box-shadow: inset 0 0 0 1px var(--team-accent-line, rgba(56, 189, 248, 0.36)), 0 0 18px var(--team-accent-glow, rgba(56, 189, 248, 0.08));
            padding-left: 0.12rem;
        }
        .driver-code {
            color: #f8fafc;
            font-weight: 900;
        }
        .muted {
            color: #94a3b8;
        }
        .call-card {
            background: linear-gradient(135deg, #08111f 0%, #0f1b2d 100%);
            border: 1px solid var(--team-accent-line, #2a3a54);
            border-left: 4px solid var(--team-accent, #5eead4);
            border-radius: 5px;
            padding: 0.3rem 0.4rem;
            box-shadow: 0 0 18px var(--team-accent-glow, rgba(56, 189, 248, 0.08));
        }
        .call-action {
            color: #f8fafc;
            font-size: 0.98rem;
            font-weight: 900;
            line-height: 1.05;
            margin-bottom: 0.08rem;
        }
        .call-meta {
            color: #94a3b8;
            font-size: 0.62rem;
        }
        .stint-strip {
            display: flex;
            gap: 0.12rem;
            width: 100%;
            margin: 0.06rem 0 0.11rem 0;
        }
        .stint-segment {
            min-width: 2.55rem;
            flex: 1 1 0;
            border-radius: 4px;
            padding: 0.18rem 0.22rem;
            color: #08111f;
            font-weight: 800;
            font-size: 0.58rem;
        }
        .stint-laps {
            display: block;
            margin-top: 0.06rem;
            color: rgba(8, 17, 31, 0.7);
            font-size: 0.54rem;
            font-weight: 800;
        }
        .mini-note {
            color: #94a3b8;
            font-size: 0.58rem;
            line-height: 1.25;
        }
        .control-mode {
            margin: 0.16rem 0 0.1rem 0;
        }
        .control-mode div[role="radiogroup"] {
            gap: 0.35rem;
        }
        .drawer-shell {
            background: linear-gradient(180deg, #07101c 0%, #060b13 100%);
            border: 1px solid #2a3a54;
            border-left: 3px solid var(--team-accent, #5eead4);
            border-radius: 5px;
            padding: 0.28rem 0.3rem 0.3rem 0.3rem;
            box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.34), 0 0 18px var(--team-accent-glow, rgba(56, 189, 248, 0.1));
        }
        .drawer-identity {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.35rem;
            align-items: start;
            padding-bottom: 0.2rem;
            border-bottom: 1px solid #1e293b;
        }
        .identity-stack {
            display: flex;
            gap: 0.42rem;
            align-items: center;
            min-width: 0;
        }
        .identity-photo {
            width: 3.15rem;
            height: 3.15rem;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--team-accent-line, #2a3a54);
            background: linear-gradient(180deg, #0a1320 0%, #08111d 100%);
            flex: 0 0 auto;
        }
        .identity-photo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .identity-text {
            min-width: 0;
        }
        .drawer-driver {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 900;
            line-height: 1;
            overflow-wrap: anywhere;
        }
        .drawer-team {
            color: #94a3b8;
            font-size: 0.6rem;
            margin-top: 0.04rem;
            display: flex;
            gap: 0.28rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .identity-logo {
            width: 1.05rem;
            height: 1.05rem;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(10, 17, 29, 0.92);
            flex: 0 0 auto;
        }
        .identity-logo img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            padding: 0.1rem;
        }
        .state-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(5.4rem, 1fr));
            gap: 0.18rem;
            margin: 0.18rem 0;
        }
        .state-cell {
            background: #0b1220;
            border: 1px solid #1b2638;
            border-radius: 4px;
            padding: 0.18rem 0.22rem;
            min-width: 0;
            min-height: 2.1rem;
        }
        .state-label {
            display: block;
            color: #64748b;
            font-size: 0.5rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .state-value {
            display: block;
            color: #f8fafc;
            font-size: 0.72rem;
            font-weight: 900;
            margin-top: 0.02rem;
            overflow-wrap: anywhere;
            line-height: 1.12;
        }
        .risk-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(6rem, 1fr));
            gap: 0.18rem;
            margin-top: 0.16rem;
        }
        .risk-pill {
            border: 1px solid #1b2638;
            background: #0b1220;
            border-radius: 4px;
            padding: 0.16rem 0.22rem;
            min-width: 0;
            min-height: 2rem;
        }
        .risk-label {
            color: #64748b;
            display: block;
            font-size: 0.48rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .risk-value {
            color: #e5e7eb;
            display: block;
            font-size: 0.62rem;
            font-weight: 800;
            line-height: 1.18;
            overflow-wrap: anywhere;
        }
        div[data-testid="stExpander"] {
            border-color: #1e293b;
            background: rgba(7, 13, 22, 0.74);
        }
        div[data-testid="stExpander"] details summary {
            font-size: 0.66rem;
            color: #94a3b8;
        }
        .analysis-shell {
            background: #060b13;
            border: 1px solid #1b2638;
            border-radius: 5px;
            padding: 0.42rem 0.5rem;
            margin-top: 0.2rem;
        }
        .analysis-brief {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(7.4rem, 1fr));
            gap: 0.2rem;
            margin-bottom: 0.28rem;
        }
        .analysis-card {
            background: #0b1220;
            border: 1px solid #1b2638;
            border-radius: 4px;
            padding: 0.3rem 0.36rem;
            min-width: 0;
            min-height: 2.6rem;
        }
        .analysis-label {
            color: #64748b;
            display: block;
            font-size: 0.52rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .analysis-value {
            color: #f8fafc;
            display: block;
            font-size: 0.76rem;
            font-weight: 900;
            line-height: 1.12;
            overflow-wrap: anywhere;
        }
        .analysis-frame [data-testid="stDataFrame"],
        .analysis-frame [data-testid="stTable"],
        .analysis-frame [data-testid="stPlotlyChart"],
        .analysis-frame [data-testid="stImage"],
        .analysis-frame [data-testid="stCaptionContainer"] {
            margin-top: 0.12rem;
        }
        .analysis-frame div[data-testid="stTabs"] [role="tablist"] {
            gap: 0.22rem;
            margin-top: 0.1rem;
        }
        .analysis-frame div[data-testid="stTabs"] [role="tab"] {
            min-height: 1.55rem;
            padding: 0 0.55rem;
            border-radius: 4px 4px 0 0;
        }
        .team-inline {
            display: inline-flex;
            align-items: center;
            gap: 0.22rem;
            min-width: 0;
        }
        .team-inline img {
            width: 0.86rem;
            height: 0.86rem;
            border-radius: 999px;
            object-fit: contain;
            padding: 0.08rem;
            background: rgba(10, 17, 29, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    """Render the replay-first dashboard shell header."""
    st.set_page_config(page_title="F1 Strategy Lab", layout="wide")
    apply_race_control_theme()
    st.markdown(
        """
        <div class="race-header">
            <div>
                <div class="race-title">F1 Strategy Lab</div>
                <div class="race-subtitle">Race-control replay surface | Miami strategy lab</div>
            </div>
            <div class="status-strip">
                <span class="status-pill">Phase 3B</span>
                <span class="status-pill">Replay</span>
                <span class="status-pill">Single-car model</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls(available_compounds: list[str]) -> dict[str, Any]:
    """Render dashboard-source controls."""
    st.sidebar.markdown("### Source")
    source_mode = st.sidebar.selectbox(
        "Snapshot type",
        options=[
            "Historical replay checkpoint",
            "Representative validation scenario",
            "Manual demo scenario",
        ],
        help="Replay checkpoints are the main Phase 3B path. Validation and demo modes remain available as shell-compatible inputs.",
    )

    controls: dict[str, Any] = {"source_mode": source_mode}

    if source_mode == "Historical replay checkpoint":
        checkpoint_options = _pre3_checkpoint_options()
        backtest_summary = load_pre3_backtest_summary() or {}
        checkpoint_lookup = {
            item["checkpoint_id"]: item
            for item in backtest_summary.get("checkpoints", [])
        }
        controls["checkpoint_id"] = st.sidebar.selectbox(
            "Replay checkpoint",
            options=checkpoint_options,
            format_func=lambda checkpoint_id: (
                f"{checkpoint_lookup[checkpoint_id]['driver']} | lap {checkpoint_lookup[checkpoint_id]['checkpoint_lap']} | {checkpoint_id}"
            ),
        )
    elif source_mode == "Representative validation scenario":
        scenario_options = _phase2d_scenario_options()
        controls["scenario_id"] = st.sidebar.selectbox(
            "Validation scenario",
            options=scenario_options,
            format_func=lambda scenario_id: scenario_id.replace("_", " "),
        )
    else:
        default_current = "MEDIUM" if "MEDIUM" in available_compounds else available_compounds[0]
        controls["demo_compound"] = st.sidebar.selectbox(
            "Current compound",
            options=available_compounds,
            index=available_compounds.index(default_current),
        )
        controls["demo_tyre_life"] = st.sidebar.slider(
            "Current tyre age",
            min_value=1,
            max_value=40,
            value=5,
        )
        controls["demo_laps_remaining"] = st.sidebar.slider(
            "Laps remaining",
            min_value=2,
            max_value=58,
            value=25,
        )

    st.sidebar.divider()
    controls["show_placeholder_notes"] = st.sidebar.checkbox(
        "Show placeholder notes",
        value=False,
        help="Keep the replay-shell limitations visible in the UI.",
    )
    return controls


def build_dashboard_snapshot(
    controls: dict[str, Any],
) -> tuple[list[RaceState], dict[str, Any]]:
    """Build the currently selected dashboard snapshot."""
    source_mode = controls["source_mode"]
    if source_mode == "Historical replay checkpoint":
        payload = load_historical_replay_snapshot_payload(controls["checkpoint_id"])
        states = [RaceState.from_dict(item) for item in payload["states"]]
        return states, payload["meta"]

    if source_mode == "Representative validation scenario":
        payloads = load_phase2d_race_state_payloads()
        state = next(
            RaceState.from_dict(item)
            for item in payloads
            if item["state_id"] == f"phase2d:{controls['scenario_id']}"
        )
        return [state], {
            "snapshot_key": f"phase2d:{controls['scenario_id']}",
            "source_label": "Representative Validation Scenario",
            "source_mode": "phase2d_validation",
            "headline": f"Representative validation: {controls['scenario_id']}",
            "subheadline": "Synthetic scenario shell sourced from the Phase 2D artifact.",
            "mode_note": (
                "This source is a representative validation checkpoint, not a full replay field. "
                "It is useful for driver-detail shell testing, but not for a real order panel."
            ),
            "field_scope_note": (
                "Driver identity, team context, and full field order are placeholders here because the Phase 2D artifact stores a single canonical scenario state."
            ),
            "default_driver_code": _driver_key(state),
            "is_full_field": False,
            "reference_path": "data/processed/phase2d_validation_summary.json",
        }

    state = build_demo_race_state(
        ROOT,
        current_compound=controls["demo_compound"],
        current_tyre_life=int(controls["demo_tyre_life"]),
        laps_remaining=int(controls["demo_laps_remaining"]),
    )
    return [state], {
        "snapshot_key": state.state_id,
        "source_label": "Manual Demo Scenario",
        "source_mode": "demo_scenario",
        "headline": "Manual replay-shell demo scenario",
        "subheadline": "Synthetic race-state checkpoint driven by the current demo inputs.",
        "mode_note": (
            "This is the Phase 2/3 demo scenario shown inside the new dashboard shell. It is not a live feed and it is not a real historical field snapshot."
        ),
        "field_scope_note": (
            "Timing panel, nearby-car context, driver identity, and track position are intentionally minimal here because the source contains a single synthetic driver state."
        ),
        "default_driver_code": _driver_key(state),
        "is_full_field": False,
        "reference_path": "app/demo_strategy.py",
    }


def render_shell_banner(snapshot_meta: dict[str, Any], pit_sample_count: int) -> None:
    """Render the main shell banner."""
    st.markdown(
        f"""
        <div class="status-strip" style="justify-content:flex-start; margin:0.12rem 0 0.08rem 0;">
            <span class="status-pill">{_html_escape(snapshot_meta["source_label"])}</span>
            <span class="status-pill">{_html_escape(snapshot_meta["headline"])}</span>
            <span class="status-pill">Pit baseline: {pit_sample_count} samples</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_circuit_panel(
    states: list[RaceState],
    selected_driver_code: str,
    snapshot_meta: dict[str, Any],
) -> None:
    """Render a schematic circuit-view placeholder with stable ordered markers."""
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-title">
                <span>Circuit</span>
                <span class="panel-kicker">Schematic replay map</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ordered_states = sorted(
            states,
            key=lambda state: (
                state.selected_driver.current_position is None,
                state.selected_driver.current_position if state.selected_driver.current_position is not None else 999,
                _driver_key(state),
            ),
        )

        track_theta = np.linspace(0, 2 * np.pi, 500)
        track_x = 1.32 * np.cos(track_theta) + 0.36 * np.cos(2 * track_theta)
        track_y = 0.9 * np.sin(track_theta) - 0.18 * np.sin(3 * track_theta)

        fig, ax = plt.subplots(figsize=(9.2, 4.08))
        fig.patch.set_facecolor("#05070b")
        ax.set_facecolor("#05070b")

        for grid_x in np.linspace(-1.5, 1.5, 7):
            ax.axvline(grid_x, color="#0f172a", linewidth=0.45, alpha=0.72, zorder=0)
        for grid_y in np.linspace(-0.98, 0.98, 5):
            ax.axhline(grid_y, color="#0f172a", linewidth=0.45, alpha=0.72, zorder=0)

        ax.plot(track_x, track_y, color="#101827", linewidth=25, solid_capstyle="round", zorder=1)
        ax.plot(track_x, track_y, color="#94a3b8", linewidth=2.0, alpha=0.75, solid_capstyle="round", zorder=2)
        ax.plot(track_x * 0.99, track_y * 0.99, color="#020617", linewidth=10.5, solid_capstyle="round", zorder=3)
        ax.plot(track_x * 1.006, track_y * 1.006, color="#1e293b", linewidth=1.0, alpha=0.9, zorder=4)
        ax.fill(track_x * 0.48, track_y * 0.45, color="#07111f", alpha=0.72, zorder=0)

        marker_theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(ordered_states), endpoint=False)
        for state, theta in zip(ordered_states, marker_theta):
            marker_x = 1.32 * np.cos(theta) + 0.36 * np.cos(2 * theta)
            marker_y = 0.9 * np.sin(theta) - 0.18 * np.sin(3 * theta)
            is_selected = _driver_key(state) == selected_driver_code
            team_visuals = _team_visuals(state.selected_driver.team_name)
            marker_color = team_visuals["primary_color"]
            marker_size = 370 if is_selected else 104
            edge_color = team_visuals["primary_color"] if is_selected else "#020617"
            edge_width = 3.0 if is_selected else 1.3
            if is_selected:
                ax.scatter(
                    [marker_x],
                    [marker_y],
                    s=680,
                    c="none",
                    edgecolors=team_visuals["primary_color"],
                    linewidths=1.4,
                    alpha=0.7,
                    zorder=4,
                )

            ax.scatter(
                [marker_x],
                [marker_y],
                s=marker_size,
                c=marker_color,
                edgecolors=edge_color,
                linewidths=edge_width,
                zorder=4,
            )
            ax.text(
                marker_x,
                marker_y,
                _driver_label(state),
                ha="center",
                va="center",
                fontsize=6.6,
                fontweight="black",
                color=_contrast_text_color(marker_color),
                zorder=5,
            )

        ax.text(
            -1.55,
            0.98,
            "START",
            fontsize=7,
            fontweight="bold",
            color="#94a3b8",
            ha="left",
            va="center",
        )
        ax.scatter([-1.17], [0.84], s=50, c="#38bdf8", edgecolors="#e0f2fe", linewidths=1, zorder=6)
        ax.set_xlim(-1.62, 1.62)
        ax.set_ylim(-1.04, 1.04)
        ax.axis("off")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        if not snapshot_meta["is_full_field"]:
            st.caption("This source does not provide a real full field, so the circuit panel is intentionally minimal.")


def render_timing_panel(
    states: list[RaceState],
    selected_driver_code: str,
    snapshot_meta: dict[str, Any],
) -> str:
    """Render the race-control style timing panel."""
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-title">
                <span>Timing</span>
                <span class="panel-kicker">Order panel</span>
            </div>
            <div class="timing-head">
                <span>Pos</span><span>Drv</span><span>Team</span><span>Tyre</span><span>Age</span><span>Call</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not snapshot_meta["is_full_field"]:
            st.caption("Single-state shell source: order panel shows the available checkpoint only.")

        ordered_states = sorted(
            states,
            key=lambda state: (
                state.selected_driver.current_position is None,
                state.selected_driver.current_position if state.selected_driver.current_position is not None else 999,
                _driver_key(state),
            ),
        )
        for state in ordered_states:
            driver_code = _driver_key(state)
            team_visuals = _team_visuals(state.selected_driver.team_name)
            recommendation_text = _compact_recommendation_summary(state.recommendation)
            is_selected = driver_code == selected_driver_code
            row_class = "timing-row selected" if is_selected else "timing-row"
            team_html = _html_escape(_team_short(state.selected_driver.team_name))
            if team_visuals.get("logo_src"):
                team_html = (
                    f"<span class='team-inline'>"
                    f"<img src='{_html_escape(team_visuals['logo_src'])}' alt='{_html_escape(state.selected_driver.team_name or 'Team')} logo'>"
                    f"<span>{team_html}</span></span>"
                )
            row_cols = st.columns([9.55, 0.45])
            row_cols[0].markdown(
                f"""
                <div class="{row_class}" style="{_team_style_vars(team_visuals.get('primary_color'))}">
                    <span>{_html_escape(_format_position(state.selected_driver.current_position))}</span>
                    <span class="driver-code">{_html_escape(_driver_label(state))}</span>
                    <span class="muted">{team_html}</span>
                    <span>{_compound_badge_html(state.selected_driver.current_compound, compact=True)}</span>
                    <span>{int(state.selected_driver.tyre_age_laps)}</span>
                    <span class="muted">{_html_escape(recommendation_text)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if row_cols[1].button(
                ">" if not is_selected else "*",
                key=f"{snapshot_meta['snapshot_key']}:{driver_code}",
                width="stretch",
            ):
                st.session_state["selected_driver_code"] = driver_code
                selected_driver_code = driver_code

        return selected_driver_code


def render_driver_identity(selected_state: RaceState) -> None:
    """Render the selected-driver identity block."""
    driver = selected_state.selected_driver
    asset_bundle = _driver_asset_bundle(selected_state)
    photo_html = (
        f"<div class='identity-photo'><img src='{_html_escape(asset_bundle['photo_src'])}' alt='{_html_escape(_driver_label(selected_state))} photo'></div>"
        if asset_bundle.get("photo_src")
        else ""
    )
    logo_html = (
        f"<span class='identity-logo'><img src='{_html_escape(asset_bundle['logo_src'])}' alt='{_html_escape(driver.team_name or 'Team')} logo'></span>"
        if asset_bundle.get("logo_src")
        else ""
    )
    st.markdown(
        f"""
        <div class="drawer-identity" style="{_team_style_vars(asset_bundle.get('primary_color'))}">
            <div class="identity-stack">
                {photo_html}
                <div class="identity-text">
                    <div class="drawer-driver">{_html_escape(_driver_label(selected_state))}</div>
                    <div class="drawer-team">{logo_html}<span>{_html_escape(driver.team_name or "Team n/a")}</span></div>
                </div>
            </div>
            <div>{_compound_badge_html(driver.current_compound)}</div>
        </div>
        <div class="state-grid">
            <div class="state-cell"><span class="state-label">Start</span><span class="state-value">{_html_escape(_format_position(driver.start_position))}</span></div>
            <div class="state-cell"><span class="state-label">Now</span><span class="state-value">{_html_escape(_format_position(driver.current_position))}</span></div>
            <div class="state-cell"><span class="state-label">Laps left</span><span class="state-value">{_html_escape(selected_state.laps_remaining)}</span></div>
            <div class="state-cell"><span class="state-label">Tyre age</span><span class="state-value">{_html_escape(driver.tyre_age_laps)}</span></div>
            <div class="state-cell"><span class="state-label">Stint</span><span class="state-value">{_html_escape(driver.current_stint_number or "n/a")}</span></div>
            <div class="state-cell"><span class="state-label">Track</span><span class="state-value">{_html_escape(_track_status_label(selected_state.event_status.track_status))}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_driver_meta_line(selected_state: RaceState, snapshot_meta: dict[str, Any]) -> None:
    """Render a compact source line for the selected-driver panel."""
    st.markdown(
        f"""
        <div class="mini-note">
            {_html_escape(snapshot_meta["source_label"])} |
            lap {_html_escape(selected_state.lap_number if selected_state.lap_number is not None else "n/a")} |
            {_html_escape(selected_state.event_name or selected_state.session_name or "synthetic checkpoint")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_state_micro_context(selected_state: RaceState) -> None:
    """Render compact nearby-driver context without overclaiming gaps."""
    context = selected_state.competitor_context
    st.markdown(
        f"""
        <div class="state-grid" style="grid-template-columns:repeat(2,minmax(0,1fr)); margin-bottom:0;">
            <div class="state-cell"><span class="state-label">Ahead</span><span class="state-value">{_html_escape(context.ahead_driver_code or "n/a")}</span></div>
            <div class="state-cell"><span class="state-label">Behind</span><span class="state-value">{_html_escape(context.behind_driver_code or "n/a")}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_support_risk_strip(selected_state: RaceState) -> None:
    """Render compact support, confidence, and risk summary."""
    recommendation = selected_state.recommendation
    if recommendation is None:
        return
    risk_count = len(recommendation.risk_notes or [])
    st.markdown(
        f"""
        <div class="risk-strip">
            <div class="risk-pill"><span class="risk-label">Support</span><span class="risk-value">{_html_escape(recommendation.support_tier or "n/a")}</span></div>
            <div class="risk-pill"><span class="risk-label">Confidence</span><span class="risk-value">{_html_escape(recommendation.confidence_label or "Pending")}</span></div>
            <div class="risk-pill"><span class="risk-label">Risk notes</span><span class="risk-value">{risk_count}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_analysis_selected_driver_view_model(
    selected_state: RaceState,
    snapshot_meta: dict[str, Any],
    availability_summary: dict | None,
    show_placeholder_notes: bool,
) -> dict[str, Any]:
    """Build a plain-data view model for the native Analysis summary surface."""
    driver = selected_state.selected_driver
    recommendation = selected_state.recommendation
    asset_bundle = _driver_asset_bundle(selected_state)

    source_line = " | ".join(
        [
            _plain_text(snapshot_meta["source_label"]),
            f"lap {_plain_text(selected_state.lap_number)}",
            _plain_text(selected_state.event_name or selected_state.session_name or "synthetic checkpoint"),
        ]
    )

    stints: list[dict[str, str]] = []
    for stint in driver.stint_history:
        compound = stint.compound or "UNKNOWN"
        if stint.start_lap is not None and stint.end_lap is not None:
            lap_label = f"L{stint.start_lap}-{stint.end_lap}"
        elif stint.laps_completed is not None:
            lap_label = f"{stint.laps_completed} laps"
        else:
            lap_label = "laps n/a"
        stints.append(
            {
                "Stint": f"{_compound_short(compound)}{int(stint.stint_number)}",
                "Compound": _plain_text(compound),
                "Laps": lap_label,
                "Positions": f"{_format_position(stint.opening_position)} -> {_format_position(stint.closing_position)}",
            }
        )

    snapshot_notes = [_plain_text(note, "") for note in selected_state.notes[:3] if _plain_text(note, "")]
    if show_placeholder_notes:
        snapshot_notes.append(_plain_text(snapshot_meta["field_scope_note"], ""))
        if availability_summary:
            readiness = availability_summary.get("readiness_summary", {})
            future_gaps = readiness.get("not reliably available yet", [])[:4]
            if future_gaps:
                snapshot_notes.append("Still missing: " + ", ".join(_plain_text(item, "") for item in future_gaps))

    return {
        "driver": _plain_text(_driver_label(selected_state), "Unknown"),
        "team": _plain_text(driver.team_name or "Team n/a"),
        "team_badge_tone": _team_badge_tone(driver.team_name),
        "compound": _plain_text(driver.current_compound or "UNKNOWN"),
        "compound_short": _compound_short(driver.current_compound or "UNKNOWN"),
        "compound_badge_tone": _compound_badge_tone(driver.current_compound),
        "photo_src": asset_bundle.get("photo_src"),
        "logo_src": asset_bundle.get("logo_src"),
        "source_line": source_line,
        "metrics": [
            {"label": "Start", "value": _format_position(driver.start_position)},
            {"label": "Now", "value": _format_position(driver.current_position)},
            {"label": "Laps left", "value": _plain_text(selected_state.laps_remaining)},
            {"label": "Tyre age", "value": _plain_text(driver.tyre_age_laps)},
            {"label": "Stint", "value": _plain_text(driver.current_stint_number)},
            {"label": "Track", "value": _track_status_label(selected_state.event_status.track_status)},
        ],
        "call_title": _plain_text(_format_call_title(recommendation), "Overlay unavailable"),
        "call_subtitle": _plain_text(_format_call_subtitle(recommendation), ""),
        "call_window": _plain_text(_format_call_window(recommendation), "") if recommendation else "",
        "support": _plain_text(recommendation.support_tier if recommendation else None),
        "confidence": _plain_text(recommendation.confidence_label if recommendation else "Pending"),
        "risk_count": str(len(recommendation.risk_notes or []) if recommendation else 0),
        "ahead": _plain_text(selected_state.competitor_context.ahead_driver_code),
        "behind": _plain_text(selected_state.competitor_context.behind_driver_code),
        "stints": stints,
        "support_basis": _plain_text(recommendation.support_reason, "") if recommendation else "",
        "risk_notes": [_plain_text(note, "") for note in (recommendation.risk_notes or []) if _plain_text(note, "")] if recommendation else [],
        "snapshot_notes": snapshot_notes,
    }


def _render_native_summary_cards(items: list[dict[str, str]], columns_per_row: int = 3) -> None:
    """Render one list of label/value items as a stable native Streamlit card grid."""
    if not items:
        return
    for start in range(0, len(items), columns_per_row):
        chunk = items[start:start + columns_per_row]
        columns = st.columns(len(chunk))
        for column, item in zip(columns, chunk):
            with column:
                with st.container(border=True):
                    st.caption(item["label"])
                    st.markdown(f"**{_plain_text(item['value'])}**")


def render_analysis_selected_driver_summary(
    selected_state: RaceState,
    snapshot_meta: dict[str, Any],
    availability_summary: dict | None,
    show_placeholder_notes: bool,
) -> None:
    """Render the Analysis selected-car summary using plain data and native Streamlit blocks only."""
    model = _build_analysis_selected_driver_view_model(
        selected_state=selected_state,
        snapshot_meta=snapshot_meta,
        availability_summary=availability_summary,
        show_placeholder_notes=show_placeholder_notes,
    )

    with st.container(border=True):
        st.markdown("**Selected car**")
        st.caption("Analysis summary")

        if model["photo_src"]:
            header_columns = st.columns([0.16, 0.62, 0.22])
            with header_columns[0]:
                st.image(model["photo_src"], width=88)
        else:
            header_columns = st.columns([0.78, 0.22])

        info_col = header_columns[1] if model["photo_src"] else header_columns[0]
        compound_col = header_columns[2] if model["photo_src"] else header_columns[1]

        with info_col:
            st.subheader(model["driver"])
            st.markdown(_markdown_badge(model["team"], model["team_badge_tone"]))
            if model["logo_src"]:
                logo_col, team_col = st.columns([0.12, 0.88])
                with logo_col:
                    st.image(model["logo_src"], width=26)
                with team_col:
                    st.caption(model["team"])
            else:
                st.caption(model["team"])
            st.caption(model["source_line"])

        with compound_col:
            with st.container(border=True):
                st.caption("Compound")
                st.markdown(_markdown_badge(model["compound"], model["compound_badge_tone"]))
                st.caption(f"{model['compound_short']} current tyre")

        _render_native_summary_cards(model["metrics"], columns_per_row=3)

        with st.container(border=True):
            st.markdown(_markdown_badge("Current pit call", model["team_badge_tone"]))
            st.markdown(f"**{model['call_title']}**")
            if model["call_subtitle"]:
                st.caption(model["call_subtitle"])
            if model["call_window"]:
                st.caption(model["call_window"])

        _render_native_summary_cards(
            [
                {"label": "Support", "value": model["support"]},
                {"label": "Confidence", "value": model["confidence"]},
                {"label": "Risk notes", "value": model["risk_count"]},
            ],
            columns_per_row=3,
        )

        if model["stints"]:
            st.caption("Stint timeline")
            st.dataframe(pd.DataFrame(model["stints"]), width="stretch", hide_index=True)
        else:
            st.caption("No stint history is attached to this checkpoint yet.")

        _render_native_summary_cards(
            [
                {"label": "Ahead", "value": model["ahead"]},
                {"label": "Behind", "value": model["behind"]},
            ],
            columns_per_row=2,
        )

        if model["risk_notes"]:
            with st.expander("Risk notes", expanded=False):
                for note in model["risk_notes"]:
                    st.caption(note)

        if model["support_basis"]:
            with st.expander("Support basis", expanded=False):
                st.caption(model["support_basis"])

        if model["snapshot_notes"]:
            with st.expander("Snapshot notes", expanded=False):
                for note in model["snapshot_notes"]:
                    st.caption(note)


def render_driver_detail_snapshot(selected_state: RaceState, snapshot_meta: dict[str, Any]) -> None:
    """Render the primary compact selected-driver control view."""
    render_driver_identity(selected_state)
    render_recommendation_panel(selected_state)
    render_stint_history(selected_state)
    render_support_risk_strip(selected_state)
    render_state_micro_context(selected_state)
    render_driver_meta_line(selected_state, snapshot_meta)


def render_selected_driver_panel(
    selected_state: RaceState,
    snapshot_meta: dict[str, Any],
    availability_summary: dict | None,
    show_placeholder_notes: bool,
) -> None:
    """Render selected-driver detail as a compact right-side control panel."""
    asset_bundle = _driver_asset_bundle(selected_state)
    st.markdown(
        f"""
        <div class="drawer-shell" style="{_team_style_vars(asset_bundle.get('primary_color'))}">
            <div class="panel-title">
                <span>Selected car</span>
                <span class="panel-kicker">Tactical drawer</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    render_driver_detail_snapshot(selected_state, snapshot_meta)
    if selected_state.notes or show_placeholder_notes:
        with st.expander("Snapshot notes", expanded=False):
            for note in selected_state.notes[:3]:
                st.caption(note)
            if show_placeholder_notes:
                st.caption(snapshot_meta["field_scope_note"])
                if availability_summary:
                    readiness = availability_summary.get("readiness_summary", {})
                    future_gaps = readiness.get("not reliably available yet", [])[:4]
                    if future_gaps:
                        st.caption("Still missing: " + ", ".join(future_gaps))
    st.markdown("</div>", unsafe_allow_html=True)


def render_stint_history(selected_state: RaceState) -> None:
    """Render stint history as a compact strategy strip."""
    stint_history = selected_state.selected_driver.stint_history
    if not stint_history:
        st.caption("No stint history is attached to this checkpoint yet.")
        return

    segments = []
    for stint in stint_history:
        compound = stint.compound or "UNKNOWN"
        color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS["UNKNOWN"])
        if stint.start_lap is not None and stint.end_lap is not None:
            lap_label = f"L{stint.start_lap}-{stint.end_lap}"
        elif stint.laps_completed is not None:
            lap_label = f"{stint.laps_completed}L"
        else:
            lap_label = "laps n/a"
        pos_label = f"{_format_position(stint.opening_position)}->{_format_position(stint.closing_position)}"
        segments.append(
            f"""
            <div class="stint-segment" style="background:{color};">
                {_html_escape(_compound_short(compound))}{int(stint.stint_number)}
                <span class="stint-laps">{_html_escape(lap_label)} | {_html_escape(pos_label)}</span>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="panel-kicker">Stint timeline</div>
        <div class="stint-strip">{''.join(segments)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_panel(selected_state: RaceState) -> None:
    """Render the selected-driver recommendation block."""
    recommendation = selected_state.recommendation
    if recommendation is None:
        st.warning("No recommendation payload is attached to this checkpoint.")
        return
    team_visuals = _team_visuals(selected_state.selected_driver.team_name)

    if recommendation.pit_in_laps is None:
        headline = f"Recommendation available -> {recommendation.next_compound or 'TBD'}"
    elif recommendation.pit_in_laps <= 1:
        headline = f"Pit now -> {recommendation.next_compound or 'TBD'}"
    else:
        headline = f"Pit in {recommendation.pit_in_laps} laps -> {recommendation.next_compound or 'TBD'}"

    target = recommendation.next_compound or "TBD"
    total_time = (
        f"{recommendation.estimated_total_race_time_s:.1f}s model time"
        if recommendation.estimated_total_race_time_s is not None
        else "model time n/a"
    )
    final_note = f" | final {recommendation.final_compound}" if recommendation.final_compound else ""
    st.markdown(
        f"""
        <div class="call-card" style="{_team_style_vars(team_visuals.get('primary_color'))}">
            <div class="panel-kicker">Current pit call</div>
            <div class="call-action">{_html_escape(headline)}</div>
            <div class="call-meta">
                {_html_escape(recommendation.strategy_type or "strategy")}
                | target {_html_escape(target)}{_html_escape(final_note)} | {_html_escape(total_time)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if recommendation.near_optimal_pit_window:
        band_start = min(recommendation.near_optimal_pit_window)
        band_end = max(recommendation.near_optimal_pit_window)
        st.caption(f"Near-optimal window: laps {band_start}-{band_end}.")

    risk_notes = recommendation.risk_notes or []
    if risk_notes:
        with st.expander("Risk notes", expanded=False):
            for note in risk_notes[:4]:
                st.caption(note)

    if recommendation.support_reason:
        with st.expander("Support basis", expanded=False):
            st.caption(recommendation.support_reason)


def render_competitor_context(selected_state: RaceState) -> None:
    """Render nearby-car placeholders honestly."""
    st.markdown("#### Nearby Context")
    context = selected_state.competitor_context
    cols = st.columns(2)
    with cols[0]:
        st.metric("Ahead", context.ahead_driver_code or "n/a")
        if context.gap_ahead_seconds is None:
            st.caption("Exact gap not available yet.")
    with cols[1]:
        st.metric("Behind", context.behind_driver_code or "n/a")
        if context.gap_behind_seconds is None:
            st.caption("Exact gap not available yet.")

    if context.context_note:
        st.caption(context.context_note)


def render_model_status(deg_result: object) -> None:
    """Render model quality and support availability."""
    st.markdown("#### Active Model Stack")
    columns = st.columns(3)
    for index, compound in enumerate(["SOFT", "MEDIUM", "HARD"]):
        info = deg_result.get_model_info(compound)
        with columns[index]:
            with st.container(border=True):
                st.metric(compound, info.get("support_tier", "n/a"))
                st.caption(f"{info.get('samples', 0)} total model laps")
                st.caption(
                    f"Miami: {info.get('miami_model_type', 'n/a')} | "
                    f"2026: {info.get('recency_model_type', 'n/a')}"
                )


def render_data_context(hybrid_context: dict) -> None:
    """Render data-source and blending context."""
    st.markdown("#### Data Context")
    grouping = hybrid_context.get("data_grouping", [])
    metric_cols = st.columns(3)
    metric_cols[0].metric("Active pools", len(grouping))
    metric_cols[1].metric("Source laps", f"{hybrid_context.get('total_laps', 0):,}")
    metric_cols[2].metric("Blend", "Role-based")

    if grouping:
        for pool in grouping:
            st.caption(
                f"- {pool['name']}: {pool['sample_counts']['total_laps']:,} laps | role={pool['role']}"
            )


def render_phase2d_validation_note() -> None:
    """Render a concise representative-validation note."""
    summary_artifact = load_phase2d_validation_summary()
    if not summary_artifact:
        return

    metadata = summary_artifact.get("metadata", {})
    aggregate = summary_artifact.get("aggregate_summary", {})
    scenario_count = metadata.get("scenario_count", 0)
    stable = aggregate.get("stability_counts", {}).get("Stable", 0)
    moderate = aggregate.get("stability_counts", {}).get("Moderately Sensitive", 0)
    st.info(
        f"Phase 2D validation artifact loaded: {scenario_count} representative scenarios | "
        f"{stable} stable | {moderate} moderately sensitive."
    )


def render_comparison_table(all_ranked_plans: list, best_plan: Any, deg_result: object) -> None:
    """Render alternative strategies in a compact secondary table."""
    rows = []
    for index, plan in enumerate(all_ranked_plans[:5], start=1):
        time_diff = plan.total_race_time - best_plan.total_race_time
        rows.append(
            {
                "Rank": "Best" if index == 1 else str(index),
                "Type": plan.strategy_type,
                "Pit": f"+{plan.pit_lap}",
                "Next": plan.next_compound,
                "Final": plan.final_compound or "-",
                "Time": f"{plan.total_race_time:.1f}s",
                "Delta": "baseline" if index == 1 else f"+{time_diff:.2f}s",
                "Support": "/".join(
                    [
                        deg_result.get_support_info(plan.next_compound).get("support_tier", "n/a")
                    ]
                    + (
                        [deg_result.get_support_info(plan.final_compound).get("support_tier", "n/a")]
                        if plan.final_compound
                        else []
                    )
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_phase2c_sensitivity(
    best_plan: Any,
    pit_loss_value: float,
    deg_result: object,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
) -> None:
    """Render the Phase 2C sensitivity analysis for the selected state."""
    stability_assessment = assess_strategy_stability(
        baseline_plan=best_plan,
        pit_loss_value=pit_loss_value,
        degradation_models=deg_result,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Stability", stability_assessment.stability_label)
    metric_cols[1].metric(
        "Pit-loss sensitive",
        "Yes" if stability_assessment.pit_loss_sensitive else "No",
    )
    metric_cols[2].metric(
        "Wear sensitive",
        "Yes" if stability_assessment.degradation_sensitive else "No",
    )

    if stability_assessment.flip_conditions:
        for condition in stability_assessment.flip_conditions:
            st.caption(f"- {condition}")


def render_advanced_analysis(
    available_compounds: list[str],
    deg_result: object,
    pit_loss_value: float,
    current_tyre_life: int,
    laps_remaining: int,
    current_compound: str,
) -> None:
    """Render the pit-timing curve and lap-time grid for the selected state."""
    target_options = [compound for compound in available_compounds if compound != current_compound]
    if not target_options:
        st.caption("No alternate compound is available for timing-curve analysis.")
        return

    columns = st.columns(2)

    with columns[0]:
        target_compound = st.selectbox(
            "Target compound",
            options=target_options,
            key=f"advanced_target_{current_compound}_{current_tyre_life}_{laps_remaining}",
        )
        comparison_df = optimize_pit_window(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            compound=current_compound,
            post_pit_compound=target_compound,
        )
        fig, ax = plt.subplots(figsize=(7.0, 3.55))
        fig.patch.set_facecolor("#060b13")
        ax.set_facecolor("#08111f")
        ax.plot(
            comparison_df["PitLap"],
            comparison_df["TotalTime"],
            marker="o",
            linewidth=2,
            markersize=4,
            color="#38bdf8",
        )
        ax.set_xlabel("Pit in laps")
        ax.set_ylabel("Estimated total race time (s)")
        ax.set_title(f"{current_compound} -> {target_compound}")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.grid(alpha=0.22, color="#334155")
        fig.tight_layout(pad=0.7)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with columns[1]:
        prediction_rows = []
        for tyre_life in [1, 5, 10, 15, 20]:
            row = {"Tyre age": tyre_life}
            for compound in available_compounds:
                lap_time = deg_result.predict_lap_time(compound, tyre_life)
                row[compound] = f"{lap_time:.2f}s" if lap_time is not None else "n/a"
            prediction_rows.append(row)
        st.dataframe(pd.DataFrame(prediction_rows), width="stretch", hide_index=True)


def render_placeholder_notes(
    snapshot_meta: dict[str, Any],
    availability_summary: dict | None,
) -> None:
    """Render real-vs-placeholder notes."""
    with st.expander("Real vs placeholder data", expanded=False):
        st.write(snapshot_meta["field_scope_note"])
        if availability_summary:
            readiness = availability_summary.get("readiness_summary", {})
            ready_now = readiness.get("available now", [])[:6]
            future_gaps = readiness.get("not reliably available yet", [])[:6]
            if ready_now:
                st.caption("Ready now: " + ", ".join(ready_now))
            if future_gaps:
                st.caption("Still missing: " + ", ".join(future_gaps))


def render_driver_detail_area(
    selected_state: RaceState,
    snapshot_meta: dict[str, Any],
    deg_result: object,
    pit_loss_value: float,
    available_compounds: list[str],
    hybrid_context: dict,
    availability_summary: dict | None,
) -> None:
    """Render the selected-driver tabs and lower analysis sections."""
    tabs = st.tabs(["Overview", "Recommendation", "Advanced"])

    with tabs[0]:
        render_driver_identity(selected_state)
        st.caption(
            f"Source: {snapshot_meta['source_label']} | "
            f"lap {selected_state.lap_number if selected_state.lap_number is not None else 'n/a'} | "
            f"{selected_state.event_name or selected_state.session_name or 'synthetic checkpoint'}"
        )
        render_stint_history(selected_state)
        render_competitor_context(selected_state)
        if selected_state.notes:
            st.markdown("#### Checkpoint Notes")
            for note in selected_state.notes:
                st.caption(f"- {note}")
        render_placeholder_notes(snapshot_meta, availability_summary)

    with tabs[1]:
        render_recommendation_panel(selected_state)

    with tabs[2]:
        current_compound = selected_state.selected_driver.current_compound
        current_tyre_life = selected_state.selected_driver.tyre_age_laps
        laps_remaining = selected_state.laps_remaining

        best_plan, all_ranked_plans = recommend_best_strategy(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            candidate_compounds=available_compounds,
            include_two_stop=True,
        )
        timing_trace = build_strategy_timing_trace(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            next_compound=best_plan.next_compound,
            final_compound=best_plan.final_compound,
        )

        with st.expander("Alternative strategies", expanded=False):
            render_comparison_table(all_ranked_plans, best_plan, deg_result)

        with st.expander("Sensitivity and timing-trace context", expanded=True):
            render_phase2c_sensitivity(
                best_plan=best_plan,
                pit_loss_value=pit_loss_value,
                deg_result=deg_result,
                current_compound=current_compound,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
            )
            st.caption(
                f"Timing trace shape: {timing_trace['curve_shape']} | "
                f"near-optimal first-stop laps: {timing_trace['near_optimal_band_laps']}"
            )

        with st.expander("Model and replay-shell context", expanded=False):
            render_model_status(deg_result)
            render_data_context(hybrid_context)
            render_phase2d_validation_note()

        with st.expander("Timing curve and lap-time grid", expanded=False):
            render_advanced_analysis(
                available_compounds=available_compounds,
                deg_result=deg_result,
                pit_loss_value=pit_loss_value,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
                current_compound=current_compound,
            )


def render_secondary_analysis_area(
    selected_state: RaceState,
    deg_result: object,
    pit_loss_value: float,
    available_compounds: list[str],
    hybrid_context: dict,
) -> None:
    """Render lower-priority analysis without dominating the control surface."""
    current_compound = selected_state.selected_driver.current_compound
    current_tyre_life = selected_state.selected_driver.tyre_age_laps
    laps_remaining = selected_state.laps_remaining

    try:
        best_plan, all_ranked_plans = recommend_best_strategy(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            candidate_compounds=available_compounds,
            include_two_stop=True,
        )
        timing_trace = build_strategy_timing_trace(
            degradation_models=deg_result,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            next_compound=best_plan.next_compound,
            final_compound=best_plan.final_compound,
        )
    except Exception as exc:
        st.warning(f"Deep-dive analysis unavailable for this checkpoint: {exc}")
        return

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="analysis-shell">
                <div class="analysis-brief">
                    <div class="analysis-card"><span class="analysis-label">Best type</span><span class="analysis-value">{_html_escape(best_plan.strategy_type)}</span></div>
                    <div class="analysis-card"><span class="analysis-label">First stop</span><span class="analysis-value">+{_html_escape(best_plan.pit_lap)} laps</span></div>
                    <div class="analysis-card"><span class="analysis-label">Next tyre</span><span class="analysis-value">{_html_escape(best_plan.next_compound)}</span></div>
                    <div class="analysis-card"><span class="analysis-label">Window</span><span class="analysis-value">{_html_escape(timing_trace["curve_shape"])}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tabs = st.tabs(["Alternatives", "Sensitivity", "Model", "Timing"])
        with tabs[0]:
            render_comparison_table(all_ranked_plans, best_plan, deg_result)
        with tabs[1]:
            render_phase2c_sensitivity(
                best_plan=best_plan,
                pit_loss_value=pit_loss_value,
                deg_result=deg_result,
                current_compound=current_compound,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
            )
            st.caption(
                f"Timing shape: {timing_trace['curve_shape']} | "
                f"near-optimal first-stop laps: {timing_trace['near_optimal_band_laps']}"
            )
        with tabs[2]:
            render_model_status(deg_result)
            render_data_context(hybrid_context)
            render_phase2d_validation_note()
        with tabs[3]:
            render_advanced_analysis(
                available_compounds=available_compounds,
                deg_result=deg_result,
                pit_loss_value=pit_loss_value,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
                current_compound=current_compound,
            )


def render_view_switch() -> str:
    """Render the primary product-mode switch."""
    st.markdown('<div class="control-mode">', unsafe_allow_html=True)
    selected_view = st.radio(
        "Dashboard view",
        options=["Race Control", "Analysis"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return selected_view


def render_assumptions_footer(snapshot_meta: dict[str, Any]) -> None:
    """Render the global assumptions footer."""
    with st.expander("Snapshot limits", expanded=False):
        st.caption(
            f"""
            Source mode: {snapshot_meta['source_mode']}. Replay order, tyres, and stint state are shown only when
            the selected source provides them. The circuit is schematic, exact gaps are not reliably available yet,
            and the strategy overlay remains deterministic, single-car, and not traffic/SC/VSC/weather aware.
            """
        )


def render_race_control_fallback(
    states: list[RaceState],
    selected_driver_code: str,
    snapshot_meta: dict[str, Any],
    availability_summary: dict | None,
    show_placeholder_notes: bool,
) -> str:
    """Render the native Streamlit race-control shell as the fallback surface."""
    circuit_col, timing_col = st.columns([1.55, 0.95])
    with circuit_col:
        render_circuit_panel(states, selected_driver_code, snapshot_meta)
    with timing_col:
        selected_driver_code = render_timing_panel(states, selected_driver_code, snapshot_meta)

    selected_state = next(
        (state for state in states if _driver_key(state) == selected_driver_code),
        states[0],
    )
    with timing_col:
        render_selected_driver_panel(
            selected_state=selected_state,
            snapshot_meta=snapshot_meta,
            availability_summary=availability_summary,
            show_placeholder_notes=show_placeholder_notes,
        )
    return selected_driver_code


def render_race_control_surface(
    states: list[RaceState],
    selected_driver_code: str,
    snapshot_meta: dict[str, Any],
    availability_summary: dict | None,
    pit_sample_count: int,
    show_placeholder_notes: bool,
) -> str:
    """Render the custom race-control component with a native fallback path."""
    if not race_control_board_available():
        st.caption("Custom race-control surface unavailable here. Using the built-in fallback shell.")
        return render_race_control_fallback(
            states=states,
            selected_driver_code=selected_driver_code,
            snapshot_meta=snapshot_meta,
            availability_summary=availability_summary,
            show_placeholder_notes=show_placeholder_notes,
        )

    payload = _build_race_control_payload(
        states=states,
        selected_driver_code=selected_driver_code,
        snapshot_meta=snapshot_meta,
        pit_sample_count=pit_sample_count,
    )
    try:
        next_selected_driver = render_race_control_board(
            payload,
            selected_driver=selected_driver_code,
            key=f"race_control_board:{snapshot_meta['snapshot_key']}",
        )
        available_driver_keys = {_driver_key(state) for state in states}
        if next_selected_driver in available_driver_keys:
            st.session_state["selected_driver_code"] = next_selected_driver
            return next_selected_driver
    except Exception as exc:
        st.caption(f"Custom race-control surface failed to load ({exc}). Using the built-in fallback shell.")
        return render_race_control_fallback(
            states=states,
            selected_driver_code=selected_driver_code,
            snapshot_meta=snapshot_meta,
            availability_summary=availability_summary,
            show_placeholder_notes=show_placeholder_notes,
        )
    return selected_driver_code


def main() -> None:
    """Main application entry point."""
    render_page_header()

    try:
        hybrid_context = load_hybrid_context()
        deg_result, pit_loss_value, pit_sample_count = build_integrated_pipeline()
        availability_summary = load_phase3a_availability_summary()
    except Exception as exc:
        st.error(f"Failed to initialize the canonical dashboard pipeline: {exc}")
        st.stop()

    available_compounds = ["SOFT", "MEDIUM", "HARD"]
    available_compounds = [
        compound
        for compound in available_compounds
        if deg_result.get_model_info(compound).get("model_type")
    ]
    if not available_compounds:
        st.error("No degradation models are available. Check the local data build.")
        st.stop()

    controls = render_sidebar_controls(available_compounds)
    try:
        states, snapshot_meta = build_dashboard_snapshot(controls)
    except Exception as exc:
        st.error(f"Failed to build the selected dashboard snapshot: {exc}")
        st.stop()

    selected_driver_code = _selection_defaults(states, snapshot_meta)
    render_shell_banner(snapshot_meta, pit_sample_count)
    selected_view = render_view_switch()

    if selected_view == "Race Control":
        selected_driver_code = render_race_control_surface(
            states=states,
            selected_driver_code=selected_driver_code,
            snapshot_meta=snapshot_meta,
            availability_summary=availability_summary,
            pit_sample_count=pit_sample_count,
            show_placeholder_notes=bool(controls.get("show_placeholder_notes")),
        )
        render_assumptions_footer(snapshot_meta)
        return

    selected_state = next(
        (
            state
            for state in states
            if _driver_key(state) == selected_driver_code
        ),
        states[0],
    )
    st.markdown(
        """
        <div class="panel-title" style="margin-top:0.25rem;">
            <span>Analysis</span>
            <span class="panel-kicker">Secondary deep dive</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_analysis_selected_driver_summary(
        selected_state=selected_state,
        snapshot_meta=snapshot_meta,
        availability_summary=availability_summary,
        show_placeholder_notes=bool(controls.get("show_placeholder_notes")),
    )
    render_secondary_analysis_area(
        selected_state=selected_state,
        deg_result=deg_result,
        pit_loss_value=pit_loss_value,
        available_compounds=available_compounds,
        hybrid_context=hybrid_context,
    )
    render_assumptions_footer(snapshot_meta)


if __name__ == "__main__":
    main()
