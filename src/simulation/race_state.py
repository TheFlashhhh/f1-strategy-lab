"""Canonical Phase 3A race-state schema and extraction helpers.

This module defines the future-facing recommendation checkpoint object for the
F1 Strategy Lab dashboard. Phase 3A is groundwork only:

- active now: deterministic strategy output, current compound / tyre age, raw
  race/session identity, lap position, and stint history that can be built from
  historical replay data
- derivable now: start position, position change, nearby car identities, and
  other fields that can be computed from existing lap-level data with light work
- future placeholders: explicit gap timing, SC/VSC booleans, driver photos,
  track-map coordinates, weather, tyre inventory, and live-feed specific fields

The current strategy engine does NOT depend on these objects yet. They exist so
future dashboard and replay work can share one canonical schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

import pandas as pd

from src.features.hybrid_modeling import build_role_based_hybrid_model
from src.simulation.strategy import estimate_pit_loss_window
from src.simulation.strategy_engine import build_strategy_timing_trace, recommend_best_strategy
from src.simulation.strategy_sensitivity import assess_strategy_stability


SchemaLifecycle = Literal["active_now", "derivable_now", "future_placeholder"]

SCHEMA_VERSION = "phase3a.v1"


@dataclass(frozen=True)
class SchemaFieldDefinition:
    """Describe one canonical race-state field and its Phase 3A lifecycle."""

    path: str
    lifecycle: SchemaLifecycle
    description: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the schema field definition."""
        return {
            "path": self.path,
            "lifecycle": self.lifecycle,
            "description": self.description,
            "notes": self.notes,
        }


SCHEMA_FIELD_CATALOG: tuple[SchemaFieldDefinition, ...] = (
    SchemaFieldDefinition(
        path="state_id",
        lifecycle="active_now",
        description="Canonical checkpoint identifier.",
        notes="Used for replay, artifact, and dashboard references.",
    ),
    SchemaFieldDefinition(
        path="source_type",
        lifecycle="active_now",
        description="Where the checkpoint came from.",
        notes="Examples: demo, phase2d_validation, pre3_backtest.",
    ),
    SchemaFieldDefinition(
        path="race_id",
        lifecycle="active_now",
        description="Race/session identity for the checkpoint.",
        notes="Synthetic IDs are allowed for demo/validation scenarios.",
    ),
    SchemaFieldDefinition(
        path="season",
        lifecycle="active_now",
        description="Season year when known.",
        notes="Backtest and raw replay checkpoints populate this directly.",
    ),
    SchemaFieldDefinition(
        path="event_name",
        lifecycle="active_now",
        description="Grand Prix or scenario event label.",
        notes="Validation scenarios use a synthetic event label.",
    ),
    SchemaFieldDefinition(
        path="session_name",
        lifecycle="active_now",
        description="Session label for the checkpoint.",
        notes="Historical replay checkpoints use Race; synthetic checkpoints use scenario labels.",
    ),
    SchemaFieldDefinition(
        path="circuit_name",
        lifecycle="active_now",
        description="Human-readable circuit identity.",
        notes="Phase 3A still uses Miami-reference or Miami historical context.",
    ),
    SchemaFieldDefinition(
        path="circuit_id",
        lifecycle="derivable_now",
        description="Normalized circuit identifier.",
        notes="Derived from circuit_name / event_name with light normalization.",
    ),
    SchemaFieldDefinition(
        path="lap_number",
        lifecycle="active_now",
        description="Current lap when known.",
        notes="Manual demo states may leave this empty while still providing laps_remaining.",
    ),
    SchemaFieldDefinition(
        path="laps_remaining",
        lifecycle="active_now",
        description="Laps remaining to race end.",
        notes="Raw replay data can derive this from total_laps and lap_number.",
    ),
    SchemaFieldDefinition(
        path="total_laps",
        lifecycle="derivable_now",
        description="Planned or observed race distance.",
        notes="Derived from raw replay data where lap_number coverage exists.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.driver_code",
        lifecycle="active_now",
        description="Driver code for the selected car.",
        notes="Historical raw data currently provides driver codes, not rich profile metadata.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.display_name",
        lifecycle="future_placeholder",
        description="Driver display name.",
        notes="Reserved for richer dashboard presentation once a driver lookup table exists.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.driver_photo_url",
        lifecycle="future_placeholder",
        description="Driver photo or asset reference.",
        notes="Phase 3A keeps the placeholder only; no asset pipeline exists yet.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.team_name",
        lifecycle="active_now",
        description="Team / constructor identity.",
        notes="Available in canonical raw parquet data.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.car_number",
        lifecycle="future_placeholder",
        description="Car number or richer car identity.",
        notes="Current canonical raw parquet does not retain car number.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.start_position",
        lifecycle="derivable_now",
        description="Grid/start position.",
        notes="Can be derived from the driver's earliest lap position in replay data.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.current_position",
        lifecycle="active_now",
        description="Current race order position.",
        notes="Available on most canonical replay rows with a few missing values.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.position_change",
        lifecycle="derivable_now",
        description="Change versus start position.",
        notes="Derived from start_position and current_position.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.current_compound",
        lifecycle="active_now",
        description="Current tyre compound.",
        notes="Required for every recommendation checkpoint.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.tyre_age_laps",
        lifecycle="active_now",
        description="Current tyre age in laps.",
        notes="Required for every recommendation checkpoint.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.current_stint_number",
        lifecycle="active_now",
        description="Current stint counter when known.",
        notes="Available in canonical replay data and manual demo assumptions.",
    ),
    SchemaFieldDefinition(
        path="selected_driver.stint_history",
        lifecycle="derivable_now",
        description="Historical stint timeline up to the checkpoint.",
        notes="Built from lap_number, stint, compound, and position.",
    ),
    SchemaFieldDefinition(
        path="competitor_context.ahead_driver_code",
        lifecycle="derivable_now",
        description="Identity of the car ahead.",
        notes="Can be derived from lap-level position ordering when positions are present.",
    ),
    SchemaFieldDefinition(
        path="competitor_context.behind_driver_code",
        lifecycle="derivable_now",
        description="Identity of the car behind.",
        notes="Can be derived from lap-level position ordering when positions are present.",
    ),
    SchemaFieldDefinition(
        path="competitor_context.gap_ahead_seconds",
        lifecycle="future_placeholder",
        description="Time gap to the car ahead.",
        notes="Canonical raw data does not currently provide reliable interval/gap timing.",
    ),
    SchemaFieldDefinition(
        path="competitor_context.gap_behind_seconds",
        lifecycle="future_placeholder",
        description="Time gap to the car behind.",
        notes="Canonical raw data does not currently provide reliable interval/gap timing.",
    ),
    SchemaFieldDefinition(
        path="recommendation.recommended_action",
        lifecycle="active_now",
        description="Current deterministic recommendation summary.",
        notes="Derived from the existing strategy engine or canonical validation/backtest artifacts.",
    ),
    SchemaFieldDefinition(
        path="recommendation.pit_in_laps",
        lifecycle="active_now",
        description="Recommended first-stop timing relative to now.",
        notes="Already present in the existing strategy artifacts.",
    ),
    SchemaFieldDefinition(
        path="recommendation.next_compound",
        lifecycle="active_now",
        description="Next tyre choice.",
        notes="Already present in the existing strategy artifacts.",
    ),
    SchemaFieldDefinition(
        path="recommendation.final_compound",
        lifecycle="active_now",
        description="Final tyre for two-stop plans.",
        notes="Left empty for one-stop plans.",
    ),
    SchemaFieldDefinition(
        path="recommendation.support_tier",
        lifecycle="active_now",
        description="Support / trust tier for the recommended future compound path.",
        notes="Built from current support-tier artifacts or the live role-based hybrid model.",
    ),
    SchemaFieldDefinition(
        path="recommendation.confidence_label",
        lifecycle="active_now",
        description="High-level confidence or stability label.",
        notes="Populated from Phase 2C stability analysis when available.",
    ),
    SchemaFieldDefinition(
        path="recommendation.risk_notes",
        lifecycle="active_now",
        description="Short risk / caution notes.",
        notes="Used for flat timing windows, flip conditions, and support warnings.",
    ),
    SchemaFieldDefinition(
        path="event_status.track_status",
        lifecycle="active_now",
        description="Raw track-status code or assumption label.",
        notes="Available in replay data and set explicitly for manual demo assumptions.",
    ),
    SchemaFieldDefinition(
        path="event_status.safety_car_active",
        lifecycle="future_placeholder",
        description="Explicit Safety Car flag.",
        notes="Phase 3A reserves the field but does not implement SC logic yet.",
    ),
    SchemaFieldDefinition(
        path="event_status.virtual_safety_car_active",
        lifecycle="future_placeholder",
        description="Explicit Virtual Safety Car flag.",
        notes="Phase 3A reserves the field but does not implement VSC logic yet.",
    ),
    SchemaFieldDefinition(
        path="event_status.weather_state",
        lifecycle="future_placeholder",
        description="Weather state or weather-source reference.",
        notes="Reserved for later realism phases once weather data is integrated.",
    ),
)


def schema_field_catalog_to_dict() -> list[dict[str, str]]:
    """Return the canonical field catalog as JSON-friendly dictionaries."""
    return [field_def.to_dict() for field_def in SCHEMA_FIELD_CATALOG]


def _slugify(value: Optional[str], fallback: str = "unknown") -> str:
    """Return a filesystem/identifier-safe slug."""
    if value is None:
        return fallback
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return text or fallback


def _int_or_none(value: Any) -> Optional[int]:
    """Convert a scalar-like value to int when possible."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def _float_or_none(value: Any) -> Optional[float]:
    """Convert a scalar-like value to float when possible."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _string_or_none(value: Any) -> Optional[str]:
    """Return a clean string or None."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@dataclass
class StintSummary:
    """Summary of one completed or active stint.

    Active now:
    - stint_number, compound, lap bounds, and opening/closing positions can be
      built from existing replay datasets
    Future placeholders:
    - richer lap-shape statistics and tyre-set inventory links can be added later
    """

    stint_number: int
    compound: Optional[str]
    start_lap: Optional[int]
    end_lap: Optional[int]
    laps_completed: Optional[int]
    opening_position: Optional[int] = None
    closing_position: Optional[int] = None

    def validate(self) -> None:
        """Validate the stint summary."""
        if self.stint_number <= 0:
            raise ValueError("stint_number must be positive")
        if self.start_lap is not None and self.end_lap is not None and self.end_lap < self.start_lap:
            raise ValueError("end_lap must be >= start_lap")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stint summary."""
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StintSummary":
        """Deserialize a stint summary."""
        obj = cls(
            stint_number=int(payload["stint_number"]),
            compound=_string_or_none(payload.get("compound")),
            start_lap=_int_or_none(payload.get("start_lap")),
            end_lap=_int_or_none(payload.get("end_lap")),
            laps_completed=_int_or_none(payload.get("laps_completed")),
            opening_position=_int_or_none(payload.get("opening_position")),
            closing_position=_int_or_none(payload.get("closing_position")),
        )
        obj.validate()
        return obj


@dataclass
class CompetitorGapContext:
    """Nearby-car context for a recommendation checkpoint.

    Active / derivable now:
    - ahead/behind identities and positions can be derived from lap-level order
    Future placeholders:
    - explicit gap timing remains reserved until a reliable interval feed exists
    """

    ahead_driver_code: Optional[str] = None
    ahead_team_name: Optional[str] = None
    ahead_position: Optional[int] = None
    behind_driver_code: Optional[str] = None
    behind_team_name: Optional[str] = None
    behind_position: Optional[int] = None
    gap_ahead_seconds: Optional[float] = None
    gap_behind_seconds: Optional[float] = None
    context_note: Optional[str] = None

    def validate(self) -> None:
        """Validate the competitor context."""
        for field_name in ("ahead_position", "behind_position"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the competitor context."""
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompetitorGapContext":
        """Deserialize competitor context."""
        obj = cls(
            ahead_driver_code=_string_or_none(payload.get("ahead_driver_code")),
            ahead_team_name=_string_or_none(payload.get("ahead_team_name")),
            ahead_position=_int_or_none(payload.get("ahead_position")),
            behind_driver_code=_string_or_none(payload.get("behind_driver_code")),
            behind_team_name=_string_or_none(payload.get("behind_team_name")),
            behind_position=_int_or_none(payload.get("behind_position")),
            gap_ahead_seconds=_float_or_none(payload.get("gap_ahead_seconds")),
            gap_behind_seconds=_float_or_none(payload.get("gap_behind_seconds")),
            context_note=_string_or_none(payload.get("context_note")),
        )
        obj.validate()
        return obj


@dataclass
class RecommendationState:
    """Serialized recommendation payload plus support / risk metadata.

    Active now:
    - pit timing, tyre choice, feasibility, support tiers, stability label, and
      flat-window risk notes all come from the existing deterministic stack
    Future placeholders:
    - competitor-aware explanation fields can be added later without changing
      the checkpoint envelope
    """

    source: str
    recommended_action: Optional[str] = None
    strategy_type: Optional[str] = None
    pit_in_laps: Optional[int] = None
    pit_on_lap: Optional[int] = None
    second_pit_in_laps: Optional[int] = None
    next_compound: Optional[str] = None
    final_compound: Optional[str] = None
    estimated_total_race_time_s: Optional[float] = None
    feasible: Optional[bool] = None
    feasibility_reason: Optional[str] = None
    support_tier: Optional[str] = None
    support_reason: Optional[str] = None
    confidence_label: Optional[str] = None
    confidence_reason: Optional[str] = None
    risk_notes: list[str] = field(default_factory=list)
    timing_window_shape: Optional[str] = None
    near_optimal_pit_window: list[int] = field(default_factory=list)

    def validate(self) -> None:
        """Validate the recommendation payload."""
        if not self.source:
            raise ValueError("recommendation.source is required")
        for field_name in ("pit_in_laps", "pit_on_lap", "second_pit_in_laps"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0 when provided")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the recommendation payload."""
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecommendationState":
        """Deserialize the recommendation payload."""
        obj = cls(
            source=str(payload["source"]),
            recommended_action=_string_or_none(payload.get("recommended_action")),
            strategy_type=_string_or_none(payload.get("strategy_type")),
            pit_in_laps=_int_or_none(payload.get("pit_in_laps")),
            pit_on_lap=_int_or_none(payload.get("pit_on_lap")),
            second_pit_in_laps=_int_or_none(payload.get("second_pit_in_laps")),
            next_compound=_string_or_none(payload.get("next_compound")),
            final_compound=_string_or_none(payload.get("final_compound")),
            estimated_total_race_time_s=_float_or_none(payload.get("estimated_total_race_time_s")),
            feasible=payload.get("feasible"),
            feasibility_reason=_string_or_none(payload.get("feasibility_reason")),
            support_tier=_string_or_none(payload.get("support_tier")),
            support_reason=_string_or_none(payload.get("support_reason")),
            confidence_label=_string_or_none(payload.get("confidence_label")),
            confidence_reason=_string_or_none(payload.get("confidence_reason")),
            risk_notes=[str(note) for note in payload.get("risk_notes", [])],
            timing_window_shape=_string_or_none(payload.get("timing_window_shape")),
            near_optimal_pit_window=[int(value) for value in payload.get("near_optimal_pit_window", [])],
        )
        obj.validate()
        return obj


@dataclass
class EventStatus:
    """Track/event status for the checkpoint.

    Active now:
    - raw track_status code can be stored directly
    Future placeholders:
    - explicit SC/VSC booleans and weather state remain placeholders until the
      current pipeline grows those feeds
    """

    track_status: Optional[str] = None
    safety_car_active: Optional[bool] = None
    virtual_safety_car_active: Optional[bool] = None
    weather_state: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize event status."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventStatus":
        """Deserialize event status."""
        return cls(
            track_status=_string_or_none(payload.get("track_status")),
            safety_car_active=payload.get("safety_car_active"),
            virtual_safety_car_active=payload.get("virtual_safety_car_active"),
            weather_state=_string_or_none(payload.get("weather_state")),
        )


@dataclass
class DriverState:
    """Selected driver / car state for the checkpoint.

    Active now:
    - driver code, team, compound, tyre age, current position, stint number
    Derivable now:
    - start position and position change from replay data
    Future placeholders:
    - photo URL, full name, richer car identity, tyre inventory links
    """

    driver_code: Optional[str]
    team_name: Optional[str]
    current_compound: str
    tyre_age_laps: int
    display_name: Optional[str] = None
    driver_photo_url: Optional[str] = None
    car_number: Optional[str] = None
    start_position: Optional[int] = None
    current_position: Optional[int] = None
    position_change: Optional[int] = None
    current_stint_number: Optional[int] = None
    stint_history: list[StintSummary] = field(default_factory=list)

    def validate(self) -> None:
        """Validate the driver state."""
        if not self.current_compound:
            raise ValueError("selected_driver.current_compound is required")
        if self.tyre_age_laps < 0:
            raise ValueError("selected_driver.tyre_age_laps must be >= 0")
        for field_name in ("start_position", "current_position", "current_stint_number"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        for stint in self.stint_history:
            stint.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the driver state."""
        self.validate()
        return {
            "driver_code": self.driver_code,
            "team_name": self.team_name,
            "current_compound": self.current_compound,
            "tyre_age_laps": self.tyre_age_laps,
            "display_name": self.display_name,
            "driver_photo_url": self.driver_photo_url,
            "car_number": self.car_number,
            "start_position": self.start_position,
            "current_position": self.current_position,
            "position_change": self.position_change,
            "current_stint_number": self.current_stint_number,
            "stint_history": [stint.to_dict() for stint in self.stint_history],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DriverState":
        """Deserialize driver state."""
        obj = cls(
            driver_code=_string_or_none(payload.get("driver_code")),
            team_name=_string_or_none(payload.get("team_name")),
            current_compound=str(payload["current_compound"]),
            tyre_age_laps=int(payload["tyre_age_laps"]),
            display_name=_string_or_none(payload.get("display_name")),
            driver_photo_url=_string_or_none(payload.get("driver_photo_url")),
            car_number=_string_or_none(payload.get("car_number")),
            start_position=_int_or_none(payload.get("start_position")),
            current_position=_int_or_none(payload.get("current_position")),
            position_change=_int_or_none(payload.get("position_change")),
            current_stint_number=_int_or_none(payload.get("current_stint_number")),
            stint_history=[
                StintSummary.from_dict(stint_payload)
                for stint_payload in payload.get("stint_history", [])
            ],
        )
        obj.validate()
        return obj


@dataclass
class RaceState:
    """Canonical Phase 3A recommendation checkpoint.

    This is the object future dashboard and replay features should exchange.
    It intentionally supports richer state than the current deterministic engine
    uses today, but it does not change the current recommendation logic.
    """

    state_id: str
    source_type: str
    race_id: str
    selected_driver: DriverState
    circuit_name: str
    laps_remaining: int
    schema_version: str = SCHEMA_VERSION
    source_reference: Optional[str] = None
    season: Optional[int] = None
    event_name: Optional[str] = None
    session_name: Optional[str] = None
    circuit_id: Optional[str] = None
    lap_number: Optional[int] = None
    total_laps: Optional[int] = None
    competitor_context: CompetitorGapContext = field(default_factory=CompetitorGapContext)
    recommendation: Optional[RecommendationState] = None
    event_status: EventStatus = field(default_factory=EventStatus)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate required Phase 3A checkpoint fields."""
        if not self.state_id:
            raise ValueError("state_id is required")
        if not self.source_type:
            raise ValueError("source_type is required")
        if not self.race_id:
            raise ValueError("race_id is required")
        if not self.circuit_name:
            raise ValueError("circuit_name is required")
        if self.laps_remaining < 0:
            raise ValueError("laps_remaining must be >= 0")
        if self.lap_number is None and self.laps_remaining is None:
            raise ValueError("At least one of lap_number or laps_remaining must be provided")
        self.selected_driver.validate()
        self.competitor_context.validate()
        if self.recommendation is not None:
            self.recommendation.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the race state."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "state_id": self.state_id,
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "race_id": self.race_id,
            "season": self.season,
            "event_name": self.event_name,
            "session_name": self.session_name,
            "circuit_name": self.circuit_name,
            "circuit_id": self.circuit_id,
            "lap_number": self.lap_number,
            "laps_remaining": self.laps_remaining,
            "total_laps": self.total_laps,
            "selected_driver": self.selected_driver.to_dict(),
            "competitor_context": self.competitor_context.to_dict(),
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "event_status": self.event_status.to_dict(),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        """Serialize the race state to pretty JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RaceState":
        """Deserialize a race state."""
        obj = cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            state_id=str(payload["state_id"]),
            source_type=str(payload["source_type"]),
            source_reference=_string_or_none(payload.get("source_reference")),
            race_id=str(payload["race_id"]),
            season=_int_or_none(payload.get("season")),
            event_name=_string_or_none(payload.get("event_name")),
            session_name=_string_or_none(payload.get("session_name")),
            circuit_name=str(payload["circuit_name"]),
            circuit_id=_string_or_none(payload.get("circuit_id")),
            lap_number=_int_or_none(payload.get("lap_number")),
            laps_remaining=int(payload["laps_remaining"]),
            total_laps=_int_or_none(payload.get("total_laps")),
            selected_driver=DriverState.from_dict(payload["selected_driver"]),
            competitor_context=CompetitorGapContext.from_dict(payload.get("competitor_context", {})),
            recommendation=(
                RecommendationState.from_dict(payload["recommendation"])
                if payload.get("recommendation")
                else None
            ),
            event_status=EventStatus.from_dict(payload.get("event_status", {})),
            notes=[str(note) for note in payload.get("notes", [])],
        )
        obj.validate()
        return obj


def save_race_states(states: Sequence[RaceState], output_path: Path | str) -> Path:
    """Save a list of race states to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [state.to_dict() for state in states]
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output


def build_stint_history(driver_laps: pd.DataFrame) -> list[StintSummary]:
    """Build stint history from canonical raw replay rows."""
    if driver_laps.empty:
        return []

    ordered = driver_laps.sort_values("lap_number").copy()
    stint_rows: list[StintSummary] = []
    for stint_value, stint_df in ordered.groupby("stint", dropna=True):
        stint_df = stint_df.sort_values("lap_number")
        positions = stint_df["position"].dropna()
        stint_rows.append(
            StintSummary(
                stint_number=int(stint_value),
                compound=_string_or_none(stint_df["compound"].dropna().iloc[0]) if stint_df["compound"].notna().any() else None,
                start_lap=_int_or_none(stint_df["lap_number"].min()),
                end_lap=_int_or_none(stint_df["lap_number"].max()),
                laps_completed=_int_or_none(len(stint_df)),
                opening_position=_int_or_none(positions.iloc[0]) if not positions.empty else None,
                closing_position=_int_or_none(positions.iloc[-1]) if not positions.empty else None,
            )
        )
    return stint_rows


def build_competitor_context_from_order(
    lap_slice: pd.DataFrame,
    current_position: Optional[int],
) -> CompetitorGapContext:
    """Build nearby-car context from same-lap position ordering."""
    if current_position is None:
        return CompetitorGapContext(
            context_note="Current position unavailable, so nearby competitors cannot be derived.",
        )

    ordered = lap_slice.dropna(subset=["position"]).sort_values("position").copy()
    if ordered.empty:
        return CompetitorGapContext(
            context_note="No same-lap position data available for nearby-car derivation.",
        )

    ahead_row = ordered[ordered["position"] == current_position - 1]
    behind_row = ordered[ordered["position"] == current_position + 1]

    return CompetitorGapContext(
        ahead_driver_code=_string_or_none(ahead_row["driver"].iloc[0]) if not ahead_row.empty else None,
        ahead_team_name=_string_or_none(ahead_row["team"].iloc[0]) if not ahead_row.empty else None,
        ahead_position=_int_or_none(ahead_row["position"].iloc[0]) if not ahead_row.empty else None,
        behind_driver_code=_string_or_none(behind_row["driver"].iloc[0]) if not behind_row.empty else None,
        behind_team_name=_string_or_none(behind_row["team"].iloc[0]) if not behind_row.empty else None,
        behind_position=_int_or_none(behind_row["position"].iloc[0]) if not behind_row.empty else None,
        gap_ahead_seconds=None,
        gap_behind_seconds=None,
        context_note=(
            "Nearby identities derived from same-lap order. Gap timing remains a Phase 3 future placeholder."
        ),
    )


def _support_summary_text(
    support_lookup: Optional[Mapping[str, Mapping[str, Any]]],
    next_compound: Optional[str],
    final_compound: Optional[str],
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Build compact support-tier text for the recommendation payload."""
    if support_lookup is None or next_compound is None:
        return None, None, []

    risk_notes: list[str] = []
    next_support = support_lookup.get(next_compound, {})
    final_support = support_lookup.get(final_compound, {}) if final_compound else {}
    tiers = [tier for tier in [next_support.get("support_tier"), final_support.get("support_tier")] if tier]
    reasons = [reason for reason in [next_support.get("support_reason"), final_support.get("support_reason")] if reason]

    for compound, support in ((next_compound, next_support), (final_compound, final_support)):
        if not compound or not support:
            continue
        tier = support.get("support_tier")
        if tier in {"Low", "Moderate"}:
            risk_notes.append(f"{compound} support is {tier.lower()}.")

    support_tier = " / ".join(tiers) if tiers else None
    support_reason = " | ".join(reasons) if reasons else None
    return support_tier, support_reason, risk_notes


def build_recommendation_state(
    plan_payload: Mapping[str, Any],
    source: str,
    lap_number: Optional[int] = None,
    support_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
    confidence_label: Optional[str] = None,
    confidence_reason: Optional[str] = None,
    extra_risk_notes: Optional[Iterable[str]] = None,
    timing_trace: Optional[Mapping[str, Any]] = None,
) -> RecommendationState:
    """Create a canonical recommendation payload from a plan-like mapping."""
    pit_in_laps = _int_or_none(plan_payload.get("pit_lap"))
    second_pit_lap = _int_or_none(plan_payload.get("second_pit_lap"))
    second_pit_in_laps = None
    if pit_in_laps is not None and second_pit_lap is not None:
        second_pit_in_laps = second_pit_lap - pit_in_laps

    support_tier, support_reason, support_risks = _support_summary_text(
        support_lookup=support_lookup,
        next_compound=_string_or_none(plan_payload.get("next_compound")),
        final_compound=_string_or_none(plan_payload.get("final_compound")),
    )

    risk_notes = list(extra_risk_notes or [])
    risk_notes.extend(support_risks)
    timing_window_shape = None
    near_optimal_pit_window: list[int] = []
    if timing_trace:
        timing_window_shape = _string_or_none(timing_trace.get("curve_shape"))
        near_optimal_pit_window = [int(value) for value in timing_trace.get("near_optimal_band_laps", [])]
        if timing_window_shape == "flat" or bool(timing_trace.get("best_on_window_edge")):
            band = timing_trace.get("near_optimal_band_laps", [])
            if band:
                risk_notes.append(
                    f"Flat or edge-bound first-stop timing window: laps {min(band)}-{max(band)} remain near-optimal."
                )

    recommendation = RecommendationState(
        source=source,
        recommended_action="PIT_NOW" if pit_in_laps is not None and pit_in_laps <= 1 else "STAY_OUT",
        strategy_type=_string_or_none(plan_payload.get("strategy_type")),
        pit_in_laps=pit_in_laps,
        pit_on_lap=(lap_number + pit_in_laps) if lap_number is not None and pit_in_laps is not None else None,
        second_pit_in_laps=second_pit_in_laps,
        next_compound=_string_or_none(plan_payload.get("next_compound")),
        final_compound=_string_or_none(plan_payload.get("final_compound")),
        estimated_total_race_time_s=_float_or_none(plan_payload.get("total_race_time")),
        feasible=plan_payload.get("feasible"),
        feasibility_reason=_string_or_none(plan_payload.get("feasibility_reason")),
        support_tier=support_tier,
        support_reason=support_reason,
        confidence_label=_string_or_none(confidence_label),
        confidence_reason=_string_or_none(confidence_reason),
        risk_notes=risk_notes,
        timing_window_shape=timing_window_shape,
        near_optimal_pit_window=near_optimal_pit_window,
    )
    recommendation.validate()
    return recommendation


def _load_json(path: Path | str) -> dict[str, Any]:
    """Load JSON from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_support_lookup(project_root: Path | str = ".") -> dict[str, Mapping[str, Any]]:
    """Load support tiers from the canonical Pre-3 artifact when available."""
    path = Path(project_root) / "data" / "processed" / "pre3_compound_support_summary.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    return payload.get("compound_support", {})


def build_demo_race_state(
    project_root: Path | str,
    current_compound: str = "MEDIUM",
    current_tyre_life: int = 5,
    laps_remaining: int = 25,
) -> RaceState:
    """Build the canonical manual demo checkpoint used for Phase 3A examples."""
    project_root = Path(project_root)
    from src.data.loader import DataLoader
    from src.data.preprocess import detect_pit_stops, select_relevant_columns

    degradation_models, _ = build_role_based_hybrid_model(project_root=project_root)
    pit_raw = DataLoader(project_root=project_root).load_data(dataset="miami_historical")
    pit_source_df = detect_pit_stops(select_relevant_columns(pit_raw))
    pit_loss_value = float(pd.Series(estimate_pit_loss_window(pit_source_df)).median())

    best_plan, _ = recommend_best_strategy(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        candidate_compounds=["SOFT", "MEDIUM", "HARD"],
        include_two_stop=True,
    )
    stability = assess_strategy_stability(
        baseline_plan=best_plan,
        pit_loss_value=pit_loss_value,
        degradation_models=degradation_models,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
    )
    timing_trace = build_strategy_timing_trace(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        next_compound=best_plan.next_compound,
        final_compound=best_plan.final_compound,
    )

    support_lookup = {
        compound: degradation_models.get_support_info(compound)
        for compound in ["SOFT", "MEDIUM", "HARD"]
    }
    recommendation = build_recommendation_state(
        plan_payload=asdict(best_plan),
        source="demo_strategy",
        support_lookup=support_lookup,
        confidence_label=stability.stability_label,
        confidence_reason="Phase 2C stability classification on the canonical demo scenario.",
        extra_risk_notes=stability.flip_conditions,
        timing_trace=timing_trace,
    )

    state = RaceState(
        state_id=f"demo:{_slugify(current_compound)}:tyre_{current_tyre_life}:remaining_{laps_remaining}",
        source_type="demo_scenario",
        source_reference="app/demo_strategy.py default scenario",
        race_id="demo:miami_reference",
        season=2026,
        event_name="Miami reference demo scenario",
        session_name="Race (synthetic)",
        circuit_name="Miami International Autodrome",
        circuit_id="miami_international_autodrome",
        lap_number=None,
        laps_remaining=laps_remaining,
        total_laps=None,
        selected_driver=DriverState(
            driver_code="DEMO",
            display_name="Demo Driver",
            team_name="Reference scenario",
            current_compound=current_compound,
            tyre_age_laps=current_tyre_life,
            current_stint_number=1,
            stint_history=[
                StintSummary(
                    stint_number=1,
                    compound=current_compound,
                    start_lap=None,
                    end_lap=None,
                    laps_completed=current_tyre_life,
                )
            ],
        ),
        competitor_context=CompetitorGapContext(
            context_note="Manual demo state: no nearby competitor ordering is attached yet.",
        ),
        recommendation=recommendation,
        event_status=EventStatus(track_status="1"),
        notes=[
            "Phase 3A manual demo checkpoint anchored to the existing deterministic Miami strategy engine.",
            "Green-flag conditions are assumed because the current demo does not model SC/VSC or weather.",
        ],
    )
    state.validate()
    return state


def race_state_from_phase2d_scenario_result(
    scenario_result: Mapping[str, Any],
    support_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> RaceState:
    """Convert one Phase 2D validation scenario result into a canonical race state."""
    recommendation = build_recommendation_state(
        plan_payload=scenario_result["best_plan"],
        source="phase2d_validation",
        support_lookup=support_lookup,
        confidence_label=_string_or_none(scenario_result.get("stability_label")),
        confidence_reason="Phase 2C stability label captured in the Phase 2D validation artifact.",
        extra_risk_notes=list(scenario_result.get("flip_conditions", [])) + list(scenario_result.get("warnings", [])),
    )

    state = RaceState(
        state_id=f"phase2d:{scenario_result['scenario_id']}",
        source_type="phase2d_validation",
        source_reference="data/processed/phase2d_validation_summary.json",
        race_id=f"phase2d:{scenario_result['scenario_id']}",
        season=None,
        event_name="Miami reference validation scenario",
        session_name="Representative validation",
        circuit_name="Miami International Autodrome (reference)",
        circuit_id="miami_reference",
        lap_number=None,
        laps_remaining=int(scenario_result["laps_remaining"]),
        total_laps=None,
        selected_driver=DriverState(
            driver_code=None,
            team_name=None,
            current_compound=str(scenario_result["current_compound"]),
            tyre_age_laps=int(scenario_result["current_tyre_life"]),
            current_stint_number=1,
            stint_history=[
                StintSummary(
                    stint_number=1,
                    compound=str(scenario_result["current_compound"]),
                    start_lap=None,
                    end_lap=None,
                    laps_completed=int(scenario_result["current_tyre_life"]),
                )
            ],
        ),
        competitor_context=CompetitorGapContext(
            context_note="Synthetic validation scenario: nearby competitors are placeholders until replay ordering is attached.",
        ),
        recommendation=recommendation,
        event_status=EventStatus(track_status=None),
        notes=[str(scenario_result.get("rationale", ""))],
    )
    state.validate()
    return state


def extract_race_states_from_phase2d_artifact(project_root: Path | str = ".") -> list[RaceState]:
    """Extract canonical race states from the Phase 2D validation artifact."""
    project_root = Path(project_root)
    artifact_path = project_root / "data" / "processed" / "phase2d_validation_summary.json"
    if not artifact_path.exists():
        return []
    artifact = _load_json(artifact_path)
    support_lookup = load_support_lookup(project_root)
    return [
        race_state_from_phase2d_scenario_result(result, support_lookup=support_lookup)
        for result in artifact.get("scenarios", [])
    ]


def race_state_from_backtest_checkpoint(
    checkpoint_result: Mapping[str, Any],
    replay_df: pd.DataFrame,
    support_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> RaceState:
    """Convert one held-out backtest checkpoint into a canonical race state."""
    driver_code = str(checkpoint_result["driver"])
    lap_number = int(checkpoint_result["checkpoint_lap"])
    season = int(checkpoint_result["season"])

    event_mask = (
        (replay_df["season"] == season)
        & (replay_df["driver"] == driver_code)
    )
    driver_all_laps = replay_df[event_mask].sort_values("lap_number").copy()
    if driver_all_laps.empty:
        raise ValueError(f"No replay rows found for {driver_code} in season {season}")

    checkpoint_row = driver_all_laps[driver_all_laps["lap_number"] == lap_number]
    if checkpoint_row.empty:
        raise ValueError(f"No replay row found for {driver_code} at lap {lap_number}")
    checkpoint_row = checkpoint_row.iloc[0]

    event_name = str(checkpoint_row["event_name"])
    session_name = str(checkpoint_row["session_name"])
    circuit_name = str(checkpoint_row["circuit_name"])
    driver_event_laps = replay_df[
        (replay_df["season"] == season)
        & (replay_df["event_name"] == event_name)
        & (replay_df["session_name"] == session_name)
        & (replay_df["driver"] == driver_code)
    ].sort_values("lap_number").copy()
    driver_laps_to_checkpoint = driver_event_laps[driver_event_laps["lap_number"] <= lap_number].copy()
    total_laps = _int_or_none(driver_event_laps["lap_number"].max())
    start_position = (
        _int_or_none(driver_event_laps["position"].dropna().iloc[0])
        if driver_event_laps["position"].notna().any()
        else None
    )
    current_position = _int_or_none(checkpoint_row["position"])
    position_change = None
    if start_position is not None and current_position is not None:
        position_change = start_position - current_position

    lap_slice = replay_df[
        (replay_df["season"] == season)
        & (replay_df["event_name"] == event_name)
        & (replay_df["session_name"] == session_name)
        & (replay_df["lap_number"] == lap_number)
    ].copy()
    competitor_context = build_competitor_context_from_order(
        lap_slice=lap_slice,
        current_position=current_position,
    )

    actual_context = checkpoint_result.get("actual_sequence_timing_context", {})
    risk_notes = []
    if checkpoint_result.get("failure_mode"):
        risk_notes.append(
            f"Historical audit context: failure_mode={checkpoint_result['failure_mode']}."
        )
    first_stop_error = checkpoint_result.get("absolute_first_stop_timing_error_laps")
    if first_stop_error is not None:
        risk_notes.append(
            f"Historical audit context: absolute first-stop timing error {first_stop_error} laps."
        )
    if actual_context.get("trace_available") is False:
        risk_notes.append("Historical audit context: actual future sequence was infeasible under the current model.")

    recommendation = build_recommendation_state(
        plan_payload=checkpoint_result["model_best"],
        source="pre3_backtest",
        lap_number=lap_number,
        support_lookup=support_lookup,
        confidence_label=None,
        confidence_reason=None,
        extra_risk_notes=risk_notes,
        timing_trace=checkpoint_result.get("model_best_timing_trace"),
    )

    state = RaceState(
        state_id=f"pre3:{checkpoint_result['checkpoint_id']}",
        source_type="pre3_backtest",
        source_reference="data/processed/pre3_backtest_summary.json",
        race_id=f"{season}:{_slugify(event_name)}:{_slugify(session_name)}",
        season=season,
        event_name=event_name,
        session_name=session_name,
        circuit_name=circuit_name,
        circuit_id=_slugify(circuit_name),
        lap_number=lap_number,
        laps_remaining=int(checkpoint_result["laps_remaining"]),
        total_laps=total_laps,
        selected_driver=DriverState(
            driver_code=driver_code,
            display_name=driver_code,
            team_name=_string_or_none(checkpoint_row["team"]),
            current_compound=str(checkpoint_row["compound"]),
            tyre_age_laps=int(checkpoint_row["tyre_life"]),
            start_position=start_position,
            current_position=current_position,
            position_change=position_change,
            current_stint_number=_int_or_none(checkpoint_row["stint"]),
            stint_history=build_stint_history(driver_laps_to_checkpoint),
        ),
        competitor_context=competitor_context,
        recommendation=recommendation,
        event_status=EventStatus(track_status=_string_or_none(checkpoint_row["track_status"])),
        notes=[
            str(checkpoint_result.get("rationale", "")),
            "Historical replay checkpoint extracted from the canonical held-out Miami backtest artifact.",
        ],
    )
    state.validate()
    return state


def extract_race_states_from_pre3_backtest(project_root: Path | str = ".") -> list[RaceState]:
    """Extract canonical race states from the held-out backtest artifact."""
    project_root = Path(project_root)
    artifact_path = project_root / "data" / "processed" / "pre3_backtest_summary.json"
    replay_path = project_root / "data" / "raw" / "miami_historical" / "combined.parquet"
    if not artifact_path.exists() or not replay_path.exists():
        return []
    artifact = _load_json(artifact_path)
    replay_df = pd.read_parquet(replay_path)
    support_lookup = load_support_lookup(project_root)
    return [
        race_state_from_backtest_checkpoint(
            checkpoint_result=checkpoint,
            replay_df=replay_df,
            support_lookup=support_lookup,
        )
        for checkpoint in artifact.get("checkpoints", [])
    ]
