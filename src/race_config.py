"""Race activation configuration and race-aware strategy context.

This module keeps the existing deterministic strategy engine intact while
making race-specific assumptions explicit. Miami remains the validated baseline;
new race activations can opt into local historical data when it exists and must
otherwise carry a cautious support tier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.preprocess import (
    build_model_df,
    clean_laps,
    detect_pit_stops,
    get_race_group_columns,
    select_relevant_columns,
)
from src.features.hybrid_modeling import build_role_based_hybrid_model


RACE_CONFIG_VERSION = "race-config.v1"
RACE_CONTEXT_VERSION = "race-context.v1"
DEFAULT_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
PIT_WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
PIT_SUPPORT_ORDER = {"Low": 0, "Moderate": 1, "High": 2}


@dataclass(frozen=True)
class BaselineScenario:
    """Small scenario used by race-pack readiness artifacts."""

    scenario_id: str
    current_compound: str
    current_tyre_life: int
    laps_remaining: int
    current_lap: Optional[int] = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RaceConfig:
    """Minimal race-specific settings for one activation target."""

    key: str
    label: str
    season: int
    event_name: str
    circuit_name: str
    circuit_id: str
    total_laps: int
    historical_dataset: str
    historical_years: tuple[int, ...]
    pit_loss_fallback_s: float
    pit_loss_fallback_source: str
    default_current_lap: int
    default_driver_code: str
    default_compound: str = "MEDIUM"
    default_tyre_age: int = 5
    recency_dataset: str = "season_2026_pre_miami"
    compounds: tuple[str, ...] = DEFAULT_COMPOUNDS
    min_pit_loss_samples: int = 8
    min_compound_model_laps: int = 80
    validated_baseline: bool = False
    notes: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    baseline_scenarios: tuple[BaselineScenario, ...] = ()
    config_version: str = RACE_CONFIG_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline_scenarios"] = [
            scenario.to_dict() for scenario in self.baseline_scenarios
        ]
        return payload


RACE_CONFIGS: dict[str, RaceConfig] = {
    "miami_2026": RaceConfig(
        key="miami_2026",
        label="Miami 2026",
        season=2026,
        event_name="Miami Grand Prix",
        circuit_name="Miami International Autodrome",
        circuit_id="miami_international_autodrome",
        total_laps=57,
        historical_dataset="miami_historical",
        historical_years=(2022, 2023, 2024, 2025),
        pit_loss_fallback_s=14.34,
        pit_loss_fallback_source="Phase 2E Miami calibrated fallback",
        default_current_lap=32,
        default_driver_code="DEMO",
        validated_baseline=True,
        notes=(
            "Miami is the current validated circuit anchor.",
            "2026 pre-Miami races are still treated as bounded recency support.",
        ),
        caveats=(
            "The strategy engine remains deterministic, single-car, and not competitor-aware.",
            "Safety-car, VSC, weather, traffic, and tyre inventory effects are not modeled.",
        ),
        baseline_scenarios=(
            BaselineScenario(
                scenario_id="miami_medium_baseline",
                current_compound="MEDIUM",
                current_tyre_life=5,
                current_lap=32,
                laps_remaining=25,
                rationale="Existing canonical demo state preserved for Miami.",
            ),
            BaselineScenario(
                scenario_id="miami_hard_late",
                current_compound="HARD",
                current_tyre_life=18,
                current_lap=43,
                laps_remaining=14,
                rationale="Late hard-stint tactical check for the validated baseline.",
            ),
        ),
    ),
    "canada_2026": RaceConfig(
        key="canada_2026",
        label="Canada 2026",
        season=2026,
        event_name="Canadian Grand Prix",
        circuit_name="Circuit Gilles-Villeneuve",
        circuit_id="circuit_gilles_villeneuve",
        total_laps=70,
        historical_dataset="canada_historical",
        historical_years=(2022, 2023, 2024, 2025),
        pit_loss_fallback_s=20.0,
        pit_loss_fallback_source=(
            "Explicit Canada activation prior; replace with race-local historical "
            "FastF1 data before using for high-confidence calls."
        ),
        default_current_lap=24,
        default_driver_code="CAN",
        default_compound="MEDIUM",
        default_tyre_age=9,
        validated_baseline=False,
        notes=(
            "Canada 2026 is a race-weekend activation target, not a validated replay baseline yet.",
            "Use race-local historical Canada data when available; otherwise the model is a cautious proxy.",
        ),
        caveats=(
            "Canada recommendations should be treated as decision support until race-local support is reviewed for the active compounds.",
            "The degradation model uses a proxy only when Canada historical laps are unavailable.",
            "The manual snapshot mode does not ingest live timing, weather, SC/VSC, or competitor strategy.",
        ),
        baseline_scenarios=(
            BaselineScenario(
                scenario_id="canada_medium_opening",
                current_compound="MEDIUM",
                current_tyre_life=8,
                current_lap=24,
                laps_remaining=46,
                rationale="Mid-opening-stint Canada race-weekend manual snapshot rehearsal.",
            ),
            BaselineScenario(
                scenario_id="canada_hard_midrace",
                current_compound="HARD",
                current_tyre_life=15,
                current_lap=38,
                laps_remaining=32,
                rationale="Midrace hard-stint Canada tactical check.",
            ),
            BaselineScenario(
                scenario_id="canada_soft_late",
                current_compound="SOFT",
                current_tyre_life=5,
                current_lap=56,
                laps_remaining=14,
                rationale="Late soft-stint support and caveat check.",
            ),
        ),
    ),
}

RACE_ALIASES = {
    "miami": "miami_2026",
    "miami_2026": "miami_2026",
    "canada": "canada_2026",
    "canadian": "canada_2026",
    "canada_2026": "canada_2026",
}


@dataclass(frozen=True)
class PitLossEstimate:
    """Race-specific pit-loss estimate plus support metadata."""

    value_s: float
    sample_count: int
    support_tier: str
    source: str
    fallback_used: bool
    method: str
    caveats: tuple[str, ...] = ()
    p25_s: Optional[float] = None
    p75_s: Optional[float] = None
    iqr_s: Optional[float] = None
    uncertainty: str = "Unknown"
    raw_value_s: Optional[float] = None
    raw_sample_count: int = 0
    raw_p25_s: Optional[float] = None
    raw_p75_s: Optional[float] = None
    raw_iqr_s: Optional[float] = None
    strategy_value_s: Optional[float] = None
    strategy_sample_count: int = 0
    strategy_p25_s: Optional[float] = None
    strategy_p75_s: Optional[float] = None
    strategy_iqr_s: Optional[float] = None
    strategy_basis: str = ""
    dropped_sample_count: int = 0
    filter_notes: tuple[str, ...] = ()
    audit_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RaceStrategyContext:
    """Strategy dependencies and audit metadata for one race config."""

    race_config: RaceConfig
    degradation_model: object
    pit_loss: PitLossEstimate
    compound_support: dict[str, dict[str, Any]]
    data_summary: dict[str, Any]
    caveats: list[str] = field(default_factory=list)
    model_config_version: str = RACE_CONTEXT_VERSION

    def to_summary(self) -> dict[str, Any]:
        return {
            "model_config_version": self.model_config_version,
            "race_config": self.race_config.to_dict(),
            "pit_loss": self.pit_loss.to_dict(),
            "compound_support": self.compound_support,
            "data_summary": self.data_summary,
            "caveats": list(self.caveats),
        }


class RaceAwareDegradationModel:
    """Delegating model that overlays race-specific support metadata."""

    def __init__(
        self,
        base_model: object,
        race_config: RaceConfig,
        compound_support: Mapping[str, Mapping[str, Any]],
        model_source: str,
    ) -> None:
        self.base_model = base_model
        self.race_config = race_config
        self.compound_support = {
            compound: dict(payload)
            for compound, payload in compound_support.items()
        }
        self.model_source = model_source

    def predict_lap_time(self, compound: str, tyre_life: int) -> Optional[float]:
        return self.base_model.predict_lap_time(compound, tyre_life)

    def get_model_info(self, compound: str) -> dict[str, Any]:
        info = dict(self.base_model.get_model_info(compound))
        support = self.compound_support.get(compound, {})
        info.update(
            {
                "support_tier": support.get("support_tier", info.get("support_tier")),
                "support_reason": support.get("support_reason", info.get("support_reason")),
                "prediction_health": support.get("prediction_health", info.get("prediction_health")),
                "race_key": self.race_config.key,
                "race_label": self.race_config.label,
                "model_source": self.model_source,
                "anchor_label": self.race_config.label if self.race_config.validated_baseline else "Race-local",
                "recency_label": "2026 recency",
                "race_model_laps": support.get("race_model_laps", 0),
                "race_model_races": support.get("race_model_races", 0),
                "warnings": support.get("warnings", info.get("warnings", [])),
            }
        )
        return info

    def get_support_info(self, compound: str) -> dict[str, Any]:
        return dict(self.compound_support.get(compound, {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "race_key": self.race_config.key,
            "model_source": self.model_source,
            "compound_support": self.compound_support,
        }


def list_race_configs() -> list[RaceConfig]:
    """Return available race configs in UI order."""
    return [RACE_CONFIGS["miami_2026"], RACE_CONFIGS["canada_2026"]]


def normalize_race_key(value: str) -> str:
    """Normalize a user-facing race identifier."""
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in RACE_ALIASES:
        raise KeyError(
            f"Unknown race '{value}'. Available: {', '.join(sorted(RACE_CONFIGS))}"
        )
    return RACE_ALIASES[key]


def get_race_config(value: str) -> RaceConfig:
    """Return a race config by key or alias."""
    return RACE_CONFIGS[normalize_race_key(value)]


def race_pack_dir(project_root: Path | str, race_key: str) -> Path:
    """Return the ignored processed output directory for one race pack."""
    return Path(project_root) / "data" / "processed" / "race_packs" / normalize_race_key(race_key)


def _prepare_model_df(df: pd.DataFrame) -> pd.DataFrame:
    selected = select_relevant_columns(df)
    pit_df = detect_pit_stops(selected)
    clean_df = clean_laps(pit_df)
    return build_model_df(clean_df)


def _load_historical_dataset(
    project_root: Path | str,
    race_config: RaceConfig,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    dataset_path = Path(project_root) / "data" / "raw" / race_config.historical_dataset
    if not dataset_path.exists():
        return (
            None,
            (
                f"Raw data group not found: {dataset_path}. "
                "Generate it with scripts/build_race_pack.py --fetch-historical "
                "or provide matching Parquet files locally."
            ),
        )
    try:
        df = DataLoader(project_root=project_root).load_data(
            dataset=race_config.historical_dataset,
            fallback=False,
        )
    except Exception as exc:
        return None, str(exc)
    if df.empty:
        return None, f"Dataset {race_config.historical_dataset} is empty"
    return df, None


def _unique_count(df: pd.DataFrame, cols: Sequence[str]) -> int:
    available = [col for col in cols if col in df.columns]
    if not available or df.empty:
        return 0
    return int(df[available].drop_duplicates().shape[0])


def _support_tier_from_laps(
    race_model_laps: int,
    race_model_races: int,
    prediction_valid: bool,
    min_compound_model_laps: int,
) -> str:
    if not prediction_valid:
        return "Low"
    if race_model_laps >= max(300, min_compound_model_laps * 3) and race_model_races >= 3:
        return "High"
    if race_model_laps >= min_compound_model_laps and race_model_races >= 2:
        return "Moderate"
    return "Low"


def _prediction_valid(model: object, compound: str) -> bool:
    probes = [model.predict_lap_time(compound, tyre_life) for tyre_life in (1, 5, 10)]
    return all(value is not None and np.isfinite(value) for value in probes)


def _build_compound_support(
    race_config: RaceConfig,
    base_model: object,
    historical_df: Optional[pd.DataFrame],
    historical_error: Optional[str],
    model_source: str,
) -> dict[str, dict[str, Any]]:
    if historical_df is not None and not historical_df.empty:
        model_df = _prepare_model_df(historical_df)
    else:
        model_df = pd.DataFrame()

    support: dict[str, dict[str, Any]] = {}
    for compound in race_config.compounds:
        base_support = (
            base_model.get_support_info(compound)
            if hasattr(base_model, "get_support_info")
            else {}
        )
        base_info = base_model.get_model_info(compound)
        race_raw = historical_df[historical_df["Compound"] == compound] if historical_df is not None else pd.DataFrame()
        race_model = model_df[model_df["Compound"] == compound] if not model_df.empty else pd.DataFrame()
        race_model_laps = int(len(race_model))
        race_model_races = _unique_count(race_model, ["season", "event_name", "session_name"])
        race_model_stints = _unique_count(race_model, ["season", "event_name", "session_name", "Driver", "Stint"])
        prediction_valid = _prediction_valid(base_model, compound)

        warnings: list[str] = []
        if model_source == "proxy_miami_anchor":
            support_tier = "Low"
            support_reason = (
                f"Low race-specific support: {race_config.label} has no local "
                f"{race_config.historical_dataset} model data loaded. Predictions use the existing "
                "Miami-anchor/2026-recency stack as a cautious proxy."
            )
            warnings.append("race-local-data-missing")
            if historical_error:
                warnings.append("historical-dataset-unavailable")
        else:
            support_tier = _support_tier_from_laps(
                race_model_laps=race_model_laps,
                race_model_races=race_model_races,
                prediction_valid=prediction_valid,
                min_compound_model_laps=race_config.min_compound_model_laps,
            )
            support_reason = (
                f"{support_tier} race-specific support for {race_config.label}: "
                f"{race_model_laps} model laps, {race_model_stints} model stints, "
                f"{race_model_races} race(s) for {compound}."
            )
            if support_tier in {"Low", "Moderate"}:
                warnings.append(f"{compound.lower()}-race-support-{support_tier.lower()}")

        if compound == "SOFT":
            warnings.append("soft-requires-caution")

        support[compound] = {
            "compound": compound,
            "support_tier": support_tier,
            "support_reason": support_reason,
            "prediction_health": "valid" if prediction_valid else "fragile",
            "model_source": model_source,
            "race_key": race_config.key,
            "race_label": race_config.label,
            "race_raw_laps": int(len(race_raw)),
            "race_model_laps": race_model_laps,
            "race_model_stints": race_model_stints,
            "race_model_races": race_model_races,
            "base_support_tier": base_support.get("support_tier"),
            "base_support_reason": (
                base_support.get("support_reason")
                if race_config.validated_baseline
                else "Underlying role-based model was rebuilt with this race as the local anchor."
            ),
            "base_model_type": base_info.get("model_type"),
            "base_model_laps": base_info.get("samples"),
            "warnings": warnings,
        }
    return support


def _pit_support_tier(sample_count: int, min_samples: int) -> str:
    if sample_count >= max(min_samples, 12):
        return "High"
    if sample_count >= max(3, min_samples // 2):
        return "Moderate"
    if sample_count > 0:
        return "Low"
    return "Low"


def _pit_spread_tier(iqr_s: Optional[float]) -> str:
    """Classify pit-loss support by dispersion, not just sample count."""
    if iqr_s is None or not np.isfinite(iqr_s):
        return "Low"
    if iqr_s <= 8.0:
        return "High"
    if iqr_s <= 15.0:
        return "Moderate"
    return "Low"


def _min_support_tier(*tiers: str) -> str:
    return min(tiers, key=lambda tier: PIT_SUPPORT_ORDER.get(tier, 0))


def _quantile_summary(values: Sequence[float]) -> dict[str, Optional[float]]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return {"median": None, "p25": None, "p75": None, "iqr": None}
    p25 = float(np.percentile(array, 25))
    p75 = float(np.percentile(array, 75))
    return {
        "median": float(np.median(array)),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
    }


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _is_false_like(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return bool(value) is False


def _is_valid_reference_lap(row: pd.Series) -> bool:
    lap_time = _as_float(row.get("LapTime"))
    if lap_time is None:
        return False
    if "Deleted" in row and not _is_false_like(row.get("Deleted")):
        return False
    if "TrackStatus" in row and pd.notna(row.get("TrackStatus")):
        if str(row.get("TrackStatus")) != "1":
            return False
    return True


def _compound_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _status_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _csv_list(values: Sequence[Any]) -> str:
    return ";".join("" if pd.isna(value) else str(value) for value in values)


def _collect_reference_laps(
    driver_df: pd.DataFrame,
    start_idx: int,
    step: int,
    limit: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = start_idx
    while idx in driver_df.index and len(rows) < limit:
        row = driver_df.loc[idx]
        if _is_valid_reference_lap(row):
            rows.append(
                {
                    "lap": row.get("LapNumber"),
                    "lap_time": _as_float(row.get("LapTime")),
                    "track_status": _status_value(row.get("TrackStatus")),
                }
            )
        idx += step
    return rows


def build_pit_loss_audit_frame(historical_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-stop audit rows for the effective pit-loss calculation.

    The raw estimate mirrors the legacy optimizer input: the stint-change lap
    plus the following lap are compared with nearby green reference laps. This
    is an effective race-loss proxy, not a pure pit-lane transit measurement.
    The legacy reference search starts immediately after the stint-change lap,
    so the following lap can be both actual-window and baseline context.
    """
    if historical_df is None or historical_df.empty:
        return pd.DataFrame()

    pit_source_df = detect_pit_stops(select_relevant_columns(historical_df))
    group_cols = get_race_group_columns(pit_source_df, include_driver=True)
    rows: list[dict[str, Any]] = []

    for _, driver_group in pit_source_df.groupby(group_cols, sort=False, dropna=False):
        driver_df = driver_group.sort_values("LapNumber").reset_index(drop=True)
        pit_indices = driver_df.index[driver_df["PitStop"] == 1].tolist()
        for idx in pit_indices:
            row = driver_df.loc[idx]
            previous_row = driver_df.loc[idx - 1] if idx - 1 in driver_df.index else None
            next_row = driver_df.loc[idx + 1] if idx + 1 in driver_df.index else None
            pit_window_indices = [idx]
            if next_row is not None:
                pit_window_indices.append(idx + 1)

            pit_in_time = _as_float(previous_row.get("PitInTime")) if previous_row is not None else None
            pit_out_time = _as_float(row.get("PitOutTime"))
            pit_lane_time = (
                pit_out_time - pit_in_time
                if pit_in_time is not None and pit_out_time is not None
                else np.nan
            )

            pit_lap_times = []
            pit_statuses = []
            pit_compounds = []
            for pit_idx in pit_window_indices:
                pit_row = driver_df.loc[pit_idx]
                lap_time = _as_float(pit_row.get("LapTime"))
                if lap_time is not None and (
                    "Deleted" not in pit_row or _is_false_like(pit_row.get("Deleted"))
                ):
                    pit_lap_times.append(lap_time)
                pit_statuses.append(_status_value(pit_row.get("TrackStatus")))
                pit_compounds.append(_compound_name(pit_row.get("Compound")))

            before_refs = _collect_reference_laps(driver_df, idx - 1, -1)
            after_refs = _collect_reference_laps(driver_df, idx + 1, 1)
            baseline_values = [
                ref["lap_time"]
                for ref in before_refs + after_refs
                if ref["lap_time"] is not None
            ]
            baseline_reference = float(np.mean(baseline_values)) if baseline_values else np.nan

            actual_time = float(np.sum(pit_lap_times)) if pit_lap_times else np.nan
            expected_time = baseline_reference * len(pit_lap_times) if pit_lap_times and np.isfinite(baseline_reference) else np.nan
            pit_loss = actual_time - expected_time if np.isfinite(actual_time) and np.isfinite(expected_time) else np.nan

            compound_before = _compound_name(previous_row.get("Compound")) if previous_row is not None else ""
            compound_after = _compound_name(row.get("Compound"))
            wet_transition = bool(
                {compound_before, compound_after, *pit_compounds} & PIT_WET_COMPOUNDS
            )
            known_pit_statuses = [status for status in pit_statuses if status]
            non_green_pit_event = any(status != "1" for status in known_pit_statuses)
            raw_sample = bool(np.isfinite(pit_loss) and pit_loss > 0)
            direct_pit_lane_sample = bool(np.isfinite(pit_lane_time) and pit_lane_time > 0)
            notes = []
            if not raw_sample:
                notes.append("non-positive-or-missing-loss")
            if not direct_pit_lane_sample:
                notes.append("missing-pit-in-out-timestamp-pair")
            if len(baseline_values) < 4:
                notes.append("thin-reference-baseline")
            if non_green_pit_event:
                notes.append("non-green-pit-window")
            if wet_transition:
                notes.append("wet-or-intermediate-transition")

            nearby_index = [label for label in range(idx - 2, idx + 4) if label in driver_df.index]
            nearby_rows = driver_df.loc[nearby_index]
            rows.append(
                {
                    "year": row.get("season"),
                    "event_name": row.get("event_name"),
                    "session_name": row.get("session_name"),
                    "driver": row.get("Driver"),
                    "lap": row.get("LapNumber"),
                    "stint_before": previous_row.get("Stint") if previous_row is not None else None,
                    "stint_after": row.get("Stint"),
                    "compound_before": compound_before,
                    "compound_after": compound_after,
                    "pit_loss_estimate": pit_loss if np.isfinite(pit_loss) else np.nan,
                    "baseline_reference_lap_time": baseline_reference,
                    "baseline_lap_count": len(baseline_values),
                    "pit_window_lap_count": len(pit_lap_times),
                    "pit_window_actual_time": actual_time,
                    "pit_window_expected_time": expected_time,
                    "pit_in_time": pit_in_time,
                    "pit_out_time": pit_out_time,
                    "pit_lane_time_s": pit_lane_time if np.isfinite(pit_lane_time) else np.nan,
                    "prev_reference_laps": _csv_list([ref["lap"] for ref in before_refs]),
                    "prev_reference_lap_times": _csv_list([ref["lap_time"] for ref in before_refs]),
                    "next_reference_laps": _csv_list([ref["lap"] for ref in after_refs]),
                    "next_reference_lap_times": _csv_list([ref["lap_time"] for ref in after_refs]),
                    "nearby_laps": _csv_list(nearby_rows.get("LapNumber", pd.Series(dtype=object)).tolist()),
                    "nearby_lap_times": _csv_list(nearby_rows.get("LapTime", pd.Series(dtype=object)).tolist()),
                    "nearby_compounds": _csv_list(nearby_rows.get("Compound", pd.Series(dtype=object)).tolist()),
                    "nearby_track_status": _csv_list(nearby_rows.get("TrackStatus", pd.Series(dtype=object)).tolist()),
                    "pit_window_track_status": _csv_list(pit_statuses),
                    "wet_intermediate_transition": wet_transition,
                    "non_green_pit_event": non_green_pit_event,
                    "raw_sample": raw_sample,
                    "direct_pit_lane_sample": direct_pit_lane_sample,
                    "clean_candidate": False,
                    "strategy_sample": False,
                    "strategy_metric": "",
                    "filter_notes": ";".join(notes),
                }
            )

    return pd.DataFrame(rows)


def _annotate_pit_loss_filters(
    audit_df: pd.DataFrame,
    min_samples: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark cleaner strategy samples and summarize raw/filtered distributions."""
    if audit_df.empty or "pit_loss_estimate" not in audit_df.columns:
        return audit_df, {
            "raw": _quantile_summary([]),
            "strategy": _quantile_summary([]),
            "notes": ("No pit-loss audit rows were available.",),
        }

    out = audit_df.copy()
    raw_mask = out["raw_sample"].astype(bool)
    raw_losses = pd.to_numeric(out.loc[raw_mask, "pit_loss_estimate"], errors="coerce").dropna()
    notes: list[str] = [
        "Raw samples measure effective stint-change loss versus nearby green laps, not pure pit-lane transit time.",
        "The legacy raw baseline can include the lap after the stint change, so the audit should be treated as diagnostic rather than a stable pit-loss measurement.",
    ]

    has_status = out["pit_window_track_status"].fillna("").astype(str).str.len().gt(0).any()
    if has_status:
        notes.append("Clean candidates exclude pit-window laps with non-green TrackStatus values.")
    else:
        notes.append("TrackStatus was unavailable; filtering used baseline quality and robust outlier checks only.")

    effective_candidate_mask = (
        raw_mask
        & (out["wet_intermediate_transition"] == False)
        & (out["baseline_lap_count"] >= 4)
        & (out["pit_window_lap_count"] >= 2)
    )
    if has_status:
        effective_candidate_mask = effective_candidate_mask & (out["non_green_pit_event"] == False)

    if int(effective_candidate_mask.sum()) < max(3, min_samples // 2):
        notes.append("Strict four-reference baseline filter was sparse; relaxed to at least two reference laps.")
        effective_candidate_mask = (
            raw_mask
            & (out["wet_intermediate_transition"] == False)
            & (out["baseline_lap_count"] >= 2)
            & (out["pit_window_lap_count"] >= 1)
        )
        if has_status:
            effective_candidate_mask = effective_candidate_mask & (out["non_green_pit_event"] == False)

    direct_mask = (
        out.get("direct_pit_lane_sample", pd.Series(False, index=out.index)).astype(bool)
        & (out["wet_intermediate_transition"] == False)
    )
    if has_status:
        direct_mask = direct_mask & (out["non_green_pit_event"] == False)

    def apply_iqr_mask(
        base_mask: pd.Series,
        value_column: str,
        label: str,
    ) -> tuple[pd.Series, Optional[float], Optional[float], list[str]]:
        base_values = pd.to_numeric(out.loc[base_mask, value_column], errors="coerce").dropna()
        local_notes: list[str] = []
        strategy_mask = base_mask.copy()
        lower = None
        upper = None
        if len(base_values) >= 4:
            q1 = float(np.percentile(base_values, 25))
            q3 = float(np.percentile(base_values, 75))
            iqr = q3 - q1
            lower = max(0.0, q1 - 1.5 * iqr)
            upper = q3 + 1.5 * iqr
            strategy_mask = base_mask & (
                pd.to_numeric(out[value_column], errors="coerce").between(lower, upper)
            )
            local_notes.append(
                f"{label} IQR filter retained samples in [{lower:.1f}s, {upper:.1f}s]."
            )
        else:
            local_notes.append(f"Too few {label} candidates for IQR filtering; using all candidates.")
        if int(strategy_mask.sum()) < 3 and int(base_mask.sum()) > 0:
            local_notes.append(f"{label} filtered sample count fell below 3; using candidates with high uncertainty.")
            strategy_mask = base_mask.copy()
        return strategy_mask, lower, upper, local_notes

    direct_strategy_mask, lower_bound, upper_bound, direct_notes = apply_iqr_mask(
        direct_mask,
        "pit_lane_time_s",
        "Direct pit-in/out transit",
    )
    effective_strategy_mask, effective_lower, effective_upper, effective_notes = apply_iqr_mask(
        effective_candidate_mask,
        "pit_loss_estimate",
        "Effective lap-window",
    )

    direct_strategy_count = int(direct_strategy_mask.sum())
    if direct_strategy_count >= max(3, min_samples // 2):
        candidate_mask = direct_mask
        strategy_mask = direct_strategy_mask
        strategy_column = "pit_lane_time_s"
        strategy_basis = "filtered_direct_pit_in_out_transit"
        notes.append(
            "Strategy estimate uses direct PitInTime/PitOutTime transit samples because lap-window effective samples are unstable."
        )
        notes.extend(direct_notes)
        out.loc[candidate_mask, "clean_candidate"] = True
        out.loc[strategy_mask, "strategy_metric"] = "pit_lane_time_s"
        out.loc[candidate_mask & ~strategy_mask, "filter_notes"] = (
            out.loc[candidate_mask & ~strategy_mask, "filter_notes"].astype(str).str.strip(";")
            + ";direct-pit-lane-iqr-outlier"
        ).str.strip(";")
    else:
        candidate_mask = effective_candidate_mask
        strategy_mask = effective_strategy_mask
        strategy_column = "pit_loss_estimate"
        strategy_basis = "filtered_effective_lap_window"
        lower_bound = effective_lower
        upper_bound = effective_upper
        notes.append(
            "Direct PitInTime/PitOutTime samples were too sparse; strategy estimate uses filtered effective lap-window samples."
        )
        notes.extend(effective_notes)
        out.loc[candidate_mask, "clean_candidate"] = True
        out.loc[strategy_mask, "strategy_metric"] = "pit_loss_estimate"
        out.loc[candidate_mask & ~strategy_mask, "filter_notes"] = (
            out.loc[candidate_mask & ~strategy_mask, "filter_notes"].astype(str).str.strip(";")
            + ";effective-lap-window-iqr-outlier"
        ).str.strip(";")

    out.loc[strategy_mask, "strategy_sample"] = True
    out.loc[raw_mask & ~candidate_mask, "filter_notes"] = out.loc[
        raw_mask & ~candidate_mask, "filter_notes"
    ].astype(str).str.strip(";")

    strategy_losses = pd.to_numeric(out.loc[strategy_mask, strategy_column], errors="coerce").dropna()
    summary = {
        "raw": _quantile_summary(raw_losses.tolist()),
        "strategy": _quantile_summary(strategy_losses.tolist()),
        "raw_sample_count": int(len(raw_losses)),
        "candidate_sample_count": int(candidate_mask.sum()),
        "strategy_sample_count": int(len(strategy_losses)),
        "dropped_sample_count": int(len(raw_losses) - len(strategy_losses)),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "strategy_basis": strategy_basis,
        "strategy_column": strategy_column,
        "direct_candidate_sample_count": int(direct_mask.sum()),
        "direct_strategy_sample_count": direct_strategy_count,
        "notes": tuple(notes),
    }
    return out, summary


def _pit_uncertainty(
    raw_iqr_s: Optional[float],
    strategy_iqr_s: Optional[float],
    raw_count: int,
    strategy_count: int,
    dropped_count: int,
    min_samples: int,
    strategy_basis: str,
) -> str:
    if strategy_count < max(3, min_samples // 2):
        return "High"
    if strategy_iqr_s is None or not np.isfinite(strategy_iqr_s) or strategy_iqr_s > 15.0:
        return "High"
    drop_fraction = dropped_count / raw_count if raw_count else 1.0
    if strategy_basis == "filtered_direct_pit_in_out_transit":
        if strategy_iqr_s > 8.0:
            return "Moderate"
        if raw_iqr_s is not None and raw_iqr_s > 20.0:
            return "Moderate"
        return "Low"
    if raw_iqr_s is not None and raw_iqr_s > 40.0 and drop_fraction > 0.45:
        return "High"
    if strategy_iqr_s > 8.0 or (raw_iqr_s is not None and raw_iqr_s > 20.0) or drop_fraction > 0.35:
        return "Moderate"
    return "Low"


def estimate_race_pit_loss(
    race_config: RaceConfig,
    historical_df: Optional[pd.DataFrame],
    historical_error: Optional[str] = None,
    audit_output_path: Optional[Path] = None,
) -> PitLossEstimate:
    """Estimate pit loss from race-local data when possible."""
    caveats: list[str] = []
    if historical_df is not None and not historical_df.empty:
        audit_df = build_pit_loss_audit_frame(historical_df)
        audit_df, audit_summary = _annotate_pit_loss_filters(
            audit_df,
            min_samples=race_config.min_pit_loss_samples,
        )
        audit_path_str = None
        if audit_output_path is not None:
            audit_output_path.parent.mkdir(parents=True, exist_ok=True)
            audit_df.to_csv(audit_output_path, index=False)
            audit_path_str = str(audit_output_path)

        raw_summary = audit_summary["raw"]
        strategy_summary = audit_summary["strategy"]
        raw_count = int(audit_summary.get("raw_sample_count", 0))
        strategy_count = int(audit_summary.get("strategy_sample_count", 0))

        if raw_count > 0:
            if race_config.validated_baseline:
                pit_value = float(raw_summary["median"])
                sample_count = raw_count
                p25_s = raw_summary["p25"]
                p75_s = raw_summary["p75"]
                iqr_s = raw_summary["iqr"]
                support_tier = _min_support_tier(
                    _pit_support_tier(sample_count, race_config.min_pit_loss_samples),
                    _pit_spread_tier(iqr_s),
                )
                uncertainty = "Moderate" if support_tier != "High" else "Low"
                method = "race-local median effective pit-window estimate"
                caveats.append(
                    "Pit loss is an effective stint-change estimate versus nearby green laps, not a pure pit-lane transit measurement."
                )
            elif strategy_count > 0 and strategy_summary["median"] is not None:
                pit_value = float(strategy_summary["median"])
                sample_count = strategy_count
                p25_s = strategy_summary["p25"]
                p75_s = strategy_summary["p75"]
                iqr_s = strategy_summary["iqr"]
                strategy_basis = str(audit_summary.get("strategy_basis", ""))
                uncertainty = _pit_uncertainty(
                    raw_iqr_s=raw_summary["iqr"],
                    strategy_iqr_s=iqr_s,
                    raw_count=raw_count,
                    strategy_count=strategy_count,
                    dropped_count=int(audit_summary.get("dropped_sample_count", 0)),
                    min_samples=race_config.min_pit_loss_samples,
                    strategy_basis=strategy_basis,
                )
                support_tier = _min_support_tier(
                    _pit_support_tier(sample_count, race_config.min_pit_loss_samples),
                    _pit_spread_tier(iqr_s),
                )
                if uncertainty == "High" and support_tier == "High":
                    support_tier = "Moderate"
                if strategy_basis == "filtered_direct_pit_in_out_transit":
                    method = "race-local filtered direct PitInTime/PitOutTime transit proxy"
                    caveats.append(
                        "Strategy pit loss uses direct pit-in/out transit timestamps because the effective lap-window samples are unstable; no sector-level racing-line subtraction is available."
                    )
                    caveats.append(
                        "Raw pit-loss distribution still reports the legacy effective stint-change estimate versus nearby green laps."
                    )
                else:
                    method = "race-local filtered effective pit-window strategy estimate"
                    caveats.append(
                        "Pit loss is an effective stint-change estimate versus nearby green laps, not a pure pit-lane transit measurement."
                    )
                if raw_summary["median"] is not None and abs(float(raw_summary["median"]) - pit_value) >= 5.0:
                    caveats.append(
                        f"Raw effective median was {float(raw_summary['median']):.1f}s; strategy estimate uses filtered dry/green/robust samples at {pit_value:.1f}s."
                    )
            else:
                pit_value = float(raw_summary["median"])
                sample_count = raw_count
                p25_s = raw_summary["p25"]
                p75_s = raw_summary["p75"]
                iqr_s = raw_summary["iqr"]
                support_tier = "Low"
                uncertainty = "High"
                method = "race-local raw effective pit-window estimate"
                caveats.append(
                    "No defensible filtered pit-loss subset was available; raw effective samples are used with high uncertainty."
                )

            if raw_summary["iqr"] is not None and raw_summary["iqr"] > 20.0:
                caveats.append(
                    f"Raw pit-loss samples are widely dispersed (IQR {float(raw_summary['iqr']):.1f}s; p25 {float(raw_summary['p25']):.1f}s, p75 {float(raw_summary['p75']):.1f}s)."
                )
            if support_tier != "High" or uncertainty != "Low":
                caveats.append(
                    f"Pit-loss support is {support_tier.lower()} with {uncertainty.lower()} uncertainty."
                )
            return PitLossEstimate(
                value_s=pit_value,
                sample_count=sample_count,
                support_tier=support_tier,
                source=race_config.historical_dataset,
                fallback_used=False,
                method=method,
                p25_s=p25_s,
                p75_s=p75_s,
                iqr_s=iqr_s,
                uncertainty=uncertainty,
                raw_value_s=raw_summary["median"],
                raw_sample_count=raw_count,
                raw_p25_s=raw_summary["p25"],
                raw_p75_s=raw_summary["p75"],
                raw_iqr_s=raw_summary["iqr"],
                strategy_value_s=pit_value,
                strategy_sample_count=sample_count,
                strategy_p25_s=p25_s,
                strategy_p75_s=p75_s,
                strategy_iqr_s=iqr_s,
                strategy_basis=str(audit_summary.get("strategy_basis", "")),
                dropped_sample_count=int(audit_summary.get("dropped_sample_count", 0)),
                filter_notes=tuple(audit_summary.get("notes", ())),
                audit_path=audit_path_str,
                caveats=tuple(caveats),
            )
        caveats.append(
            f"No valid race-local pit-loss samples were produced from {race_config.historical_dataset}."
        )

    if historical_error:
        caveats.append(
            f"Race-local historical data unavailable: {historical_error}"
        )
    fallback_source = race_config.pit_loss_fallback_source.rstrip(".")
    caveats.append(
        f"Using fallback pit loss {race_config.pit_loss_fallback_s:.2f}s from {fallback_source}."
    )
    return PitLossEstimate(
        value_s=float(race_config.pit_loss_fallback_s),
        sample_count=0,
        support_tier="Low",
        source=race_config.pit_loss_fallback_source,
        fallback_used=True,
        method="explicit race-config fallback",
        uncertainty="High",
        strategy_value_s=float(race_config.pit_loss_fallback_s),
        caveats=tuple(caveats),
    )


def build_race_strategy_context(
    project_root: Path | str = ".",
    race_key: str = "miami_2026",
) -> RaceStrategyContext:
    """Build a race-aware strategy context around the existing engine."""
    project_root = Path(project_root)
    race_config = get_race_config(race_key)
    historical_df, historical_error = _load_historical_dataset(project_root, race_config)

    use_local_anchor = historical_df is not None and not historical_df.empty
    if race_config.validated_baseline:
        use_local_anchor = False

    base_model, _ = build_role_based_hybrid_model(
        project_root=project_root,
        miami_raw_override=historical_df if use_local_anchor else None,
    )
    model_source = "race_local_historical" if use_local_anchor else "proxy_miami_anchor"
    if race_config.validated_baseline:
        model_source = "validated_miami_baseline"

    compound_support = _build_compound_support(
        race_config=race_config,
        base_model=base_model,
        historical_df=historical_df,
        historical_error=historical_error,
        model_source=model_source,
    )
    race_model = RaceAwareDegradationModel(
        base_model=base_model,
        race_config=race_config,
        compound_support=compound_support,
        model_source=model_source,
    )
    pit_loss = estimate_race_pit_loss(
        race_config=race_config,
        historical_df=historical_df,
        historical_error=historical_error,
        audit_output_path=race_pack_dir(project_root, race_config.key) / "pit_loss_audit.csv",
    )

    data_summary = {
        "historical_dataset": race_config.historical_dataset,
        "historical_dataset_available": historical_df is not None and not historical_df.empty,
        "historical_error": historical_error,
        "historical_raw_laps": int(len(historical_df)) if historical_df is not None else 0,
        "historical_years_configured": list(race_config.historical_years),
        "model_source": model_source,
    }
    caveats = list(race_config.caveats)
    caveats.extend(pit_loss.caveats)
    if model_source == "proxy_miami_anchor":
        caveats.append(
            f"{race_config.label} is not using a race-local degradation anchor yet."
        )

    return RaceStrategyContext(
        race_config=race_config,
        degradation_model=race_model,
        pit_loss=pit_loss,
        compound_support=compound_support,
        data_summary=data_summary,
        caveats=caveats,
    )


def confidence_label_for_context(
    stability_label: Optional[str],
    race_context: RaceStrategyContext,
    future_compounds: Sequence[str],
) -> str:
    """Combine Phase 2C stability with race-specific support caution."""
    base_label = stability_label or "Pending"
    support_tiers = [
        race_context.compound_support.get(compound, {}).get("support_tier", "Low")
        for compound in future_compounds
        if compound
    ]
    cautious = (
        race_context.pit_loss.support_tier == "Low"
        or race_context.data_summary.get("model_source") == "proxy_miami_anchor"
        or any(tier == "Low" for tier in support_tiers)
    )
    if cautious:
        return f"Cautious / {base_label}"
    return base_label


def recommendation_pit_window(recommendation: Any) -> Optional[dict[str, int]]:
    """Return a compact pit-window payload from a recommendation object."""
    if recommendation is None:
        return None
    window = list(getattr(recommendation, "near_optimal_pit_window", []) or [])
    if not window and getattr(recommendation, "pit_in_laps", None) is not None:
        window = [int(recommendation.pit_in_laps)]
    if not window:
        return None
    return {"start_in_laps": int(min(window)), "end_in_laps": int(max(window))}


def append_recommendation_ledger_entry(
    project_root: Path | str,
    race_context: RaceStrategyContext,
    race_state: Any,
    snapshot_meta: Mapping[str, Any],
) -> Path:
    """Append one optional local recommendation ledger entry."""
    output_dir = Path(project_root) / "data" / "processed" / "recommendation_ledger"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{race_context.race_config.key}.jsonl"

    recommendation = race_state.recommendation
    driver = race_state.selected_driver
    caveats = []
    if recommendation is not None:
        caveats.extend(recommendation.risk_notes or [])
    caveats.extend(race_state.notes or [])
    caveats.extend(race_context.caveats)

    entry = {
        "generated_at": datetime.utcnow().isoformat(),
        "ledger_version": "recommendation-ledger.v1",
        "race": race_context.race_config.label,
        "race_key": race_context.race_config.key,
        "source_mode": snapshot_meta.get("source_mode"),
        "lap": race_state.lap_number,
        "driver": driver.driver_code or driver.display_name,
        "tyre": driver.current_compound,
        "tyre_age": driver.tyre_age_laps,
        "position": driver.current_position,
        "recommendation": recommendation.recommended_action if recommendation else None,
        "strategy_type": recommendation.strategy_type if recommendation else None,
        "pit_window": recommendation_pit_window(recommendation),
        "pit_in_laps": recommendation.pit_in_laps if recommendation else None,
        "next_compound": recommendation.next_compound if recommendation else None,
        "final_compound": recommendation.final_compound if recommendation else None,
        "confidence": recommendation.confidence_label if recommendation else None,
        "support_tier": recommendation.support_tier if recommendation else None,
        "pit_loss": race_context.pit_loss.to_dict(),
        "caveats": sorted({str(item) for item in caveats if item}),
        "model_config_version": race_context.model_config_version,
        "race_config_version": race_context.race_config.config_version,
    }

    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return output_path
