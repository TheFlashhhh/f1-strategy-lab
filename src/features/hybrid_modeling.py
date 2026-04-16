"""Phase 2B / Pre-3 hybrid modeling helpers.

This module now treats Miami historical data as the circuit anchor and uses
current-season 2026 races only as a bounded recency adjustment at prediction
time. It no longer presents a replicated cross-circuit row pool as Miami truth.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.evaluate_degradation import DegradationEvaluationResult, evaluate_all_degradation

logger = logging.getLogger(__name__)


@dataclass
class DataPoolMetadata:
    """Metadata and configuration for a data pool."""

    pool_id: str
    name: str
    years: List[int]
    circuits: List[str]
    regulation_era: str
    recency_weight: float
    circuit_role: str
    target_race_context: str
    description: str
    sample_count: int = 0
    compound_samples: Dict[str, int] = field(default_factory=dict)
    excluded_reason: Optional[str] = None


@dataclass
class HybridModelingContext:
    """Configuration and runtime summary for the role-based hybrid stack."""

    pools_config: List[DataPoolMetadata]
    active_pools: List[DataPoolMetadata] = field(default_factory=list)
    blended_data: Optional[pd.DataFrame] = None
    weighting_scheme: str = "role_based_adjustment"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "weighting_scheme": self.weighting_scheme,
            "active_pools": [
                {
                    "pool_id": pool.pool_id,
                    "name": pool.name,
                    "years": pool.years,
                    "circuits": pool.circuits,
                    "recency_weight": pool.recency_weight,
                    "circuit_role": pool.circuit_role,
                    "sample_count": pool.sample_count,
                    "compound_samples": pool.compound_samples,
                }
                for pool in self.active_pools
            ],
            "total_samples": sum(pool.sample_count for pool in self.active_pools),
            "notes": self.notes,
        }


@dataclass
class CompoundPoolSupport:
    """Support metrics for one compound inside one data pool."""

    raw_laps: int
    model_laps: int
    raw_stints: int
    model_stints: int
    raw_races: int
    model_races: int
    model_type: Optional[str]
    prediction_valid: bool


@dataclass
class CompoundSupportSummary:
    """Combined support summary for one compound."""

    compound: str
    support_tier: str
    support_reason: str
    prediction_health: str
    miami: CompoundPoolSupport
    recency: CompoundPoolSupport
    combined_raw_laps: int
    combined_model_laps: int
    combined_model_stints: int
    combined_model_races: int
    hybrid_adjustment_weight: float
    warnings: List[str]


def create_default_hybrid_context() -> HybridModelingContext:
    """Create the default role-based hybrid context."""
    pools = [
        DataPoolMetadata(
            pool_id="miami_historical",
            name="Miami Historical (2022-2025)",
            years=[2022, 2023, 2024, 2025],
            circuits=["Miami"],
            regulation_era="2022-2025",
            recency_weight=0.0,
            circuit_role="miami_anchor",
            target_race_context="Miami",
            description="Circuit anchor for Miami degradation and pit-loss behavior.",
        ),
        DataPoolMetadata(
            pool_id="season_2026_pre_miami",
            name="2026 Pre-Miami Races (Australia, China, Japan, etc.)",
            years=[2026],
            circuits=["Australia", "China", "Japan"],
            regulation_era="2026",
            recency_weight=1.0,
            circuit_role="recency_adjustment",
            target_race_context="General",
            description="Current-season recency support used only as a bounded adjustment.",
        ),
    ]
    return HybridModelingContext(
        pools_config=pools,
        weighting_scheme="role_based_adjustment",
        notes=(
            "Miami remains the circuit anchor. Non-Miami 2026 data provides a bounded "
            "recency adjustment and support signal at prediction time."
        ),
    )


def load_data_pool(dataset: str, project_root: Path | str = ".") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load one canonical dataset with error capture."""
    try:
        loader = DataLoader(project_root=project_root)
        df = loader.load_data(dataset=dataset, fallback=False)
        if df is None or df.empty:
            return None, f"Dataset {dataset} returned empty"
        required = {"Compound", "TyreLife", "LapTime", "Driver", "Stint"}
        if not required.issubset(df.columns):
            return None, f"Missing columns: {sorted(required - set(df.columns))}"
        return df, None
    except Exception as exc:
        return None, str(exc)


def _prepare_pool_model_df(df: pd.DataFrame) -> pd.DataFrame:
    selected = select_relevant_columns(df)
    pit_df = detect_pit_stops(selected)
    clean_df = clean_laps(pit_df)
    return build_model_df(clean_df)


def build_hybrid_dataset(
    context: HybridModelingContext,
    project_root: Path | str = ".",
    apply_weights: bool = False,
) -> HybridModelingContext:
    """Load the configured pools for inspection/context.

    ``apply_weights`` is retained for compatibility but ignored. The resulting
    ``blended_data`` is only a raw concatenation for context/audit and is not
    the canonical modeling source anymore.
    """
    del apply_weights

    project_root = Path(project_root)
    active_pools: List[DataPoolMetadata] = []
    raw_frames: List[pd.DataFrame] = []

    for pool in context.pools_config:
        df, error = load_data_pool(pool.pool_id, project_root=project_root)
        if error:
            pool.excluded_reason = error
            logger.warning("Hybrid pool %s excluded: %s", pool.pool_id, error)
            continue

        pool.sample_count = int(len(df))
        pool.compound_samples = {
            str(compound): int(count)
            for compound, count in df["Compound"].fillna("nan").value_counts(dropna=False).items()
        }

        frame = df.copy()
        frame["__phase2b_pool_id"] = pool.pool_id
        raw_frames.append(frame)
        active_pools.append(pool)

    context.active_pools = active_pools
    context.blended_data = pd.concat(raw_frames, ignore_index=True) if raw_frames else None
    return context


def _count_unique_units(df: pd.DataFrame) -> Tuple[int, int]:
    """Return unique stint and race counts for a compound slice."""
    if df.empty:
        return 0, 0
    stint_cols = [col for col in ["season", "event_name", "session_name", "Driver", "Stint"] if col in df.columns]
    race_cols = [col for col in ["season", "event_name", "session_name"] if col in df.columns]
    stint_count = int(df[stint_cols].drop_duplicates().shape[0]) if stint_cols else 0
    race_count = int(df[race_cols].drop_duplicates().shape[0]) if race_cols else 0
    return stint_count, race_count


def _prediction_valid(model: DegradationEvaluationResult, compound: str) -> bool:
    probes = [model.predict_lap_time(compound, tyre_life) for tyre_life in (1, 5, 10)]
    return all(value is not None and np.isfinite(value) for value in probes)


def _build_pool_support(
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    model: DegradationEvaluationResult,
    compound: str,
) -> CompoundPoolSupport:
    raw_slice = raw_df[raw_df["Compound"] == compound].copy()
    model_slice = model_df[model_df["Compound"] == compound].copy()
    raw_stints, raw_races = _count_unique_units(raw_slice)
    model_stints, model_races = _count_unique_units(model_slice)
    info = model.get_model_info(compound)
    return CompoundPoolSupport(
        raw_laps=int(len(raw_slice)),
        model_laps=int(len(model_slice)),
        raw_stints=raw_stints,
        model_stints=model_stints,
        raw_races=raw_races,
        model_races=model_races,
        model_type=info.get("model_type"),
        prediction_valid=_prediction_valid(model, compound),
    )


def _support_reason_and_tier(compound: str, miami: CompoundPoolSupport, recency: CompoundPoolSupport) -> Tuple[str, str]:
    """Return explicit support tier and rationale."""
    combined_model_laps = miami.model_laps + recency.model_laps
    combined_model_stints = miami.model_stints + recency.model_stints
    combined_model_races = miami.model_races + recency.model_races
    prediction_valid = miami.prediction_valid or recency.prediction_valid

    if (
        prediction_valid
        and miami.model_laps >= 300
        and combined_model_laps >= 600
        and miami.model_races >= 3
        and combined_model_stints >= 20
    ):
        return (
            "High",
            (
                f"High support: Miami anchor has {miami.model_laps} model laps across "
                f"{miami.model_races} races, with {combined_model_laps} total model laps "
                f"and {combined_model_stints} model stints across anchor+recency pools."
            ),
        )

    if (
        prediction_valid
        and miami.model_laps >= 50
        and combined_model_laps >= 150
        and miami.model_races >= 2
        and combined_model_stints >= 8
    ):
        return (
            "Moderate",
            (
                f"Moderate support: Miami anchor is usable but limited "
                f"({miami.model_laps} Miami model laps; {combined_model_laps} total model laps; "
                f"{combined_model_stints} model stints)."
            ),
        )

    return (
        "Low",
        (
            f"Low support: anchor+support data are thin or prediction health is weak "
            f"({miami.model_laps} Miami model laps; {combined_model_laps} total model laps; "
            f"{combined_model_stints} model stints; prediction valid={prediction_valid})."
        ),
    )


def _adjustment_weight_for_tier(support_tier: str) -> float:
    return {"High": 0.20, "Moderate": 0.35, "Low": 0.55}.get(support_tier, 0.35)


def _adjustment_cap(compound: str) -> float:
    return {"SOFT": 3.0, "MEDIUM": 1.5, "HARD": 1.5}.get(compound, 2.0)


class RoleBasedHybridModel:
    """Miami-anchor degradation model with bounded recency adjustment."""

    def __init__(
        self,
        miami_models: DegradationEvaluationResult,
        recency_models: DegradationEvaluationResult,
        support_summary: Dict[str, CompoundSupportSummary],
    ) -> None:
        self.miami_models = miami_models
        self.recency_models = recency_models
        self.support_summary = support_summary
        self.compounds = ("SOFT", "MEDIUM", "HARD")

    def predict_lap_time(self, compound: str, tyre_life: int) -> Optional[float]:
        support = self.support_summary.get(compound)
        miami_pred = self.miami_models.predict_lap_time(compound, tyre_life)
        recency_pred = self.recency_models.predict_lap_time(compound, tyre_life)

        if miami_pred is None and recency_pred is None:
            return None
        if miami_pred is None:
            return recency_pred
        if recency_pred is None:
            return miami_pred

        adjustment_weight = support.hybrid_adjustment_weight if support else 0.35
        weighted_delta = adjustment_weight * (recency_pred - miami_pred)
        bounded_delta = float(np.clip(weighted_delta, -_adjustment_cap(compound), _adjustment_cap(compound)))
        return float(miami_pred + bounded_delta)

    def get_model_info(self, compound: str) -> Dict:
        support = self.support_summary.get(compound)
        miami_info = self.miami_models.get_model_info(compound)
        recency_info = self.recency_models.get_model_info(compound)
        if support is None:
            return {
                "compound": compound,
                "model_type": None,
                "support_tier": "Low",
                "samples": 0,
                "is_piecewise": False,
                "breakpoint_tyre_life": None,
            }

        return {
            "compound": compound,
            "model_type": "ROLE-BASED HYBRID",
            "support_tier": support.support_tier,
            "support_reason": support.support_reason,
            "prediction_health": support.prediction_health,
            "samples": support.combined_model_laps,
            "miami_model_laps": support.miami.model_laps,
            "recency_model_laps": support.recency.model_laps,
            "hybrid_adjustment_weight": support.hybrid_adjustment_weight,
            "miami_model_type": miami_info.get("model_type"),
            "recency_model_type": recency_info.get("model_type"),
            "is_piecewise": bool(miami_info.get("is_piecewise") or recency_info.get("is_piecewise")),
            "breakpoint_tyre_life": miami_info.get("breakpoint_tyre_life") or recency_info.get("breakpoint_tyre_life"),
            "warnings": list(support.warnings),
        }

    def get_support_info(self, compound: str) -> Dict:
        support = self.support_summary.get(compound)
        return asdict(support) if support else {}

    def to_dict(self) -> dict:
        return {compound: asdict(summary) for compound, summary in self.support_summary.items()}


def build_role_based_hybrid_model(
    project_root: Path | str = ".",
    miami_raw_override: Optional[pd.DataFrame] = None,
    recency_raw_override: Optional[pd.DataFrame] = None,
) -> Tuple[RoleBasedHybridModel, HybridModelingContext]:
    """Build the canonical role-based hybrid degradation model."""
    project_root = Path(project_root)
    context = build_hybrid_dataset(create_default_hybrid_context(), project_root=project_root)
    loader = DataLoader(project_root=project_root)

    miami_raw = miami_raw_override if miami_raw_override is not None else loader.load_data("miami_historical")
    recency_raw = recency_raw_override if recency_raw_override is not None else loader.load_data("season_2026_pre_miami")

    miami_model_df = _prepare_pool_model_df(miami_raw)
    recency_model_df = _prepare_pool_model_df(recency_raw)

    miami_models = evaluate_all_degradation(miami_model_df, use_fuel_correction=True, use_piecewise=True)
    recency_models = evaluate_all_degradation(recency_model_df, use_fuel_correction=True, use_piecewise=True)

    support_summary: Dict[str, CompoundSupportSummary] = {}
    for compound in ("SOFT", "MEDIUM", "HARD"):
        miami_support = _build_pool_support(miami_raw, miami_model_df, miami_models, compound)
        recency_support = _build_pool_support(recency_raw, recency_model_df, recency_models, compound)
        support_tier, support_reason = _support_reason_and_tier(compound, miami_support, recency_support)
        warnings: List[str] = []
        if support_tier == "Low":
            warnings.append("low_support")
        elif support_tier == "Moderate":
            warnings.append("moderate_support")
        if not miami_support.prediction_valid:
            warnings.append("miami_anchor_prediction_weak")
        if compound == "SOFT":
            warnings.append("soft_requires_caution")

        support_summary[compound] = CompoundSupportSummary(
            compound=compound,
            support_tier=support_tier,
            support_reason=support_reason,
            prediction_health="valid" if (miami_support.prediction_valid or recency_support.prediction_valid) else "fragile",
            miami=miami_support,
            recency=recency_support,
            combined_raw_laps=miami_support.raw_laps + recency_support.raw_laps,
            combined_model_laps=miami_support.model_laps + recency_support.model_laps,
            combined_model_stints=miami_support.model_stints + recency_support.model_stints,
            combined_model_races=miami_support.model_races + recency_support.model_races,
            hybrid_adjustment_weight=_adjustment_weight_for_tier(support_tier),
            warnings=warnings,
        )

    return RoleBasedHybridModel(miami_models, recency_models, support_summary), context


def summarize_hybrid_context(
    context: HybridModelingContext,
    output_path: Optional[Path | str] = None,
) -> Dict:
    """Generate a role-based summary for UI/docs/artifacts."""
    total_samples = int(sum(pool.sample_count for pool in context.active_pools))
    summary = {
        "metadata": {
            "timestamp": context.timestamp,
            "weighting_scheme": context.weighting_scheme,
            "total_active_pools": len(context.active_pools),
        },
        "data_grouping": [
            {
                "pool_id": pool.pool_id,
                "name": pool.name,
                "years": pool.years,
                "circuits": pool.circuits,
                "role": pool.circuit_role,
                "target_race_context": pool.target_race_context,
                "recency_weight": pool.recency_weight,
                "normalized_weight": None,
                "sample_counts": {
                    "total_laps": pool.sample_count,
                    "by_compound": pool.compound_samples,
                },
            }
            for pool in context.active_pools
        ],
        "blending_strategy": {
            "method": "role-based prediction adjustment",
            "rationale": (
                "Miami remains the circuit anchor. 2026 pre-Miami races are not flattened "
                "into Miami truth; they only supply bounded recency adjustments and support."
            ),
            "pools": [
                {
                    "pool_id": pool.pool_id,
                    "role": pool.circuit_role,
                    "description": pool.description,
                }
                for pool in context.active_pools
            ],
        },
        "total_laps": total_samples,
        "notes": context.notes,
    }

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    return summary


def load_or_build_hybrid_dataset(
    project_root: Path | str = ".",
    custom_context: Optional[HybridModelingContext] = None,
) -> Tuple[pd.DataFrame, HybridModelingContext]:
    """Load configured pools for inspection/context reporting."""
    context = custom_context or create_default_hybrid_context()
    context = build_hybrid_dataset(context, project_root=project_root)
    if context.blended_data is None or context.blended_data.empty:
        raise RuntimeError(
            "Hybrid context loading failed: no source pools were available."
        )
    return context.blended_data, context
