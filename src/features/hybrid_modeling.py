"""Phase 2B: Hybrid Data/Model Context for Miami-2026 Strategy Blend.

This module implements a structured hybrid modeling approach that combines:
1. Miami historical prior (2022–2025)
   - Circuit-specific context
   - Long-term degradation/pit-loss baseline
   - Medium recency weight
   
2. Current-season 2026 completed races prior
   - Highest recency weight (most recent car/tyre behavior)
   - Current season performance
   - Used for performance adjustments
   
3. Optional broader historical support data
   - Non-Miami historical races (if available)
   - Lower-weight support for sample size
   - Fallback if Miami/2026 data insufficient

**Design Philosophy:**
- Explicit, reviewable weighting (not hidden heuristics)
- Each pool has a defined role
- Weights are transparent and configurable
- Outputs are inspectable and auditable
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataPoolMetadata:
    """Metadata and configuration for a data pool (race group)."""
    
    pool_id: str
    """Unique identifier: 'miami_historical', 'season_2026_pre_miami', etc."""
    
    name: str
    """Human-readable name: 'Miami Historical (2022-2025)'."""
    
    years: List[int]
    """Years included in this pool."""
    
    circuits: List[str]
    """Circuits included (e.g., ['Miami'])."""
    
    regulation_era: str
    """Regulation era (e.g., '2022', '2024-2025', '2026')."""
    
    recency_weight: float
    """Raw weight factor (before normalization)."""
    
    circuit_role: str
    """Role of this pool: 'miami_specific', 'current_season', 'historical_support'."""
    
    target_race_context: str
    """What race this data used for modeling: 'Miami' or 'General'."""
    
    description: str
    """Purpose/rationale for this pool."""
    
    sample_count: int = 0
    """Number of laps in this pool (filled after load)."""
    
    compound_samples: Dict[str, int] = field(default_factory=dict)
    """Samples per compound: {'SOFT': 100, 'MEDIUM': 200, 'HARD': 300}."""
    
    excluded_reason: Optional[str] = None
    """If not used, why excluded."""


@dataclass
class HybridModelingContext:
    """Configuration for hybrid data/model blending."""
    
    pools_config: List[DataPoolMetadata]
    """Configuration for each data pool to attempt loading."""
    
    active_pools: List[DataPoolMetadata] = field(default_factory=list)
    """Pools that were successfully loaded (filled by build process)."""
    
    blended_data: Optional[pd.DataFrame] = None
    """Combined/weighted data (filled by build process)."""
    
    weighting_scheme: str = "explicit_recency"
    """Name of weighting strategy used: 'explicit_recency', 'equal', etc."""
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    """When this context was created."""
    
    notes: str = ""
    """Optional notes about the modeling context."""
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "weighting_scheme": self.weighting_scheme,
            "active_pools": [
                {
                    "pool_id": p.pool_id,
                    "name": p.name,
                    "years": p.years,
                    "circuits": p.circuits,
                    "recency_weight": p.recency_weight,
                    "circuit_role": p.circuit_role,
                    "sample_count": p.sample_count,
                    "compound_samples": p.compound_samples,
                }
                for p in self.active_pools
            ],
            "total_samples": sum(p.sample_count for p in self.active_pools),
            "total_compounds": sum(len(p.compound_samples) for p in self.active_pools),
            "notes": self.notes,
        }


def create_default_hybrid_context() -> HybridModelingContext:
    """Create default Phase 2B hybrid modeling context.
    
    **Strategy:**
    - Miami historical: weight 0.4 (circuit-specific, medium recency)
    - 2026 pre-Miami: weight 0.6 (current season, highest recency)
    - Broader historical: weight 0.1 if available (fallback support)
    
    **Rationale:**
    Miami data is valuable for circuit-specific context (pit loss, degradation patterns).
    But 2026 races are more recent and reflect current car/tyre performance.
    By blending 60% recency + 40% circuit-specific, we get adaptive strategy.
    
    Returns:
        HybridModelingContext with recommended pools.
    """
    pools = [
        DataPoolMetadata(
            pool_id="miami_historical",
            name="Miami Historical (2022–2025)",
            years=[2022, 2023, 2024, 2025],
            circuits=["Miami"],
            regulation_era="2022-2024",
            recency_weight=0.4,
            circuit_role="miami_specific",
            target_race_context="Miami",
            description="Miami-specific degradation, pit-loss, and strategy baseline",
        ),
        DataPoolMetadata(
            pool_id="season_2026_pre_miami",
            name="2026 Pre-Miami Races (Australia, China, Japan, etc.)",
            years=[2026],
            circuits=["Generic"],  # Will be filled from data
            regulation_era="2026",
            recency_weight=0.6,
            circuit_role="current_season",
            target_race_context="General",
            description="Current-season performance data (highest recency weight)",
        ),
    ]
    
    return HybridModelingContext(
        pools_config=pools,
        weighting_scheme="explicit_recency",
        notes="Default Phase 2B hybrid context: 40% Miami-specific + 60% 2026 recency",
    )


def load_data_pool(
    dataset: str,
    project_root: Path | str = ".",
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load a single data pool by name.
    
    Args:
        dataset: Dataset name ('miami_historical', 'season_2026_pre_miami', etc.)
        project_root: Path to project root
        
    Returns:
        (DataFrame or None, error_message or None)
        If successful: returns (DataFrame, None)
        If failed: returns (None, error_message)
    """
    from src.data.loader import DataLoader
    
    try:
        loader = DataLoader(project_root=project_root)
        df = loader.load_data(dataset=dataset, fallback=False)
        
        if df is None or len(df) == 0:
            return None, f"Dataset {dataset} returned empty"
        
        # Ensure required columns are present (handle both PascalCase from DataLoader and lowercase)
        # The DataLoader._normalize_schema converts to PascalCase for backward compatibility
        required_cols = ["Compound", "TyreLife", "LapTime", "Driver", "Stint"]
        lowercase_cols = [c.lower() for c in required_cols]
        
        # Check for either PascalCase or lowercase
        has_required = all(
            (c in df.columns) or (c.lower() in df.columns)
            for c in required_cols
        )
        
        if not has_required:
            missing_cols = [c for c in required_cols if c not in df.columns and c.lower() not in df.columns]
            logger.warning(
                f"  ⚠ {dataset}: Missing columns {missing_cols}; "
                "this pool may not be usable for modeling"
            )
            return None, f"Missing columns: {missing_cols}"
        
        return df, None
    except Exception as e:
        return None, str(e)


def build_hybrid_dataset(
    context: HybridModelingContext,
    project_root: Path | str = ".",
    apply_weights: bool = True,
) -> HybridModelingContext:
    """Load all pools in context and blend with weighting.
    
    Args:
        context: HybridModelingContext with pools_config
        project_root: Path to project root
        apply_weights: If True, duplicate samples per pool weight (optional; see note below)
        
    Returns:
        Updated context with active_pools and blended_data filled
        
    **Note on weighting:**
    We implement weighting at the sample level by duplicating/downsampling laps:
    - If weight > 1.0, duplicate (upweight)
    - If weight < 1.0, random sample (downweight)
    This ensures degradation models trained on blended data reflect pool importance.
    
    Alternative: weight models after fitting (per-pool alpha weighting).
    We use sample-level for simplicity and transparency.
    """
    project_root = Path(project_root)
    active_pools = []
    all_dfs = []
    
    # Load each pool
    for pool_config in context.pools_config:
        logger.info(f"Attempting to load pool: {pool_config.pool_id}")
        
        df, error = load_data_pool(
            dataset=pool_config.pool_id,
            project_root=project_root,
        )
        
        if error:
            logger.warning(f"  ✗ Failed: {error}")
            pool_config.excluded_reason = error
            continue
        
        # Record metadata
        pool_config.sample_count = len(df)
        
        # Safely get compound samples (use PascalCase from DataLoader)
        try:
            compound_col = "Compound" if "Compound" in df.columns else "compound"
            if compound_col in df.columns:
                pool_config.compound_samples = {
                    compound: len(df[df[compound_col] == compound])
                    for compound in df[compound_col].unique()
                }
            else:
                logger.warning(f"  ⚠ {pool_config.pool_id}: No compound column; skipping breakdown")
                pool_config.compound_samples = {}
        except Exception as e:
            logger.warning(f"  ⚠ {pool_config.pool_id}: Failed to compute compound samples: {e}")
            pool_config.compound_samples = {}
        
        # Add pool identifier column (for transparency/audit)
        df = df.copy()
        df["__phase2b_pool_id"] = pool_config.pool_id
        
        # Store pool with weight annotation
        all_dfs.append({
            "df": df,
            "pool_id": pool_config.pool_id,
            "weight": pool_config.recency_weight,
            "metadata": pool_config,
        })
        
        active_pools.append(pool_config)
        logger.info(
            f"  ✓ Loaded {len(df)} laps from {pool_config.pool_id} "
            f"(weight: {pool_config.recency_weight})"
        )
    
    # Normalize weights
    if all_dfs:
        total_weight = sum(item["weight"] for item in all_dfs)
        for item in all_dfs:
            item["normalized_weight"] = item["weight"] / total_weight
            logger.info(
                f"  {item['pool_id']}: normalized weight {item['normalized_weight']:.2%}"
            )
    
    # Log explicit pool composition BEFORE weighting (for audit trail)
    logger.info("\n" + "=" * 70)
    logger.info("HYBRID DATA POOL COMPOSITION BEFORE WEIGHTING")
    logger.info("=" * 70)
    total_raw_laps = sum(item["df"].shape[0] for item in all_dfs)
    for item in all_dfs:
        n_laps = item["df"].shape[0]
        raw_pct = 100.0 * n_laps / total_raw_laps if total_raw_laps > 0 else 0.0
        norm_weight = item["normalized_weight"]
        logger.info(
            f"  {item['pool_id']:30s} {n_laps:7,d} laps "
            f"({raw_pct:5.1f}% of raw) | weight={item['weight']} → {norm_weight:.1%}"
        )
    logger.info(f"  {'TOTAL RAW':30s} {total_raw_laps:7,d} laps")
    logger.info("=" * 70 + "\n")
    
    # Blend with weighting (optional: duplicate/downsample to reflect weights)
    if all_dfs and apply_weights:
        logger.info("HYBRID WEIGHTING & BLENDING (Sample-Level Replication)")
        logger.info("=" * 70)
        blended_parts = []
        total_blended_laps = 0
        
        for item in all_dfs:
            df = item["df"]
            norm_weight = item["normalized_weight"]
            pool_id = item['pool_id']
            
            # If weight > 1.0, duplicate; if < 1.0, downsample
            # This is approximate but transparent and reproducible
            if norm_weight > 0.5:  # Oversample
                repeat_factor = int(np.round(norm_weight * 10))
                df_replicated = pd.concat([df] * repeat_factor, ignore_index=True)
                blended_parts.append(df_replicated)
                n_before = len(df)
                n_after = len(df_replicated)
                total_blended_laps += n_after
                logger.info(
                    f"  {pool_id:30s} weight={norm_weight:.1%} "
                    f"→ replicate ×{repeat_factor:2d}  "
                    f"{n_before:7,d} → {n_after:7,d} laps"
                )
            else:  # Downsample
                sample_size = max(1, int(len(df) * norm_weight * 10))
                df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)
                blended_parts.append(df_sampled)
                n_before = len(df)
                n_after = len(df_sampled)
                total_blended_laps += n_after
                logger.info(
                    f"  {pool_id:30s} weight={norm_weight:.1%} "
                    f"→ downsample    "
                    f"{n_before:7,d} → {n_after:7,d} laps"
                )
        
        blended_data = pd.concat(blended_parts, ignore_index=True)
        logger.info("=" * 70)
        logger.info(f"  TOTAL BLENDED: {len(blended_data):,d} laps")
        logger.info("=" * 70 + "\n")
    elif all_dfs:
        # No sample-level weighting, just concatenate
        logger.info("BLENDING POOLS (No Sample-Level Reweighting)")
        logger.info("=" * 70)
        blended_data = pd.concat([item["df"] for item in all_dfs], ignore_index=True)
        logger.info(f"  TOTAL CONCATENATED: {len(blended_data):,d} laps")
        logger.info("=" * 70 + "\n")
    else:
        blended_data = None
    
    # Update context
    context.active_pools = active_pools
    context.blended_data = blended_data
    
    return context


def summarize_hybrid_context(
    context: HybridModelingContext,
    output_path: Optional[Path | str] = None,
) -> Dict:
    """Generate a summary of hybrid modeling context.
    
    Args:
        context: Populated HybridModelingContext
        output_path: If provided, save summary as JSON to this path
        
    Returns:
        Summary dict
    """
    summary = {
        "metadata": {
            "timestamp": context.timestamp,
            "weighting_scheme": context.weighting_scheme,
            "total_active_pools": len(context.active_pools),
        },
        "data_grouping": [
            {
                "pool_id": p.pool_id,
                "name": p.name,
                "years": p.years,
                "circuits": p.circuits,
                "role": p.circuit_role,
                "target_race_context": p.target_race_context,
                "recency_weight": p.recency_weight,
                "normalized_weight": (
                    p.recency_weight / sum(x.recency_weight for x in context.active_pools)
                    if context.active_pools
                    else 0.0
                ),
                "sample_counts": {
                    "total_laps": p.sample_count,
                    "by_compound": p.compound_samples,
                },
            }
            for p in context.active_pools
        ],
        "blending_strategy": {
            "method": "sample-level replication/downsampling",
            "rationale": "Ensures degradation models reflect pool importance without changing statistics",
            "pools": [
                {
                    "pool_id": p.pool_id,
                    "role": p.circuit_role,
                    "description": p.description,
                }
                for p in context.active_pools
            ],
        },
        "total_laps": (
            len(context.blended_data) if context.blended_data is not None else 0
        ),
        "notes": (
            f"Phase 2B Hybrid Modeling: Combines {len(context.active_pools)} data pools "
            f"with explicit recency weighting. "
            "Miami provides circuit-specific baseline; 2026 provides current-season recency."
        ),
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved hybrid context summary to {output_path}")
    
    return summary


def load_or_build_hybrid_dataset(
    project_root: Path | str = ".",
    custom_context: Optional[HybridModelingContext] = None,
) -> Tuple[pd.DataFrame, HybridModelingContext]:
    """Convenience function: load hybrid dataset with default or custom context.
    
    Args:
        project_root: Path to project root
        custom_context: Custom context (default: create_default_hybrid_context)
        
    Returns:
        (blended_data, populated_context)
    """
    if custom_context is None:
        context = create_default_hybrid_context()
    else:
        context = custom_context
    
    context = build_hybrid_dataset(context, project_root=project_root)
    
    if context.blended_data is None or len(context.blended_data) == 0:
        raise RuntimeError(
            "Hybrid dataset loading failed: no data available. "
            "Check that data/raw/miami_historical or data/raw/season_2026_pre_miami exist."
        )
    
    return context.blended_data, context
