"""Phase 1C: Improved degradation modeling with cliff-aware piecewise fitting.

This module builds on Phase 1B's fuel correction to fit more realistic degradation models
that detect tyre-wear cliffs (abrupt transitions from manageable to sharp wear).

Approach:
- Fit piecewise linear degradation models with a breakpoint in tyre life
- Compare against baseline linear fit
- Fall back to linear if insufficient data
- Use fuel-corrected lap times by default when available
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PiecewiseModel(NamedTuple):
    """Piecewise degradation model for a single compound."""

    # Breakpoint (cliff tyre-life value)
    breakpoint_tyre_life: int

    # Pre-cliff segment
    pre_cliff_slope: float
    pre_cliff_intercept: float
    pre_cliff_samples: int

    # Post-cliff segment
    post_cliff_slope: float
    post_cliff_intercept: float
    post_cliff_samples: int

    # Fit quality
    total_samples: int
    rss_piecewise: float  # Residual sum of squares
    rss_linear: Optional[float]  # For comparison
    improvement_percent: Optional[float]  # (rss_linear - rss_piecewise) / rss_linear
    fell_back_to_linear: bool


class LinearModel(NamedTuple):
    """Linear degradation model for a single compound."""

    slope: float
    intercept: float
    samples: int
    rss: float


def find_cliff_breakpoint(
    compound_data: pd.DataFrame,
    time_col: str = "LapTime",
    min_samples_per_segment: int = 10,
) -> Optional[int]:
    """Find optimal tyre-life breakpoint (cliff) by grid search.

    Searches candidate breakpoints and fits two line segments, minimizing total
    residual sum of squares subject to minimum sample counts in both segments.

    Args:
        compound_data: DataFrame with columns TyreLife, time_col (LapTime or FuelCorrectedLapTime)
        time_col: Column name for lap times
        min_samples_per_segment: Minimum samples required in pre- and post-cliff segments

    Returns:
        Optimal breakpoint tyre-life value, or None if no valid breakpoint found
    """
    if len(compound_data) < 2 * min_samples_per_segment:
        return None

    # Get candidate breakpoints (must have sufficient samples on both sides)
    tyre_life_values = sorted(compound_data["TyreLife"].unique())
    if len(tyre_life_values) < 5:
        return None

    best_breakpoint = None
    best_rss = float("inf")

    for candidate_bp in tyre_life_values[1:-1]:
        pre_cliff = compound_data[compound_data["TyreLife"] <= candidate_bp]
        post_cliff = compound_data[compound_data["TyreLife"] > candidate_bp]

        if len(pre_cliff) < min_samples_per_segment or len(post_cliff) < min_samples_per_segment:
            continue

        try:
            # Fit pre-cliff segment
            slope_pre, intercept_pre = np.polyfit(pre_cliff["TyreLife"], pre_cliff[time_col], 1)

            # Fit post-cliff segment
            slope_post, intercept_post = np.polyfit(post_cliff["TyreLife"], post_cliff[time_col], 1)

            # Compute total RSS
            pred_pre = slope_pre * pre_cliff["TyreLife"] + intercept_pre
            pred_post = slope_post * post_cliff["TyreLife"] + intercept_post
            rss = float(
                np.sum((pre_cliff[time_col] - pred_pre) ** 2)
                + np.sum((post_cliff[time_col] - pred_post) ** 2)
            )

            if rss < best_rss:
                best_rss = rss
                best_breakpoint = candidate_bp
        except Exception as e:
            logger.debug(f"Failed to fit breakpoint {candidate_bp}: {e}")
            continue

    return best_breakpoint


def fit_piecewise_model(
    compound_data: pd.DataFrame,
    compound: str,
    time_col: str = "LapTime",
    min_samples_per_segment: int = 10,
) -> Tuple[Optional[PiecewiseModel], Optional[LinearModel]]:
    """Fit piecewise and linear degradation models for a single compound.

    Returns:
        (piecewise_model, linear_model) where either may be None on failure
    """
    if len(compound_data) < 2 * min_samples_per_segment:
        logger.info(f"Compound {compound}: insufficient samples ({len(compound_data)}) for piecewise fitting")
        return None, None

    # Fit baseline linear model
    try:
        slope_linear, intercept_linear = np.polyfit(compound_data["TyreLife"], compound_data[time_col], 1)
        pred_linear = slope_linear * compound_data["TyreLife"] + intercept_linear
        rss_linear = float(np.sum((compound_data[time_col] - pred_linear) ** 2))
        linear_model = LinearModel(
            slope=float(slope_linear),
            intercept=float(intercept_linear),
            samples=len(compound_data),
            rss=rss_linear,
        )
    except Exception as e:
        logger.warning(f"Compound {compound}: failed to fit linear model: {e}")
        return None, None

    # Find optimal breakpoint
    breakpoint = find_cliff_breakpoint(compound_data, time_col, min_samples_per_segment)

    # If no breakpoint found, return linear model only
    if breakpoint is None:
        piecewise_model = PiecewiseModel(
            breakpoint_tyre_life=None,
            pre_cliff_slope=slope_linear,
            pre_cliff_intercept=intercept_linear,
            pre_cliff_samples=len(compound_data),
            post_cliff_slope=np.nan,
            post_cliff_intercept=np.nan,
            post_cliff_samples=0,
            total_samples=len(compound_data),
            rss_piecewise=rss_linear,
            rss_linear=rss_linear,
            improvement_percent=0.0,
            fell_back_to_linear=True,
        )
        logger.info(
            f"Compound {compound}: no cliff detected, using linear fallback "
            f"(slope={slope_linear:.4f}, rss={rss_linear:.2f})"
        )
        return piecewise_model, linear_model

    # Fit piecewise model at breakpoint
    pre_cliff = compound_data[compound_data["TyreLife"] <= breakpoint]
    post_cliff = compound_data[compound_data["TyreLife"] > breakpoint]

    try:
        slope_pre, intercept_pre = np.polyfit(pre_cliff["TyreLife"], pre_cliff[time_col], 1)
        slope_post, intercept_post = np.polyfit(post_cliff["TyreLife"], post_cliff[time_col], 1)

        pred_pre = slope_pre * pre_cliff["TyreLife"] + intercept_pre
        pred_post = slope_post * post_cliff["TyreLife"] + intercept_post
        rss_piecewise = float(
            np.sum((pre_cliff[time_col] - pred_pre) ** 2) + np.sum((post_cliff[time_col] - pred_post) ** 2)
        )

        improvement_pct = None
        if rss_linear > 0:
            improvement_pct = float((rss_linear - rss_piecewise) / rss_linear * 100)

        piecewise_model = PiecewiseModel(
            breakpoint_tyre_life=int(breakpoint),
            pre_cliff_slope=float(slope_pre),
            pre_cliff_intercept=float(intercept_pre),
            pre_cliff_samples=len(pre_cliff),
            post_cliff_slope=float(slope_post),
            post_cliff_intercept=float(intercept_post),
            post_cliff_samples=len(post_cliff),
            total_samples=len(compound_data),
            rss_piecewise=rss_piecewise,
            rss_linear=rss_linear,
            improvement_percent=improvement_pct,
            fell_back_to_linear=False,
        )

        improvement_str = f"{improvement_pct:.1f}%" if improvement_pct is not None else "N/A"

        logger.info(
            f"Compound {compound}: cliff at tyre-life={breakpoint}, "
            f"pre_slope={slope_pre:.4f}, post_slope={slope_post:.4f}, "
            f"improvement={improvement_str}"
        )

        return piecewise_model, linear_model
    except Exception as e:
        logger.warning(f"Compound {compound}: failed to fit piecewise model: {e}")
        return None, linear_model


def fit_all_piecewise_models(
    model_laps: pd.DataFrame,
    use_fuel_corrected: bool = True,
    compounds: Tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
) -> Tuple[Dict[str, PiecewiseModel], Dict[str, LinearModel]]:
    """Fit piecewise and linear models for all compounds.

    Args:
        model_laps: Model-grade laps (pit-excluded, accurate, green-flag)
        use_fuel_corrected: If True, use FuelCorrectedLapTime; else use LapTime
        compounds: Tuple of compound names to fit

    Returns:
        (piecewise_models_dict, linear_models_dict)
    """
    time_col = "FuelCorrectedLapTime" if use_fuel_corrected else "LapTime"

    if use_fuel_corrected and time_col not in model_laps.columns:
        logger.warning(f"FuelCorrectedLapTime not found, falling back to LapTime")
        time_col = "LapTime"

    piecewise_models = {}
    linear_models = {}

    for compound in compounds:
        compound_data = model_laps[model_laps["Compound"] == compound].copy()

        if len(compound_data) < 5:
            logger.warning(f"Compound {compound}: only {len(compound_data)} samples, skipping")
            continue

        pw_model, lin_model = fit_piecewise_model(compound_data, compound, time_col)

        if pw_model is not None:
            piecewise_models[compound] = pw_model
        if lin_model is not None:
            linear_models[compound] = lin_model

    return piecewise_models, linear_models


def create_degradation_comparison_table(
    piecewise_models: Dict[str, PiecewiseModel],
    linear_models: Dict[str, LinearModel],
) -> pd.DataFrame:
    """Create a human-readable comparison table for degradation models.

    Returns:
        DataFrame with one row per compound showing linear vs piecewise comparison
    """
    rows = []

    for compound in ["SOFT", "MEDIUM", "HARD"]:
        pw = piecewise_models.get(compound)
        lin = linear_models.get(compound)

        if lin is None and pw is None:
            continue

        row = {
            "Compound": compound,
            "Samples": pw.total_samples if pw else lin.samples,
            "Model Type": "Piecewise" if (pw and not pw.fell_back_to_linear) else "Linear",
            "Slope (s/lap)": pw.pre_cliff_slope if pw else lin.slope,
        }

        if pw and not pw.fell_back_to_linear:
            row["Breakpoint (tyre-life)"] = pw.breakpoint_tyre_life
            row["Pre-Cliff Slope"] = f"{pw.pre_cliff_slope:.4f}"
            row["Post-Cliff Slope"] = f"{pw.post_cliff_slope:.4f}"
            row["RSS"] = f"{pw.rss_piecewise:.1f}"
            row["Improvement %"] = f"{pw.improvement_percent:.1f}%" if pw.improvement_percent else "N/A"
        else:
            row["Breakpoint (tyre-life)"] = "None"
            row["Pre-Cliff Slope"] = "N/A"
            row["Post-Cliff Slope"] = "N/A"
            row["RSS"] = f"{lin.rss:.1f}" if lin else "N/A"
            row["Improvement %"] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def save_degradation_comparison(
    piecewise_models: Dict[str, PiecewiseModel],
    linear_models: Dict[str, LinearModel],
    output_path: Path | str = "data/processed/degradation_model_comparison.json",
) -> Path:
    """Save degradation model comparison to JSON.

    Args:
        piecewise_models: Dict of piecewise models
        linear_models: Dict of linear models
        output_path: Where to save the artifact

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "method": "Piecewise linear degradation with cliff detection",
        "piecewise_models": {},
        "linear_models": {},
    }

    for compound, model in piecewise_models.items():
        comparison["piecewise_models"][compound] = {
            "breakpoint_tyre_life": model.breakpoint_tyre_life,
            "pre_cliff_slope_s_per_lap": model.pre_cliff_slope,
            "pre_cliff_intercept_s": model.pre_cliff_intercept,
            "pre_cliff_samples": model.pre_cliff_samples,
            "post_cliff_slope_s_per_lap": model.post_cliff_slope if not np.isnan(model.post_cliff_slope) else None,
            "post_cliff_intercept_s": model.post_cliff_intercept if not np.isnan(model.post_cliff_intercept) else None,
            "post_cliff_samples": model.post_cliff_samples,
            "total_samples": model.total_samples,
            "rss_piecewise": model.rss_piecewise,
            "rss_linear": model.rss_linear,
            "improvement_percent": model.improvement_percent,
            "fell_back_to_linear": model.fell_back_to_linear,
        }

    for compound, model in linear_models.items():
        comparison["linear_models"][compound] = {
            "slope_s_per_lap": model.slope,
            "intercept_s": model.intercept,
            "samples": model.samples,
            "rss": model.rss,
        }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Degradation model comparison saved to {output_path}")
    return output_path


def print_degradation_comparison(
    piecewise_models: Dict[str, PiecewiseModel],
    linear_models: Dict[str, LinearModel],
) -> None:
    """Print human-readable degradation model comparison."""
    print("\n" + "=" * 120)
    print("PHASE 1C: DEGRADATION MODELING (Piecewise with Cliff Detection)")
    print("=" * 120)

    table = create_degradation_comparison_table(piecewise_models, linear_models)
    if len(table) > 0:
        print("\nDegradation Model Comparison:")
        print(table.to_string(index=False))
    else:
        print("\nNo degradation models available.")

    print("\n" + "=" * 120)
