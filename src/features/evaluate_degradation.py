"""Phase 1 end-to-end degradation evaluation.

This module integrates Phase 1A (data), Phase 1B (fuel correction), and Phase 1C 
(piecewise degradation modeling) into a unified evaluation pipeline.

**Advantages:**
- Fuel correction applied automatically and transparently
- Piecewise models used when supported (with automatic fallback to linear)
- Clear reporting on which models are actually active
- Clean API for strategy layer

**Usage:**
    result = evaluate_all_degradation(model_laps, use_fuel_correction=True)
    prediction = result.predict_lap_time(compound="MEDIUM", tyre_life=5)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.features.degradation_modeling import LinearModel, PiecewiseModel, fit_all_piecewise_models
from src.features.fuel_correction import apply_fuel_correction, estimate_fuel_effect

logger = logging.getLogger(__name__)


class DegradationEvaluationResult:
    """Unified degradation evaluation result from Phase 1 pipeline.
    
    Provides a clean interface abstracting away detail of model type
    and providing convenient prediction and reporting methods.
    """

    def __init__(
        self,
        piecewise_models: Dict[str, PiecewiseModel],
        linear_models: Dict[str, LinearModel],
        fuel_corrected: bool,
        compounds: Tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    ):
        """Initialize evaluation result.
        
        Args:
            piecewise_models: Dict of piecewise models (may be empty if all fell back to linear)
            linear_models: Dict of linear models
            fuel_corrected: Whether fuel correction was applied
            compounds: Available compounds
        """
        self.piecewise_models = piecewise_models
        self.linear_models = linear_models
        self.fuel_corrected = fuel_corrected
        self.compounds = compounds

    def predict_lap_time(
        self,
        compound: str,
        tyre_life: int,
    ) -> Optional[float]:
        """Predict lap time for a given compound and tyre-life.
        
        Automatically uses piecewise model if available, falls back to linear.
        
        Args:
            compound: Compound name (e.g., "MEDIUM")
            tyre_life: Current tyre-life value
            
        Returns:
            Predicted lap time in seconds, or None if compound not found
        """
        # Try piecewise first
        if compound in self.piecewise_models:
            pw_model = self.piecewise_models[compound]
            
            # Use piecewise if not fallen back to linear
            if not pw_model.fell_back_to_linear:
                if tyre_life <= pw_model.breakpoint_tyre_life:
                    return (
                        pw_model.pre_cliff_slope * tyre_life
                        + pw_model.pre_cliff_intercept
                    )
                else:
                    return (
                        pw_model.post_cliff_slope * tyre_life
                        + pw_model.post_cliff_intercept
                    )
            # Fall back to linear within piecewise model structure
            else:
                return (
                    pw_model.pre_cliff_slope * tyre_life
                    + pw_model.pre_cliff_intercept
                )
        
        # Fall back to linear model
        if compound in self.linear_models:
            lin_model = self.linear_models[compound]
            return lin_model.slope * tyre_life + lin_model.intercept
        
        logger.warning(f"No degradation model found for compound {compound}")
        return None

    def get_model_info(self, compound: str) -> Dict:
        """Get information about which model is active for a compound.
        
        Args:
            compound: Compound name
            
        Returns:
            Dict with keys: model_type, is_piecewise, breakpoint (if piecewise), 
                            samples, fuel_corrected_applied
        """
        info = {
            "compound": compound,
            "fuel_corrected_applied": self.fuel_corrected,
            "model_type": None,
            "is_piecewise": False,
            "breakpoint_tyre_life": None,
            "samples": 0,
        }
        
        if compound in self.piecewise_models:
            pw = self.piecewise_models[compound]
            info["samples"] = pw.total_samples
            if not pw.fell_back_to_linear:
                info["model_type"] = "PIECEWISE"
                info["is_piecewise"] = True
                info["breakpoint_tyre_life"] = pw.breakpoint_tyre_life
            else:
                info["model_type"] = "LINEAR (piecewise fallback)"
                info["is_piecewise"] = False
        elif compound in self.linear_models:
            lin = self.linear_models[compound]
            info["samples"] = lin.samples
            info["model_type"] = "LINEAR"
            info["is_piecewise"] = False
        
        return info

    def to_legacy_linear_models(self) -> Dict[str, Tuple[float, float]]:
        """Convert to legacy linear model format for backward compatibility.
        
        Returns:
            Dict mapping compound → (slope, intercept)
            Uses piecewise pre-cliff slope if available, linear otherwise.
        """
        legacy = {}
        for compound in self.compounds:
            if compound in self.piecewise_models:
                pw = self.piecewise_models[compound]
                legacy[compound] = (pw.pre_cliff_slope, pw.pre_cliff_intercept)
            elif compound in self.linear_models:
                lin = self.linear_models[compound]
                legacy[compound] = (lin.slope, lin.intercept)
        return legacy


def evaluate_all_degradation(
    model_laps: pd.DataFrame,
    use_fuel_correction: bool = True,
    use_piecewise: bool = True,
    compounds: Tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
) -> DegradationEvaluationResult:
    """Run full Phase 1 degradation pipeline.
    
    Integrates:
    - Phase 1B: Fuel correction (optional, recommended)
    - Phase 1C: Piecewise degradation with cliff detection (optional, recommended)
    - Linear baseline (always available as fallback)
    
    Args:
        model_laps: Model-grade laps (pit-excluded, accurate, green-flag)
        use_fuel_correction: If True, apply Phase 1B fuel correction
        use_piecewise: If True, attempt Phase 1C piecewise fitting
        compounds: Tuple of compound names to fit
        
    Returns:
        DegradationEvaluationResult with unified interface
    """
    corrected_laps = model_laps.copy()
    fuel_corrected_applied = False
    
    # Step 1: Apply fuel correction (Phase 1B) if requested
    if use_fuel_correction:
        try:
            fuel_effects = estimate_fuel_effect(model_laps)
            if fuel_effects:
                corrected_laps = apply_fuel_correction(model_laps, fuel_effects)
                fuel_corrected_applied = True
                logger.info(
                    f"Phase 1B fuel correction applied for {len(fuel_effects)} compound(s)"
                )
            else:
                logger.info("Phase 1B fuel correction skipped (no fuel effects estimated)")
        except Exception as e:
            logger.warning(f"Phase 1B fuel correction failed: {e}. Using raw lap times.")
    else:
        logger.info("Phase 1B fuel correction disabled")
    
    # Step 2: Fit piecewise models (Phase 1C) if requested
    piecewise_models = {}
    linear_models = {}
    
    if use_piecewise:
        try:
            piecewise_models, linear_models = fit_all_piecewise_models(
                corrected_laps,
                use_fuel_corrected=fuel_corrected_applied,
                compounds=compounds,
            )
            num_piecewise = sum(
                1 for m in piecewise_models.values() 
                if not m.fell_back_to_linear
            )
            logger.info(
                f"Phase 1C piecewise fitting complete: "
                f"{num_piecewise} piecewise, "
                f"{len(piecewise_models) - num_piecewise} linear fallback"
            )
        except Exception as e:
            logger.warning(f"Phase 1C piecewise fitting failed: {e}. Using linear models.")
    else:
        logger.info("Phase 1C piecewise degradation disabled")
    
    # If no piecewise or piecewise disabled, build linear models
    if not piecewise_models:
        try:
            from src.features.build_features import create_degradation_table, fit_degradation_models
            
            time_col = "FuelCorrectedLapTime" if fuel_corrected_applied else "LapTime"
            deg_table = create_degradation_table(
                corrected_laps,
                use_fuel_corrected=fuel_corrected_applied,
            )
            linear_dict = fit_degradation_models(deg_table, compounds=compounds)
            
            # Convert to LinearModel objects
            for compound, (slope, intercept) in linear_dict.items():
                compound_data = corrected_laps[corrected_laps["Compound"] == compound]
                pred = slope * compound_data["TyreLife"] + intercept
                rss = float(np.sum((compound_data[time_col] - pred) ** 2))
                linear_models[compound] = LinearModel(
                    slope=slope,
                    intercept=intercept,
                    samples=len(compound_data),
                    rss=rss,
                )
            
            logger.info(f"Linear models fitted for {len(linear_models)} compound(s)")
        except Exception as e:
            logger.warning(f"Linear model fitting failed: {e}")
    
    return DegradationEvaluationResult(
        piecewise_models=piecewise_models,
        linear_models=linear_models,
        fuel_corrected=fuel_corrected_applied,
        compounds=compounds,
    )
