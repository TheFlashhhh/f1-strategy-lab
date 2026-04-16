"""Phase 2C: Strategy Sensitivity & Uncertainty Analysis.

This module analyzes the stability of pit strategy recommendations under
reasonable variations in pit-loss and degradation assumptions.

**Key Design:**
- Scenario-based sensitivity analysis (NOT probabilistic/stochastic)
- Evaluates pit-loss sensitivity: ±1.0s, ±2.0s from baseline
- Evaluates degradation sensitivity: optimistic, baseline, pessimistic
- Reports which changes cause recommendation flips
- Provides stability classification: Stable, Moderately Sensitive, Fragile

**Scope & Limitations:**
- Does NOT add weather/thermal modeling
- Does NOT consider traffic, safety cars, or opponents
- Does NOT run Monte Carlo simulations
- Pure scenario-based: explore impact of specific assumption changes
- Honest about uncertainty: identifies when recommendations are brittle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.simulation.strategy_engine import recommend_best_strategy, StrategyPlan

logger = logging.getLogger(__name__)


@dataclass
class DegradationScaledModel:
    """Wrapper to apply degradation scaling to a baseline model.
    
    Allows unified interface for:
    - Original model (scale_factor=1.0)
    - Optimistic degradation (scale_factor<1.0: slower wear)
    - Pessimistic degradation (scale_factor>1.0: faster wear)
    """
    
    base_model: object
    scale_factor: float = 1.0
    
    def predict_lap_time(self, compound: str, tyre_life: int) -> Optional[float]:
        """Predict lap time with scaled degradation.
        
        Scaling works by adjusting the slope (degradation rate):
        - result = intercept + scale_factor * slope * tyre_life
        """
        base_prediction = self.base_model.predict_lap_time(compound, tyre_life)
        if base_prediction is None:
            return None
        
        # Try to extract model info to properly scale degradation
        try:
            model_info = self.base_model.get_model_info(compound)
            if model_info and model_info.get("model_type"):
                # For piecewise models, scale the slopes
                # For now, use simple scaling: scale the degradation delta
                # This is an approximation: (base_pred - intercept) * scale_factor + intercept
                
                # Get reference prediction at tyre_life = 1
                ref_pred = self.base_model.predict_lap_time(compound, 1)
                if ref_pred is not None:
                    # Estimate degradation per lap (approximately)
                    # This is simplified but reasonable for scenario analysis
                    base_at_1 = ref_pred
                    base_at_current = base_prediction
                    
                    # Extrapolate with scaled degradation
                    degradation_delta = base_at_current - base_at_1  # Wear from lap 1 to current
                    
                    # Apply scale factor to degradation
                    scaled_delta = degradation_delta * self.scale_factor
                    scaled_prediction = base_at_1 + scaled_delta
                    
                    return float(scaled_prediction) if not np.isnan(scaled_prediction) else None
        except Exception:
            pass
        
        # Fallback: return unscaled prediction (conservative)
        return base_prediction
    
    def get_model_info(self, compound: str) -> Dict:
        """Delegate to base model."""
        return self.base_model.get_model_info(compound)


@dataclass
class StrategyScenarioResult:
    """Result of evaluating strategy under a single sensitivity scenario."""
    
    scenario_name: str
    """Label for this scenario (e.g., 'pit_loss_minus_2s', 'degradation_pessimistic')."""
    
    pit_loss_value: float
    """Pit loss value used in this scenario."""
    
    degradation_scale_factor: float
    """Degradation scale factor (1.0 = baseline)."""
    
    best_plan: StrategyPlan
    """Best strategy recommendation under this scenario."""
    
    recommendation_changed: bool
    """Whether this recommendation differs from baseline."""
    
    change_details: Optional[str] = None
    """Human-readable description of what changed (if changed)."""


@dataclass
class PitLossSensitivityResult:
    """Results of pit-loss sensitivity analysis (baseline degradation only)."""
    
    baseline_pit_loss: float
    scenarios: List[StrategyScenarioResult] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary of pit-loss sensitivity."""
        changes = [s for s in self.scenarios if s.recommendation_changed]
        
        if not changes:
            return "Stable across pit-loss variations"
        
        flip_details = []
        for scenario in changes:
            delta_str = f"{scenario.pit_loss_value - self.baseline_pit_loss:+.1f}s"
            flip_details.append(f"  {scenario.scenario_name} ({delta_str}): {scenario.change_details}")
        
        return "Sensitive to pit-loss variation:\n" + "\n".join(flip_details)


@dataclass
class DegradationSensitivityResult:
    """Results of degradation sensitivity analysis (baseline pit-loss only)."""
    
    baseline_model: object
    scenarios: List[StrategyScenarioResult] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary of degradation sensitivity."""
        changes = [s for s in self.scenarios if s.recommendation_changed]
        
        if not changes:
            return "Stable across degradation scenarios"
        
        flip_details = []
        for scenario in changes:
            scenario_type = ""
            if "optimistic" in scenario.scenario_name:
                scenario_type = "(optimistic wear: slower degradation)"
            elif "pessimistic" in scenario.scenario_name:
                scenario_type = "(pessimistic wear: faster degradation)"
            
            flip_details.append(f"  {scenario.scenario_name} {scenario_type}: {scenario.change_details}")
        
        return "Sensitive to degradation scenarios:\n" + "\n".join(flip_details)


@dataclass
class StrategyStabilityAssessment:
    """Complete sensitivity analysis and stability assessment for a strategy."""
    
    baseline_plan: StrategyPlan
    """The baseline (deterministic) recommendation."""
    
    pit_loss_baseline: float
    """Baseline pit-loss value (seconds)."""
    
    pit_loss_sensitivity: PitLossSensitivityResult
    """Pit-loss variation results."""
    
    degradation_sensitivity: DegradationSensitivityResult
    """Degradation variation results."""
    
    stability_label: str
    """Classification: Stable, Moderately Sensitive, or Fragile."""
    
    flip_conditions: List[str] = field(default_factory=list)
    """Conditions under which recommendation changes or flips."""
    
    @property
    def pit_loss_sensitive(self) -> bool:
        """Whether recommendation changes under pit-loss variation."""
        return any(s.recommendation_changed for s in self.pit_loss_sensitivity.scenarios)
    
    @property
    def degradation_sensitive(self) -> bool:
        """Whether recommendation changes under degradation variation."""
        return any(s.recommendation_changed for s in self.degradation_sensitivity.scenarios)
    
    def _compute_stability_label(self) -> str:
        """Compute stability label based on sensitivity results."""
        pit_loss_changes = sum(1 for s in self.pit_loss_sensitivity.scenarios if s.recommendation_changed)
        deg_changes = sum(1 for s in self.degradation_sensitivity.scenarios if s.recommendation_changed)
        
        total_changes = pit_loss_changes + deg_changes
        
        # Classify based on number and severity of changes
        if total_changes == 0:
            return "Stable"
        elif total_changes <= 2:
            return "Moderately Sensitive"
        else:
            return "Fragile"
    
    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        return {
            "baseline_plan": {
                "strategy_type": self.baseline_plan.strategy_type,
                "current_compound": self.baseline_plan.current_compound,
                "next_compound": self.baseline_plan.next_compound,
                "pit_lap": self.baseline_plan.pit_lap,
                "second_pit_lap": self.baseline_plan.second_pit_lap,
                "final_compound": self.baseline_plan.final_compound,
                "total_race_time": round(self.baseline_plan.total_race_time, 2),
                "feasible": self.baseline_plan.feasible,
            },
            "pit_loss_baseline": round(self.pit_loss_baseline, 2),
            "pit_loss_sensitivity": {
                "scenarios": [
                    {
                        "scenario_name": s.scenario_name,
                        "pit_loss_value": round(s.pit_loss_value, 2),
                        "best_plan_type": s.best_plan.strategy_type,
                        "next_compound": s.best_plan.next_compound,
                        "pit_lap": s.best_plan.pit_lap,
                        "total_race_time": round(s.best_plan.total_race_time, 2),
                        "recommendation_changed": s.recommendation_changed,
                        "change_details": s.change_details,
                    }
                    for s in self.pit_loss_sensitivity.scenarios
                ]
            },
            "degradation_sensitivity": {
                "scenarios": [
                    {
                        "scenario_name": s.scenario_name,
                        "scale_factor": round(s.degradation_scale_factor, 2),
                        "best_plan_type": s.best_plan.strategy_type,
                        "next_compound": s.best_plan.next_compound,
                        "pit_lap": s.best_plan.pit_lap,
                        "total_race_time": round(s.best_plan.total_race_time, 2),
                        "recommendation_changed": s.recommendation_changed,
                        "change_details": s.change_details,
                    }
                    for s in self.degradation_sensitivity.scenarios
                ]
            },
            "stability_label": self.stability_label,
            "pit_loss_sensitive": self.pit_loss_sensitive,
            "degradation_sensitive": self.degradation_sensitive,
            "flip_conditions": self.flip_conditions,
        }


def analyze_pit_loss_sensitivity(
    degradation_models: object,
    baseline_pit_loss: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    baseline_plan: StrategyPlan,
) -> PitLossSensitivityResult:
    """Evaluate strategy sensitivity to pit-loss variations.
    
    Tests pit-loss scenarios: baseline ±1.0s, ±2.0s
    
    Args:
        degradation_models: Baseline degradation model (DegradationEvaluationResult)
        baseline_pit_loss: Reference pit-loss value (seconds)
        current_compound: Current tyre compound
        current_tyre_life: Current tyre life in laps
        laps_remaining: Laps remaining in race
        baseline_plan: Baseline recommendation (for comparison)
        
    Returns:
        PitLossSensitivityResult with scenarios and summary
    """
    pit_loss_variations = [
        (baseline_pit_loss - 2.0, "pit_loss_minus_2s"),
        (baseline_pit_loss - 1.0, "pit_loss_minus_1s"),
        (baseline_pit_loss, "pit_loss_baseline"),
        (baseline_pit_loss + 1.0, "pit_loss_plus_1s"),
        (baseline_pit_loss + 2.0, "pit_loss_plus_2s"),
    ]
    
    scenarios = []
    
    for pit_loss_value, scenario_name in pit_loss_variations:
        # Skip baseline (we already have it)
        if abs(pit_loss_value - baseline_pit_loss) < 0.01:
            continue
        
        try:
            best_plan, _ = recommend_best_strategy(
                degradation_models=degradation_models,
                pit_loss_value=pit_loss_value,
                current_compound=current_compound,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
            )
            
            # Determine if recommendation changed
            recommendation_changed = (
                best_plan.strategy_type != baseline_plan.strategy_type or
                best_plan.next_compound != baseline_plan.next_compound or
                best_plan.pit_lap != baseline_plan.pit_lap
            )
            
            change_details = None
            if recommendation_changed:
                baseline_desc = f"{baseline_plan.strategy_type}: {baseline_plan.next_compound} @ L{baseline_plan.pit_lap}"
                new_desc = f"{best_plan.strategy_type}: {best_plan.next_compound} @ L{best_plan.pit_lap}"
                time_delta = best_plan.total_race_time - baseline_plan.total_race_time
                change_details = f"Changed from [{baseline_desc}] to [{new_desc}] ({time_delta:+.2f}s)"
            
            scenario = StrategyScenarioResult(
                scenario_name=scenario_name,
                pit_loss_value=pit_loss_value,
                degradation_scale_factor=1.0,
                best_plan=best_plan,
                recommendation_changed=recommendation_changed,
                change_details=change_details,
            )
            scenarios.append(scenario)
        except Exception as e:
            logger.warning(f"Pit-loss scenario {scenario_name} failed: {e}")
            continue
    
    return PitLossSensitivityResult(
        baseline_pit_loss=baseline_pit_loss,
        scenarios=scenarios,
    )


def analyze_degradation_sensitivity(
    degradation_models: object,
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    baseline_plan: StrategyPlan,
) -> DegradationSensitivityResult:
    """Evaluate strategy sensitivity to degradation assumptions.
    
    Tests degradation scenarios: optimistic (0.7x wear), baseline (1.0x), pessimistic (1.3x wear)
    
    Args:
        degradation_models: Baseline degradation model (DegradationEvaluationResult)
        pit_loss_value: Fixed pit-loss value for this analysis
        current_compound: Current tyre compound
        current_tyre_life: Current tyre life in laps
        laps_remaining: Laps remaining in race
        baseline_plan: Baseline recommendation (for comparison)
        
    Returns:
        DegradationSensitivityResult with scenarios and summary
    """
    degradation_variations = [
        (0.7, "degradation_optimistic"),   # Slower wear
        (1.0, "degradation_baseline"),     # Baseline (for reference)
        (1.3, "degradation_pessimistic"),  # Faster wear
    ]
    
    scenarios = []
    
    for scale_factor, scenario_name in degradation_variations:
        # Skip baseline (we already have it)
        if abs(scale_factor - 1.0) < 0.01:
            continue
        
        try:
            # Create scaled degradation model
            scaled_model = DegradationScaledModel(
                base_model=degradation_models,
                scale_factor=scale_factor,
            )
            
            # Evaluate strategy under scaled degradation
            best_plan, _ = recommend_best_strategy(
                degradation_models=scaled_model,
                pit_loss_value=pit_loss_value,
                current_compound=current_compound,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
            )
            
            # Determine if recommendation changed
            recommendation_changed = (
                best_plan.strategy_type != baseline_plan.strategy_type or
                best_plan.next_compound != baseline_plan.next_compound or
                best_plan.pit_lap != baseline_plan.pit_lap
            )
            
            change_details = None
            if recommendation_changed:
                baseline_desc = f"{baseline_plan.strategy_type}: {baseline_plan.next_compound} @ L{baseline_plan.pit_lap}"
                new_desc = f"{best_plan.strategy_type}: {best_plan.next_compound} @ L{best_plan.pit_lap}"
                time_delta = best_plan.total_race_time - baseline_plan.total_race_time
                change_details = f"Changed from [{baseline_desc}] to [{new_desc}] ({time_delta:+.2f}s)"
            
            scenario = StrategyScenarioResult(
                scenario_name=scenario_name,
                pit_loss_value=pit_loss_value,
                degradation_scale_factor=scale_factor,
                best_plan=best_plan,
                recommendation_changed=recommendation_changed,
                change_details=change_details,
            )
            scenarios.append(scenario)
        except Exception as e:
            logger.warning(f"Degradation scenario {scenario_name} failed: {e}")
            continue
    
    return DegradationSensitivityResult(
        baseline_model=degradation_models,
        scenarios=scenarios,
    )


def assess_strategy_stability(
    baseline_plan: StrategyPlan,
    pit_loss_value: float,
    degradation_models: object,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
) -> StrategyStabilityAssessment:
    """Perform complete sensitivity analysis and compute stability assessment.
    
    Args:
        baseline_plan: The baseline (deterministic) recommendation
        pit_loss_value: Baseline pit-loss value used for recommendation
        degradation_models: Degradation model used for recommendation
        current_compound: Current tyre compound
        current_tyre_life: Current tyre life
        laps_remaining: Laps remaining
        
    Returns:
        StrategyStabilityAssessment with full sensitivity results
    """
    # Analyze pit-loss sensitivity
    pit_loss_sensitivity = analyze_pit_loss_sensitivity(
        degradation_models=degradation_models,
        baseline_pit_loss=pit_loss_value,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        baseline_plan=baseline_plan,
    )
    
    # Analyze degradation sensitivity
    degradation_sensitivity = analyze_degradation_sensitivity(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        baseline_plan=baseline_plan,
    )
    
    # Identify flip conditions
    flip_conditions = []
    
    # Pit-loss flips
    for scenario in pit_loss_sensitivity.scenarios:
        if scenario.recommendation_changed:
            delta_str = f"pit loss ≈ {scenario.pit_loss_value:.1f}s"
            flip_conditions.append(f"Recommendation flips when {delta_str}")
    
    # Degradation flips
    for scenario in degradation_sensitivity.scenarios:
        if scenario.recommendation_changed:
            scenario_label = "pessimistic wear" if "pessimistic" in scenario.scenario_name else "optimistic wear"
            flip_conditions.append(f"Recommendation flips under {scenario_label}")
    
    # Create assessment
    assessment = StrategyStabilityAssessment(
        baseline_plan=baseline_plan,
        pit_loss_baseline=pit_loss_value,
        pit_loss_sensitivity=pit_loss_sensitivity,
        degradation_sensitivity=degradation_sensitivity,
        stability_label="",  # Will be computed below
        flip_conditions=flip_conditions,
    )
    
    # Compute stability label
    assessment.stability_label = assessment._compute_stability_label()
    
    return assessment
