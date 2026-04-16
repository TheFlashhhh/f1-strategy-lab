# Phase 2C: Strategy Sensitivity & Uncertainty Analysis

## Status: ✅ COMPLETE

Phase 2C adds **scenario-based sensitivity analysis** to evaluate the stability of pit strategy recommendations under reasonable variations in pit-loss and degradation assumptions.

---

## What Phase 2C Solves

### The Brittle Strategy Problem (Phase 2A Baseline)
**Problem:** Phase 2A provides a deterministic recommendation based on point estimates:
- ✓ **Advantage:** Clear, single best answer
- ✗ **Disadvantage:** No insight into recommendation robustness
- ✗ **Disadvantage:** Unknown how much the recommendation would change if assumptions were different
- ✗ **Disadvantage:** Hides uncertainty and fragile decision points

### Phase 2C Solution: Scenario-Based Sensitivity Analysis
**Approach:** For the baseline recommendation, evaluate what happens under reasonable variations:

1. **Pit-Loss Sensitivity:** Test ±1.0s, ±2.0s from baseline pit-loss estimate
   - Identifies if a small pit-timing measurement error would flip the recommendation
   - Reports which alternatives emerge under different pit-loss assumptions

2. **Degradation Sensitivity:** Test optimistic (0.7x wear), baseline (1.0x), pessimistic (1.3x wear)
   - Identifies if pessimistic wear assumptions would change strategy
   - Identifies if optimistic assumptions are prerequisites for the recommendation

3. **Stability Classification:** Label recommendations as:
   - **Stable:** No recommendation changes across any scenario
   - **Moderately Sensitive:** 1-2 changes across scenarios
   - **Fragile:** 3+ changes or multiple flip points

4. **Flip Condition Identification:** Report specific conditions where recommendation changes

---

## Why This Matters

### Example 1: Stable Recommendation
```
Baseline: ONE-STOP to HARD @ L22
Stability: Stable

Pit-Loss Sensitivity:
  -2.0s: ONE-STOP to HARD @ L22 [same]
  -1.0s: ONE-STOP to HARD @ L22 [same]
  +1.0s: ONE-STOP to HARD @ L22 [same]
  +2.0s: ONE-STOP to HARD @ L22 [same]

Conclusion: Recommendation is robust; small errors in pit-loss don't matter
```

### Example 2: Fragile Recommendation
```
Baseline: TWO-STOP HARD->SOFT @ L10,L18
Stability: Fragile

Pit-Loss Sensitivity:
  -2.0s: ONE-STOP SOFT @ L23 [FLIPPED!]
  -1.0s: ONE-STOP SOFT @ L23 [FLIPPED!]
  +1.0s: TWO-STOP HARD->SOFT @ L10,L18 [same]
  +2.0s: ONE-STOP MEDIUM @ L20 [FLIPPED!]

Conclusion: Recommendation depends heavily on pit-loss accuracy;
           small errors could change strategy significantly
```

---

## Design & Implementation

### Key Principles
1. **Scenario-based, NOT probabilistic**
   - We explicitly test specific scenarios (e.g., pit-loss -2.0s)
   - We do NOT run Monte Carlo or assume probability distributions
   - We are honest: this is sensitivity, not full uncertainty quantification

2. **Inspectable and auditable**
   - All scenarios are documented
   - Scale factors (0.7x, 1.0x, 1.3x) are explicit and reasonable
   - No hidden heuristics or black-box uncertainty models

3. **Lightweight and composable**
   - Runs on top of existing strategy engine
   - No changes to core degradation/pit loss logic
   - Easy to extend with additional scenarios or variations

### Module: `src/simulation/strategy_sensitivity.py`

**Key Classes:**

#### `StrategyScenarioResult`
Represents one sensitivity test scenario:
- `scenario_name`: Label (e.g., "pit_loss_plus_2s")
- `pit_loss_value`: Pit loss used in this scenario
- `degradation_scale_factor`: Degradation rate multiplier
- `best_plan`: Recommended strategy under this scenario
- `recommendation_changed`: Whether recommendation differs from baseline
- `change_details`: Description if changed

#### `DegradationScaledModel`
Wrapper to apply degradation scaling to baseline model:
- Allows unified interface for original, optimistic, and pessimistic models
- Uses simple scaling of degradation delta: `scaled = baseline + (delta * scale_factor)`

#### `PitLossSensitivityResult`
Results from pit-loss sensitivity analysis:
- `baseline_pit_loss`: Reference value
- `scenarios`: List of `StrategyScenarioResult` for each pit-loss test
- `summary()`: Human-readable summary of changes

#### `DegradationSensitivityResult`
Results from degradation sensitivity analysis:
- `baseline_model`: Reference degradation model
- `scenarios`: List of `StrategyScenarioResult` for each degradation test
- `summary()`: Human-readable summary of changes

#### `StrategyStabilityAssessment`
Complete assessment combining both analyses:
- `baseline_plan`: The deterministic recommendation
- `pit_loss_sensitivity`: Full pit-loss analysis results
- `degradation_sensitivity`: Full degradation analysis results
- `stability_label`: "Stable", "Moderately Sensitive", or "Fragile"
- `flip_conditions`: List of conditions where recommendation change
- `to_dict()`: Serialize to JSON (for artifacts and logging)

### Key Functions

#### `analyze_pit_loss_sensitivity()`
Evaluates strategy across pit-loss variations: baseline ±1.0s, ±2.0s

#### `analyze_degradation_sensitivity()`
Evaluates strategy across degradation scales: 0.7x, 1.0x, 1.3x

#### `assess_strategy_stability()`
Main entry point; coordinates both analyses and produces full assessment

---

## Integration

### Demo Integration (`app/demo_strategy.py`)
Phase 2C demo output:
```
================================================================================     
PHASE 2C: STRATEGY SENSITIVITY & UNCERTAINTY ANALYSIS
================================================================================

[Baseline Recommendation]
  Strategy: ONE-STOP
  Next Tyre: HARD
  Pit Lap: 22
  Total Time: 2342.25s

[Stability Assessment]
  Label: Stable

[Pit-Loss Sensitivity] (baseline: 0.00s)
  Status: STABLE - recommendation unchanged under ±1-2s variation

[Degradation Sensitivity] (baseline model)
  Status: STABLE - recommendation unchanged across degradation scenarios

[Flip Conditions]
  • None identified - recommendation is robust
```

### Streamlit Integration (`app/streamlit_app.py`)
New section: **"🔍 Recommendation Stability (Phase 2C)"**
- Displays stability label with metrics
- Pit-loss sensitivity table
- Degradation sensitivity table
- Flip conditions warnings

### Artifact: `data/processed/phase2c_sensitivity_summary.json`
Complete serialized sensitivity assessment including:
- Baseline recommendation
- All pit-loss scenarios
- All degradation scenarios
- Stability label
- Flip conditions

---

## Configuration & Tuning

### Pit-Loss Variations
Current: baseline ±1.0s, ±2.0s

To change, edit `analyze_pit_loss_sensitivity()`:
```python
pit_loss_variations = [
    (baseline_pit_loss - 2.0, "pit_loss_minus_2s"),
    (baseline_pit_loss - 1.0, "pit_loss_minus_1s"),
    (baseline_pit_loss, "pit_loss_baseline"),
    (baseline_pit_loss + 1.0, "pit_loss_plus_1s"),
    (baseline_pit_loss + 2.0, "pit_loss_plus_2s"),
]
```

### Degradation Scale Factors
Current: 0.7x (optimistic), 1.0x (baseline), 1.3x (pessimistic)

To change, edit `analyze_degradation_sensitivity()`:
```python
degradation_variations = [
    (0.7, "degradation_optimistic"),   # Slower wear
    (1.0, "degradation_baseline"),     # Baseline
    (1.3, "degradation_pessimistic"),  # Faster wear
]
```

### Stability Classification Thresholds
Current:
- **Stable:** 0 changes
- **Moderately Sensitive:** 1-2 changes
- **Fragile:** 3+ changes

To change, edit `_compute_stability_label()` in `StrategyStabilityAssessment`

---

## Limitations & Future Work

### Explicit Limitations
1. **NOT fully probabilistic** - We test scenarios, not probability distributions
2. **NOT accounting for correlation** - Pit-loss and degradation assumed independent
3. **NOT considering traffic/opponents** - Same as Phase 2A
4. **NOT adaptive** - No response to dynamic race evolution

### What This Analysis Does NOT Provide
- Probability that recommendation is correct
- Confidence intervals on strategy times
- Joint sensitivity (pit-loss AND degradation correlation)
- Safety car scenarios
- Traffic models

### Why: Scope Control
Phase 2C is intentionally lightweight and scenario-based:
- Adds transparency without re-architecting the strategy engine
- Identifies brittle decision points without full Monte Carlo
- Supports decision-making without claiming unrealistic precision
- Honest about limitations

### Future Enhancements (Out of Scope for Phase 2C)
1. Monte Carlo uncertainty quantification (would need probabilistic model of uncertainties)
2. Traffic-aware sensitivity (would require traffic/overtaking models)
3. Safety car response scenarios (would require race event modeling)
4. Joint pit-loss & degradation correlation (would need empirical correlation analysis)
5. Time-series sensitivity (feedback loops within race)
6. Driver-specific degradation profiles (would need driver classification)

---

## Usage Examples

### Command-Line Demo
```bash
python app/demo_strategy.py
```
Outputs Phase 2C analysis to stdout + saves `data/processed/phase2c_sensitivity_summary.json`

### Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Navigate to "🔍 Recommendation Stability (Phase 2C)" section

### Programmatic Usage
```python
from src.simulation.strategy_sensitivity import assess_strategy_stability

# After building baseline recommendation and models:
stability = assess_strategy_stability(
    baseline_plan=best_plan,
    pit_loss_value=pit_loss_value,
    degradation_models=deg_result,
    current_compound="MEDIUM",
    current_tyre_life=5,
    laps_remaining=25,
)

# Access results
print(f"Stability: {stability.stability_label}")
print(f"Pit-loss sensitive: {stability.pit_loss_sensitive}")
print(f"Degradation sensitive: {stability.degradation_sensitive}")

# Serialize to JSON
import json
with open("sensitivity_report.json", "w") as f:
    json.dump(stability.to_dict(), f, indent=2)
```

---

## Verification Checklist

✅ Phase 2C module created (`src/simulation/strategy_sensitivity.py`)
✅ Demo integration complete (`app/demo_strategy.py`)
✅ Streamlit integration complete (`app/streamlit_app.py`)
✅ Artifact saved to JSON (`data/processed/phase2c_sensitivity_summary.json`)
✅ Documentation complete (this file)
✅ No weather/thermal modeling added (correctly out-of-scope)
✅ No safety cars/traffic modeling added (correctly out-of-scope)
✅ Scenario-based, not probabilistic (honest about limitations)
✅ Inspectable and auditable (explicit scenarios, not black-box)

---

## References

- [Phase 2A Strategy Engine](phase2a_strategy_search.md) - Baseline recommendation
- [Phase 2B Hybrid Modeling](phase2b_hybrid_modeling.md) - Data blending context
- [Phase 1 Pipeline](PHASE_1_INTEGRATION.md) - Core analysis pipeline
