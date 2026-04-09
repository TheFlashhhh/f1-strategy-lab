# Phase 1B: Fuel Correction Implementation

## Overview

Phase 1B adds a fuel-correction layer to remove the confounding effect of fuel-load changes on lap-time measurements before fitting tyre-degradation models.

**Problem:** As a race progresses, fuel load decreases, making cars faster due to reduced weight. This residual speed advantage masks the true tyre degradation effect, confounding degradation estimates.

**Solution:** Estimate fuel-burn time effect and create fuel-corrected lap times normalized to a common fuel baseline (e.g., full tank).

---

## Methodology

### 1. Fuel Effect Estimation

**Input:** Model-grade laps (pit-excluded, accurate, green-flag only)

**Approach:**
- Normalize lap number to race progress (0–1) per driver
- Fit linear regression: $\text{LapTime} \sim \text{RaceProgress}$ per compound
- Extract slope as fuel-effect coefficient (in seconds per unit race progress, i.e., seconds across full race)
- Negative slope = improvement as race progresses (fuel burns, car gets lighter)

**Rationale:**
- Fuel load decreases monotonically as race progresses
- Lap time improvement from fuel burn should correlate with race progress
- Linear approximation is tractable and interpretable
- Compound-specific estimation captures different fuel strategies

**Note on Units:**
The coefficient is NOT "seconds per lap". Since race progress is normalized to [0, 1], 
the coefficient is "seconds per unit race progress", which equals "total seconds of improvement across the entire race".

**Example Coefficients (Miami 2022–2025):**
| Compound | Fuel Effect | Interpretation |
|---|---|---|
| MEDIUM | -2.33 s/race | 2.33 seconds faster at end of race vs. start due to fuel burn |
| HARD | -1.68 s/race | 1.68 seconds faster at end due to fuel burn |
| SOFT | -0.59 s/race | 0.59 seconds faster (high uncertainty; only 33 samples) |

All negative values indicate faster lap times as the race progresses.

### 2. Fuel Correction Application

**Formula:**
$$\text{FuelCorrectedLapTime} = \text{LapTime} - (\text{fuel\_effect} \times \text{race\_progress})$$

Where:
- `race_progress` ∈ [0, 1] per driver per race (0 = first lap in stint, 1 = last lap in stint)
- `fuel_effect` is the coefficient from Step 1 (negative = lap times improve as race progresses)
- Subtracting the fuel effect "adds back" the time advantage from lower fuel load, normalizing to full-tank equivalent

**Effect:**
- Early-race laps (low progress) → no or small correction
- Late-race laps (high progress) → larger correction
- Result: Normalized lap times as if all laps driven with equivalent fuel load

### 3. Degradation Model Comparison

**Before fuel correction:**
- Degradation slope = true degradation + fuel-load improvement
- Confounded estimate

**After fuel correction:**
- Degradation slope = true degradation only
- Cleaner, more interpretable estimate

---

## Implementation Details

### Files Added

1. **`src/features/fuel_correction.py`** — Fuel correction module with:
   - `estimate_fuel_effect()` — Estimate fuel-burn coefficients
   - `apply_fuel_correction()` — Apply correction to laps
   - `evaluate_fuel_correction()` — Compare before-vs-after degradation models
   - `save_fuel_correction_summary()` — Export evaluation JSON
   - `print_fuel_correction_summary()` — Human-readable report

2. **`app/demo_phase1b.py`** — End-to-end demonstration script

### Files Modified

1. **`src/features/build_features.py`**
   - Added import of fuel correction functions (optional; gracefully skipped if not used)
   - Added `use_fuel_corrected` parameter to `create_degradation_table()` for backward compatibility
   - Imports fuel correction but preserves existing API

### Integration Pattern

```python
# Old pipeline (Phase 1A)
model_df = build_model_df(clean_df)
degradation_table = create_degradation_table(model_df)
models = fit_degradation_models(degradation_table)

# New pipeline (Phase 1B-aware)
model_df = build_model_df(clean_df)

# Optional: Apply fuel correction
fuel_effects = estimate_fuel_effect(model_df)
corrected_df = apply_fuel_correction(model_df, fuel_effects)

# Use corrected laps for degradation
degradation_table = create_degradation_table(corrected_df, use_fuel_corrected=True)
models = fit_degradation_models(degradation_table)
```

---

## Results on Miami 2022–2025

### Fuel Effects Estimated

| Compound | Coefficient (s/race) | Sample Count | Interpretation |
|---|---|---|---|
| SOFT | -0.586 | 33 | High uncertainty (few laps) |
| MEDIUM | -2.330 | 691 | Moderate confidence |
| HARD | -1.683 | 1325 | Good confidence (many laps) |

**Key observation:** MEDIUM improves more over a race than HARD, suggesting fuel management strategy varies by compound.

### Degradation Slope Changes (Before → After)

| Compound | Raw Slope | Corrected Slope | Slope Change | % Change |
|---|---|---|---|---|
| SOFT | -0.0690 s/lap | -0.0511 s/lap | +0.0179 | -26% |
| MEDIUM | -0.0456 s/lap | +0.0143 s/lap | +0.0599 | -131% |
| HARD | -0.0021 s/lap | +0.0289 s/lap | +0.0303 | -1450% |

**Interpretation:**
- **SOFT:** Minimal change (26%), suggesting fuel effect relatively small for soft compound
- **MEDIUM:** Dramatic sign flip (negative → positive), indicating raw slope was heavily confounded; true degradation is positive not negative
- **HARD:** Massive percentage change (1450%), but note raw slope was very close to zero; fuel effect dominated

### Example Lap Corrections (MEDIUM Compound)

```
Driver  LapNumber  LapTime  FuelCorrectedLapTime  Correction  TyreLife
   ALB          1   94.158          94.158         0.000        1
   ALB          5   93.842          94.200        -0.358        5
   ALB         10   93.456          94.523        -1.067        10
   ALB         15   93.234          94.750        -1.516        15
   ALB         20   93.012          94.977        -1.965        20
```

**Pattern:** Later laps in race marked by larger fuel corrections, as expected.

---

## Filtering & Data Quality

### Filtering Rules Applied

- **Pit stops:** Excluded (confounded by pit strategy)
- **Tyre life:** > 2 laps (exclude unreliable early-stint data)
- **Accuracy:** IsAccurate == True (exclude inaccurate timing)
- **Validity:** NotDeleted == True (exclude deleted laps)
- **Track status:** Green flag (1) only (exclude weather/incident-affected laps)

### Sample Counts

- Total Miami 2022–2025: **4,311 laps**
- After filtering: **2,049 model-grade laps**
- Used for fuel estimation:
  - SOFT: 33 laps (1.6%)
  - MEDIUM: 691 laps (33.7%)
  - HARD: 1,325 laps (64.6%)

---

## Assumptions & Limitations

### Assumptions

1. **Linear fuel effect:** Fuel burn time effect changes linearly with race progress
2. **Per-compound fuel effects:** Different compounds have different fuel sensitivities (not merged)
3. **Full race normalization:** All correction normalized to full-tank baseline
4. **No traffic effects:** Assumes consistent track conditions; traffic not explicitly modeled

### Limitations

1. **Linear model assumption:** Assumes fuel-burn effect is constant per unit race progress
   - Reality: Early-race fuel mass changes more impactful on lap time than late-race changes
   - Status: Not validated against telemetry; first-order approximation

2. **Temperature confounding:** Warmer tyres late in race also make cars faster
   - Model cannot separate fuel-load improvement from tyre-temperature improvement
   - Both contribute to observed late-race speed advantage
   - Status: Known limitation; no temperature data available in dataset

3. **No uncertainty quantification:** Reports point estimates only
   - No confidence intervals on fuel coefficients
   - Example: SOFT compound (n=33) estimates are noisy but appear as point values
   - Status: Users should weight estimates by sample count when using results

4. **Circuit and era specificity:** Calibrated only on Miami 2022–2025
   - Fuel effects may differ significantly on longer/shorter circuits
   - May differ across regulation eras (DRS era vs. 2026 active-aero)
   - Not validated on other races or tracks
   - Status: Generalization to other circuits untested

5. **Sample size disparities:** Varies by 40x across compounds
   - HARD: 1,325 laps (stable estimate)
   - MEDIUM: 691 laps (moderate stability)
   - SOFT: 33 laps (high variance, unreliable)
   - Status: Results reported uniformly despite variance differences

6. **Unaccounted factors:**
   - Safety-car cycles reduce fuel burn rates (fuel-saving effects not modeled)
   - Track evolution (grip changes, rubber buildup over race)
   - Driver-specific fuel-saving techniques
   - Tyre temperature loss before pit stops

---

**User Responsibility:** Validate corrected lap times before using in production strategy:
- Check if corrected degradation slopes pass sanity checks
- Compare corrected pit timing predictions against ground truth race outcomes
- Monitor whether corrected-based strategy decisions improve win probability

---

## Integration with Existing Pipeline

### Backward Compatibility

- **Original demo still works:** `python app/demo_strategy.py` runs unchanged, uses raw lap times
- **Strategy.py unchanged:** Strategy optimization and pit-loss estimation use raw times by default
- **Opt-in fuel correction:** If adopted, strategy fitting would use corrected times

### Where to Use Fuel-Corrected Times

For **tyre-degradation modeling**: Use `FuelCorrectedLapTime` instead of `LapTime`
```python
degradation_table = create_degradation_table(corrected_df, use_fuel_corrected=True)
```

For **pit-loss estimation**: Consider using corrected times to avoid confounding pit loss with fuel effects

For **strategy optimization**: Requires re-fitting degradation models on corrected data, then re-tuning pit-window thresholds

---

## Output Artifacts

### `data/processed/fuel_correction_summary.json`

Complete evaluation saved in JSON format:

```json
{
  "method": "Linear degradation: LapTime ~ TyreLife",
  "fuel_correction_applied": true,
  "raw_sample_count": 4311,
  "model_sample_count": 2049,
  "fuel_effects": {
    "MEDIUM": {
      "coefficient_s_per_full_race": -2.3302,
      "interpretation": "seconds of total lap-time improvement over the entire race due to fuel burn",
      "sample_count": 691
    }
  },
  "degradation_comparison": {
    "MEDIUM": {
      "raw_slope_s_per_lap": -0.0456,
      "raw_intercept_s": 95.123,
      "corrected_slope_s_per_lap": 0.0143,
      "corrected_intercept_s": 94.890,
      "slope_change_s_per_lap": 0.0599,
      "intercept_change_s": -0.233,
      "slope_change_percent": -131.4,
      "sample_count": 691
    }
  }
}
```

### Console Output

`demo_phase1b.py` prints human-readable summary:
- Estimated fuel effects per compound
- Before vs after degradation slopes
- Impact percentages

---

## Next Steps (Phase 1C & Beyond)

### Phase 1C: Degradation Modeling Improvements
- **Piecewise degradation:** Detect cliff points where degradation accelerates
- **Per-season trends:** Model differences between 2022–2025 regulations changes
- **Uncertainty quantification:** Confidence intervals on degradation estimates

### Phase 2: Multi-Circuit Generalization
- Apply fuel correction to other circuits (Bahrain, Singapore, Monaco)
- Test whether fuel coefficients generalize or circuit-specific
- Build regulation-era-specific models (DRS vs. active-aero)

### Phase 3: Stochastic & Multi-Agent Strategy
- Incorporate uncertainty in degradation and pit-loss estimates
- Multi-agent game-theoretic pit timing
- Traffic and safety-car interaction modeling

---

## Running the Demo

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Run fuel correction demo
python app/demo_phase1b.py

# Inspect results
cat data/processed/fuel_correction_summary.json

# Verify backward compatibility
python app/demo_strategy.py
```

---

## References

This implementation is inspired by standard F1 engineering practice of correcting for fuel load changes when analyzing performance deltas. See phase1_data_plan.md and phase1a_summary.md for context on Phase 1A groundwork.
