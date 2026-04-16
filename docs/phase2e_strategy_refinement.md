# Phase 2E: Strategy Search Refinement / Calibration

## Status: Complete

Phase 2E is a calibration pass on the existing strategy stack. It does not add weather, safety cars, traffic, opponents, or Monte Carlo simulation. It fixes the highest-value weaknesses exposed by Phase 2D and reruns the representative validation harness to measure the effect.

## What Changed

### 1. Race-context leakage fixed

The hybrid pipeline contains multiple races for the same driver. Several core steps were still grouping only by `Driver`, which stitched separate events together and distorted:

- pit-stop detection
- fuel-progress normalization
- pit-loss estimation

Phase 2E makes those steps group by available race context plus driver, so each race is treated independently.

### 2. Pit-loss calibration fixed

Pre-Phase-2E, pit loss collapsed to `0.00s`. The main causes were:

- cross-race grouping leakage
- estimating pit loss from the blended hybrid pool instead of a Miami-specific pit-loss baseline
- treating the pit window too loosely in a way that admitted degenerate samples

Phase 2E now:

- estimates pit loss from `miami_historical`
- uses race-local pit-stop grouping
- keeps representative baseline laps clean
- drops non-positive pit-loss samples

Current Miami median pit loss:

- `14.34s`

This is materially more credible than `0.00s` because it is no longer produced by stitched multi-race driver histories.

### 3. SOFT model-health handling improved

Pre-Phase-2E, SOFT was reported as active but often returned invalid predictions. The root issue was upstream grouping leakage collapsing SOFT support and corrupting fuel-progress normalization.

Phase 2E improves this by:

- fixing race-local grouping
- restoring valid SOFT sample support in the model-grade dataset
- preventing invalid strategy scoring from silently using fake fallback laps
- keeping SOFT honest through feasibility and model-info reporting

Result:

- SOFT probe predictions are now valid in the active pipeline
- the Phase 2D artifact no longer reports SOFT prediction health as invalid

### 4. Bounded two-stop search refinement

Pre-Phase-2E, two-stop plans used a rough fixed heuristic around one-third and two-thirds distance.

Phase 2E replaces that with:

- a bounded search across valid first/second pit-lap pairs
- minimum-stint constraints to keep the search tractable
- invalid-plan removal when a required lap prediction is missing

This is still a compact deterministic search, not a full race simulator.

### 5. Piecewise model acceptance tightened

Phase 2E also prevents weak or non-cliff piecewise fits from being treated as real cliffs by:

- requiring a minimum fit-improvement threshold
- falling back to linear when the post-cliff slope does not worsen
- guarding against artificial pace gains from discontinuous post-breakpoint intercept jumps

## Phase 2D Before vs After

Using the same representative Phase 2D scenario suite:

| Metric | Before Phase 2E | After Phase 2E |
|--------|------------------|----------------|
| Pit-loss baseline | `0.00s` | `14.34s` |
| One-stop count | `12` | `10` |
| Two-stop count | `0` | `2` |
| Stable | `8` | `9` |
| Moderately Sensitive | `4` | `3` |
| Fragile | `0` | `0` |
| SOFT weak-data signal | `true` | `false` |

## What Remains Limited

- Phase 2E is still deterministic
- no traffic, safety car, weather, or opponent modeling was added
- validation is still representative-scenario evaluation, not historical backtesting
- hybrid modeling still depends on cross-race blending assumptions from Phase 2B

## Canonical Checks

Run:

```bash
python app/demo_strategy.py
python scripts/run_phase2d_validation.py
python -c "from app import streamlit_app; print('streamlit import ok')"
```
