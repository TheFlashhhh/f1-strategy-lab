# Phase 1 Specification: Trustworthy Lap-Time Prediction

## Objective

Build lap-time prediction trustworthy enough for confident strategy search. Current linear degradation model is suitable for exploration but lacks precision for optimization. Phase 1 adds fuel correction, representative-lap filtering, piecewise degradation, and used-set history.

---

## Current Baseline

**Implemented:**
- Lap-level race data ingestion (CSV datasets with driver, lap time, compound, tyre life, stint, pit, track status)
- Pit-stop detection via stint changes
- Lap cleaning and green-flag filtering
- Linear compound-specific degradation: `LapTime ~ slope × TyreLife + intercept`
- Empirical pit-loss estimation (~15.45s median for Abu Dhabi 2020)
- Deterministic pit-window optimization (exhaustive search, min-time policy)
- Modular Python pipeline: preprocess → feature → simulate → UI
- Streamlit interactive interface

**Not yet implemented:**
- Fuel-burn separation (conflated with degradation)
- Tyre-cliff detection (linear only)
- Used-set history (binary new/used only)
- Traffic-aware pace suppression
- Undercut modeling (warmup, outlap, rejoin penalties)
- Regulation enforcement (fuel limits, compound mandates, minimum stints)
- Opponent/multi-agent logic
- Live ingestion
- Uncertainty/stochastic modeling

---

## 1. Fuel Correction

**Problem:** Lap time = baseline + fuel_effect(m) + tyre_degradation(L). Current model conflates these, biasing degradation slopes high.

**Approach:** Physically informed prior + empirical fit.
- Prior: 0.03–0.05 s/kg per lap (from aero physics)
- Typical stint fuel swing: ~1.2–2.0 seconds
- Compute fuel load per lap; fit small empirical correction
- Avoid pure regression (collinearity with tyre life)

**Validation:** Corrected early/late slope ratio should align; slope variance <10% across stints; pit-loss estimate remains consistent.

---

## 2. Representative-Lap Filtering

**Definition:** Laps unaffected by pit stops, traffic, weather, or other pace suppressors.

**Exclude:**
- Pit-in, pit-out, pit-stop flag rows
- Deleted/inaccurate laps
- Non-green-flag (yellow/SC/VSC)
- First 2 laps of stint (warmup, cold tyres)
- Outliers (>3σ per compound)
- Traffic proxy if gap-ahead available (e.g., gap <1s)

**Fallback:** If gap-ahead unavailable, disable traffic filter; document limitation.

---

## 3. Degradation and Cliff Modeling

**Current:** Linear per-compound fit.

**Phase 1:** Piecewise (segmented) degradation with cliff detection.

$$\text{LapTime} = \begin{cases} \text{early\_slope} \cdot L + \text{intercept} & L < L_{\text{cliff}} \\ \text{early\_slope} \cdot L_{\text{cliff}} + \text{intercept} + \text{penalty} + \text{late\_slope} \cdot (L - L_{\text{cliff}}) & L \geq L_{\text{cliff}} \end{cases}$$

**Requirements:**
- Cliff detection only on stints with ≥10 representative laps
- Breakpoints estimated on representative laps only
- Residual validation: piecewise MSE ≥5% improvement over linear
- Compound-specific cliff ages

---

## 4. Used-Set Feature Modeling

**Problem:** Binary new/used is insufficient. Used tyres vary based on prior laps, stint count, and history.

**Feature schema:**
```python
{
  "compound": str,                      # SOFT, MEDIUM, HARD
  "total_prior_laps": int,              # Total laps before this stint
  "prior_stint_count": int,             # Number of previous stints
  "current_stint_laps": int,            # Laps into current stint
}
```

**Impact areas:** Starting pace (0.5–1.0s slower per reuse), degradation slope (flatter or steeper), cliff location (earlier due to wear).

**Integration:** Regress `LapTime ~ f(compound, L, prior_laps, stint_count)` to separate fresh vs. used vs. very-old pace estimates.

---

## 5. Undercut Evaluation

**Problem:** Current model has no multi-car undercut logic.

**Components:**
- Gap to target car
- Pit loss (~15.45s)
- Fresh-tyre pace gain
- Warmup/outlap/rejoin penalties

**Pace-gain curve** (indexed by post-pit lap-in-stint):
```python
pace_gain_curve = [-1.0, 0.5, 1.0, 1.2, ...]  # warm-up ramp-up
```

**Success criterion:** Net pace gain from curve > pit loss + traffic cost vs. gap to target.

**Phase 1 scope:** Define curve empirically; implement basic evaluator; validate on sanity checks (e.g., gap=12s, pit_loss=15.45s with 4s gains should succeed).

---

## Success Criteria

1. ✅ Fuel-corrected model reproducible; empirical fit within ±0.01 s/kg of prior
2. ✅ Filtering rules explicit; no hidden data loss (audited)
3. ✅ Refitted degradation slopes vary <10% across stints
4. ✅ Piecewise models improve MSE ≥5%; residuals symmetric/normal
5. ✅ Used-set features integrated; regression R² > 0.80 (variance explained)
6. ✅ Undercut evaluator passes sanity checks; pace-gain curve empirically justified
7. ✅ Full pipeline on Abu Dhabi 2020 reproduces baseline + improvements

---

## Failure Criteria

1. ❌ Fuel-corrected slopes still vary >10% (collinearity or data issues)
2. ❌ Cliff breakpoints <5 laps in dry context (overfitting/misalignment)
3. ❌ Piecewise MSE doesn't improve or residuals worsen
4. ❌ Representative-lap filtering retains <50% of laps (too aggressive)
5. ❌ Undercut model fails obvious scenarios (gap=12s, pit_loss=15.45s, gains=4s should succeed)
6. ❌ Used-set regression R² < 0.80 (features not capturing variation)

**Any one constitutes Phase 1 block.** Resolve before proceeding to Phase 2 (traffic, opponent modeling).

---

## Roadmap

| Phase | Focus |
|-------|-------|
| **Phase 0** | Baseline linear degradation, pit optimization ✓ |
| **Phase 1** | Fuel correction, rep-laps, cliffs, used-sets, undercuts |
| **Phase 2** | Traffic-aware pace, multi-agent strategy |
| **Phase 3** | Regulation enforcement, stochastic simulation |
| **Phase 4** | Live ingestion, real-time, opponent modeling |

---

## Summary

Phase 1 upgrades F1 Strategy Lab from exploratory tool to trustworthy modeling engine. Rigorous validation is essential: slopes must stabilize, cliffs must appear at plausible ages, undercuts must pass sanity checks. Only after Phase 1 should development venture into traffic, opponents, and live racing.
