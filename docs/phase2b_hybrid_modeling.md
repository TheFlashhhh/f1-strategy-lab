# Phase 2B: Hybrid Modeling

## Status: Complete, with Pre-3 redesign

Phase 2B originally introduced Miami historical data plus 2026 pre-Miami races into the strategy stack. Pre-3 keeps the same source pools but changes the modeling story to a more defensible one:

- Miami historical data is the circuit anchor
- 2026 pre-Miami data is a recency adjustment/support signal
- non-Miami 2026 rows are not presented as direct Miami degradation truth
- pit-loss remains Miami-only

## What changed from the old blend story

The earlier implementation used sample-level replication/downsampling and could be read as a flat cross-circuit truth pool. That was easy to criticize because Australia, China, and Japan are not Miami.

The current implementation instead fits:

1. a Miami-only degradation source
2. a 2026 pre-Miami degradation source
3. a role-based hybrid predictor that starts from the Miami prediction and applies only a bounded recency adjustment

This is still a simplification, but it is much easier to defend:

- Miami remains the source of circuit-specific truth
- 2026 data can move the prediction, but only within a controlled adjustment path
- the UI and docs no longer overclaim that the cross-circuit pool is itself Miami truth

## Current design

### Pool roles

| Pool | Role | Used for |
|------|------|----------|
| `miami_historical` | `miami_anchor` | circuit-specific degradation anchor and pit-loss calibration |
| `season_2026_pre_miami` | `recency_adjustment` | bounded pace/degradation adjustment and support signal |

### Prediction path

For each compound:

1. predict from the Miami anchor model
2. predict from the 2026 recency model
3. compute a bounded adjustment from the recency delta
4. apply only part of that delta, with the adjustment strength controlled by support tier

Higher-support compounds get smaller recency movement. Lower-support compounds can move more, but still through a bounded adjustment rather than a flat truth-pool blend.

## Support-aware behavior

Pre-3 adds explicit compound support tiers:

- `High`
- `Moderate`
- `Low`

The tier is based on explicit counts such as:

- Miami model-grade laps
- total model-grade laps across Miami + 2026 pools
- model-grade stints
- Miami race coverage
- prediction validity

This matters most for SOFT, which is still not treated as equally trustworthy as MEDIUM/HARD.

## Artifacts

- `data/processed/phase2b_data_summary.json`
- `data/processed/pre3_compound_support_summary.json`

The first describes the active pool roles. The second adds the compound support tiers and support counts.

## What remains limited

- This is still deterministic and simple
- non-Miami 2026 races are still an approximation, even with bounded adjustment
- the role-based hybrid is more defensible than the old flat blend, but it is not a full circuit-transfer model
- no weather, traffic, safety-car, or opponent effects are modeled
