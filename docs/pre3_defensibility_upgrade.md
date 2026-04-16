# Pre-3: Defensibility Upgrade

## Purpose

Pre-3 is a controlled credibility pass on top of Phase 2E. It does three things:

1. makes SOFT confidence more honest
2. replaces the flat blend story with a role-based Miami-anchor design
3. adds one honest held-out Miami backtest

It does not add weather, traffic, safety cars, opponents, or Monte Carlo simulation.

## What was weak before

### SOFT confidence was too implicit

Even after Phase 2E fixed invalid predictions, SOFT could still appear beside MEDIUM and HARD without an explicit statement that support was weaker.

### The blend story was easy to criticize

The old explanation sounded like a flat 40/60 Miami-plus-2026 truth blend. That made it too easy to argue that non-Miami races were being overclaimed as Miami degradation truth.

### There was no held-out Miami backtest

Phase 2D validated representative scenarios, which is useful, but it is not the same thing as checking whether the engine can say something reasonable at real checkpoints in a held-out Miami race.

## What changed

### 1. Explicit support tiers

Each compound now gets an inspectable support summary with:

- raw laps
- model-grade laps
- distinct stints
- distinct races
- active model types by pool
- prediction-health status
- support tier: `High`, `Moderate`, or `Low`

This is saved in:

- `data/processed/pre3_compound_support_summary.json`

Current practical effect:

- MEDIUM and HARD are high-support
- SOFT is moderate-support and is shown that way in the demo/app/reporting

### 2. Role-based hybrid redesign

The modeling path is now:

- Miami-only model = circuit anchor
- 2026 pre-Miami model = recency adjustment/support
- hybrid prediction = Miami prediction plus a bounded recency adjustment

This means the project no longer claims that cross-circuit 2026 rows are Miami truth. They are used only to adjust the anchor in a controlled way.

### 3. Honest held-out Miami backtest

Canonical script:

- `python scripts/run_pre3_backtest.py`

Artifact:

- `data/processed/pre3_backtest_summary.json`

Current design:

- hold out Miami 2024
- train only on earlier Miami years
- exclude 2026 recency from the holdout fit to avoid future leakage
- evaluate several real checkpoints from the held-out race
- compare model-best vs actual remaining strategy

This is intentionally a decision-support backtest, not a fake “predict the race winner” task.

## What remains limited

- SOFT is still not high-support
- the role-based hybrid is still a simple approximation, not a full circuit-transfer model
- the current backtest is a single held-out Miami exercise, not broad historical certification
- no traffic, weather, safety-car, or opponent effects are modeled

## Canonical verification

Run:

```bash
python app/demo_strategy.py
python scripts/run_phase2d_validation.py
python scripts/run_pre3_backtest.py
python -c "from app import streamlit_app; print('streamlit import ok')"
```
