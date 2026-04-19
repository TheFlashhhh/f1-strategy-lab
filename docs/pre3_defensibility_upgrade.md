# Pre-3: Defensibility Upgrade

## Purpose

Pre-3 is a controlled credibility pass on top of Phase 2E. It does three things:

1. makes SOFT confidence more honest
2. replaces the flat blend story with a role-based Miami-anchor design
3. expands the honest held-out Miami backtest
4. adds stop-timing diagnostics so flat optima and timing-heavy misses are visible
5. adds a pace-shape audit so the remaining real misses can be explained more clearly

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
- evaluate a modestly expanded set of 10 real checkpoints from the held-out race
- compare model-best vs actual remaining strategy

Current practical takeaway:

- exact top-3 hits are still `0/10`
- actual next-compound match rate improved to `9/10`
- `4/10` misses are now clearly near-equivalent timing/rank misses
- the remaining `6/10` are the real failures worth inspecting before any Phase 3 work

Miami 2025 was considered for expansion but is still excluded from the canonical backtest because many drivers have unknown compounds through laps `23-24` in the local file, which makes first-stint checkpoint selection unreliable.

This is intentionally a decision-support backtest, not a fake “predict the race winner” task.

### 4. Stop-timing audit and tolerance-aware diagnostics

Canonical scripts:

- `python scripts/run_pre3_backtest.py`
- `python scripts/run_stop_timing_audit.py`

Artifacts:

- `data/processed/pre3_backtest_diagnostics.json`
- `data/processed/pre3_stop_timing_audit.json`

Current practical takeaway:

- exact backtest alignment is still limited
- timing-heavy misses are now easier to separate from real strategy-shape failures
- several one-stop timing curves are flat or sit on the feasible-window edge
- the engine now treats later pit laps inside a 1.0s one-stop timing band as equivalent, rather than always forcing the earliest minimum-time lap

This is a calibration/interpretation improvement, not a claim that timing is now solved.

### 5. Pace-shape audit

Canonical script:

- `python scripts/run_pace_shape_audit.py`

Artifact:

- `data/processed/pre3_pace_shape_audit.json`

Current practical takeaway:

- the repeated long-HARD misses are no longer mainly compound-choice problems after the low-margin SOFT tie-break
- those hard cases still miss badly on stop timing, with the model wanting to stay out until about laps `35-36`
- the strongest remaining real failures are now:
  - two-stop structure disagreements such as `BOT`, `PER`, and `PIA`
  - a soft-finish compound/path disagreement such as `ZHO`, where the real path is not feasible under the current model assumptions

## What remains limited

- SOFT is still not high-support
- the role-based hybrid is still a simple approximation, not a full circuit-transfer model
- the current backtest is still one held-out Miami exercise, not broad historical certification
- long-HARD stop timing still looks under-calibrated even after low-margin SOFT tie-breaks
- several real failures are still structure/compound disagreements rather than pure timing misses
- no traffic, weather, safety-car, or opponent effects are modeled

## Canonical verification

Run:

```bash
python app/demo_strategy.py
python scripts/run_phase2d_validation.py
python scripts/run_pre3_backtest.py
python scripts/run_stop_timing_audit.py
python scripts/run_pace_shape_audit.py
python -c "from app import streamlit_app; print('streamlit import ok')"
```
