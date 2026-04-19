# Pre-3: Pace-Shape Audit

## Purpose

This audit is the last deterministic-core check before any Phase 3 work.

The narrower question is:

- are the remaining real backtest misses still mostly stop-timing noise
- or do they come from bad strategy structure / compound pace-shape assumptions

## Canonical scripts

- `python scripts/run_pre3_backtest.py`
- `python scripts/run_pace_shape_audit.py`

## Artifacts

- `data/processed/pre3_backtest_diagnostics.json`
- `data/processed/pre3_pace_shape_audit.json`

## What the pace-shape audit adds

For each remaining real backtest miss, the audit compares:

- the model-best remaining strategy
- the actual executed remaining strategy
- predicted stint-by-stint lap-time progression
- observed clean-lap progression inside the held-out race

It reports:

- dominant issue
- mismatch source
- whether the current-stint shape looks too flat
- whether the miss is mainly:
  - timing-only
  - structure + compound
  - or a stronger pace-shape / feasibility gap

The artifact also includes a small representative probe set from the Phase 2D suite so the current deterministic core can be inspected outside the held-out backtest.

## Current practical findings

- the expanded backtest now has `10` Miami 2024 checkpoints
- `4/10` misses are near-equivalent timing/rank misses
- `6/10` are the real failures that still matter
- the repeated long-HARD cases now look more like stop-timing failures than pure compound-choice failures
- the clearest remaining structure/compound misses are the real two-stop disagreements and the soft-finish `ZHO` case

## What this means

The current deterministic core is more interpretable than before, but it is still not strong enough to build Phase 3 on top of yet.

The current state is:

- timing precision is still weak on several flat or edge-bound one-stop curves
- the backtest evidence is better than before, but still not strong
- the remaining failures are no longer just one problem:
  - some are long-stint timing calibration misses
  - some are real structure/compound disagreements

## Readiness implication

This pass improves honesty more than headline performance.

That is useful, but it still means:

- Phase 3 should wait
- the next core step should still be deterministic-model refinement and validation, not new simulation architecture
