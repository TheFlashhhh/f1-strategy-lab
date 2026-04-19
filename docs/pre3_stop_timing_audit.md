# Pre-3: Stop-Timing Audit

## Purpose

This audit was added after the SOFT credibility pass showed that exact backtest alignment was still weak even when future-compound choice was often directionally correct.

The goal is to answer a narrower question before any Phase 3 work:

- are the remaining misses mainly true timing mistakes
- or are they cases where multiple nearby pit laps are effectively equivalent

## Canonical scripts

- `python scripts/run_pre3_backtest.py`
- `python scripts/run_stop_timing_audit.py`

## Artifacts

- `data/processed/pre3_backtest_diagnostics.json`
- `data/processed/pre3_stop_timing_audit.json`

## What the timing audit traces

For a fixed strategy sequence, the audit traces total estimated race time across the feasible first-stop window.

It reports:

- best first stop
- near-optimal first-stop band
- whether the optimum is `sharp`, `moderate`, or `flat`
- whether the best lap sits on a feasible-window edge
- actual stop timing when a held-out backtest checkpoint is available

For two-stop sequences, the trace varies the first stop while optimizing the second stop for that first-stop choice.

## Current practical findings

- the expanded Miami 2024 backtest now has 10 checkpoints instead of 4
- backtest misses are still dominated by stop-timing rather than pure next-compound choice
- some one-stop timing curves are flat enough that exact lap rank overstates precision
- several early-pit one-stop recommendations were coming from a flat band where later laps stayed within about a second of the minimum
- the engine now prefers the latest one-stop lap inside a 1.0s near-optimal band instead of always taking the earliest minimum-time lap

This improves honesty and timing calibration, but it does not solve every miss.

## What still looks real, not cosmetic

The repeated long HARD-stint cases in the held-out backtest still prefer staying out far longer than the actual race. After the low-margin SOFT tie-break, these now look more like timing-calibration problems on a flat HARD continuation than pure future-compound mistakes.

## Relationship to Phase 2D

The timing audit and the updated Phase 2D suite together suggest:

- ordinary representative states remain one-stop dominant under the current pit-loss and degradation assumptions
- the previous Phase 2D suite was also too normal-state heavy to exercise the two-stop path
- adding a small number of extreme long-distance stress states is enough to confirm that the two-stop path still works, without claiming that two-stops are common

## Readiness implication

This pass strengthens interpretation and timing calibration, but it is still a Pre-Phase-3 hardening step.

Phase 3 should wait until:

- timing-heavy backtest misses are better bounded
- long-stint pace-shape behavior is more convincing
- representative validation is broader than a one-stop-heavy ordinary-state slice
