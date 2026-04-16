# Phase 2D: Broader Validation / Robustness Evaluation

## Status: Complete

Phase 2D adds a canonical validation harness for the strategy system. Instead of relying on a single demo scenario, it runs the existing Phase 2A strategy engine and Phase 2C stability analysis across a compact representative scenario suite and saves inspectable artifacts.

Phase 2E keeps this same harness as the canonical comparison point. The latest artifact therefore reflects the post-calibration pipeline, not the earlier pre-fix run that still had pit-loss collapse and invalid SOFT predictions.

---

## What Phase 2D Validates

Phase 2D asks a narrower but important question:

- Does the strategy system behave sensibly across multiple representative race states?
- When does it prefer one-stop versus two-stop plans?
- Which recommendations stay stable under Phase 2C assumption shifts?
- Where do warnings, fragility, or weak-data signals cluster?

It does not add new race simulation features. It evaluates the current deterministic strategy stack as it exists today.

---

## Validation Gap It Closes

Before Phase 2D, the repository had:

- A single main walkthrough in `app/demo_strategy.py`
- Strategy search logic in `src/simulation/strategy_engine.py`
- Per-scenario sensitivity analysis in `src/simulation/strategy_sensitivity.py`
- General pipeline tests and Phase 1 validation scripts

What was missing was a broader validation harness that:

- Exercised multiple current compounds
- Covered low, medium, and high tyre age states
- Covered short, medium, and long laps-remaining states
- Summarized robustness patterns across scenarios instead of only one scenario at a time
- Produced a durable, inspectable artifact

Phase 2D fills that gap.

---

## Scenario Suite

The representative suite is defined in `src/simulation/strategy_validation.py` and currently contains 12 scenarios.

### Coverage Design

- Current compound: `SOFT`, `MEDIUM`, `HARD`
- Tyre age buckets: `low`, `medium`, `high`
- Laps remaining buckets: `short`, `medium`, `long`

The suite is intentionally compact rather than combinatorial. It mixes:

- baseline states
- realistic trade-off zones
- stress cases such as aged SOFT or heavily used HARD with long distance remaining

### Current Scenarios

| Scenario ID | Compound | Tyre Age | Laps Remaining | Purpose |
|-------------|----------|----------|----------------|---------|
| `soft_low_short` | SOFT | low | short | Fresh soft near race end |
| `soft_medium_medium` | SOFT | medium | medium | Standard soft mid-stint |
| `soft_high_long` | SOFT | high | long | Stress case for aged soft |
| `soft_high_short` | SOFT | high | short | Immediate pit-pressure soft case |
| `medium_low_medium` | MEDIUM | low | medium | Fresh-medium baseline |
| `medium_medium_long` | MEDIUM | medium | long | One-stop vs two-stop trade-off zone |
| `medium_high_short` | MEDIUM | high | short | Worn medium late in race |
| `medium_high_medium` | MEDIUM | high | medium | Aged medium fragility check |
| `hard_low_long` | HARD | low | long | Fresh hard long finish |
| `hard_medium_short` | HARD | medium | short | Hard tyre short sprint |
| `hard_high_medium` | HARD | high | medium | Used hard medium finish |
| `hard_high_long` | HARD | high | long | Stress case for aged hard |

---

## What Gets Recorded Per Scenario

For each representative scenario, Phase 2D records:

- best strategy type
- next tyre
- final tyre when applicable
- pit lap(s)
- estimated total time
- feasibility and feasibility reason
- runner-up time gap
- Phase 2C stability label
- pit-loss sensitivity flag
- degradation sensitivity flag
- flip conditions
- warning labels for pathological or suspicious cases

---

## Robustness Summary

The aggregate summary includes:

- one-stop versus two-stop recommendation counts
- stability label counts
- fragile states grouped by current compound
- fragile states grouped by tyre-age bucket
- fragile states grouped by laps-remaining bucket
- unstable scenario IDs by current compound
- warning counts
- pathological cases such as infeasible best plans or tiny runner-up margins
- SOFT-specific weak-data assessment based on active model metadata and scenario warnings

This gives a quick answer to:

- where the engine looks robust
- where recommendations become fragile
- whether SOFT still appears weak or under-supported

---

## Artifacts

Phase 2D writes:

- `data/processed/phase2d_validation_summary.json`
- `data/processed/phase2d_validation_summary.csv`

The JSON artifact contains:

- metadata
- all scenario outputs
- aggregate robustness summary

The CSV artifact flattens the scenario-level outputs for quick inspection or spreadsheet use.

### Latest Post-Phase-2E Snapshot

Compared with the pre-Phase-2E validation run:

- pit-loss baseline moved from `0.00s` to `14.34s` using Miami-only, race-grouped pit-loss calibration
- one-stop vs two-stop changed from `12 / 0` to `10 / 2`
- stability changed from `8 Stable / 4 Moderately Sensitive / 0 Fragile` to `9 / 3 / 0`
- SOFT prediction health moved from invalid (`null` probe predictions) to valid scenario usage

This keeps Phase 2D useful as an evaluation harness while showing exactly how Phase 2E changed the recommendation mix.

---

## Canonical Script

Run:

```bash
python scripts/run_phase2d_validation.py
```

The script:

- loads the same hybrid pipeline used by the app/demo
- fits degradation models through the existing Phase 1 stack
- estimates pit-loss from the current data
- runs the representative Phase 2D suite
- prints a concise console summary
- saves JSON and CSV artifacts

---

## Relationship to Other Phases

- Phase 2A chooses the best strategy for a given race state.
- Phase 2C evaluates sensitivity for one chosen race state.
- Phase 2D evaluates many representative race states and summarizes the robustness pattern across them.

Phase 2D is therefore an evaluation layer, not a new optimizer.

---

## What Remains Unvalidated

Phase 2D still does not validate:

- historical backtesting against actual race outcomes
- traffic-aware recommendations
- safety-car or VSC response
- weather effects
- opponent-dependent strategy interactions
- Monte Carlo uncertainty or probability distributions
- multi-circuit generalization beyond the current Miami-plus-recency modeling design

So the output should be read as:

- broader representative validation of the current system

not as:

- full real-world race strategy certification

---

## Key Files

- `src/simulation/strategy_validation.py`
- `scripts/run_phase2d_validation.py`
- `app/demo_strategy.py`
- `app/streamlit_app.py`
