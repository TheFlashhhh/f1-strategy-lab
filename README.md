# F1 Strategy Lab

Real-time pit strategy recommendations built on Formula 1 race data. The current system combines:

- Phase 1 empirical modeling
- Phase 2 automatic strategy optimization
- Phase 2E calibration
- Pre-3 defensibility and held-out backtest diagnostics

The product vision is bigger than the currently implemented system. Today the engine is still a deterministic, single-car recommendation stack, but the repository now also includes Phase 3A race-state groundwork and a Phase 3B replay-first dashboard shell built on top of it.

<div align="center">
  <pre>
  <b>Current State</b>: Driver on MEDIUM, tyre-life 5, 25 laps remaining
                 ->
         [Degradation Models] <- Phase 1C
         [Fuel Correction]    <- Phase 1B
         [Data Loading]       <- Phase 1A
                 ->
  <b>Output</b>: PIT in lap 1 -> Switch to SOFT
         Est. time: 2321.97 s | Feasible: yes
  </pre>
</div>

**Quick Start**

```bash
pip install -r requirements.txt
python src/data/build_phase1_dataset.py
streamlit run app/streamlit_app.py
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Dataset | 7,349 laps |
| Miami Historical | 4,311 laps (2022-2025) |
| 2026 Pre-Miami | 3,038 laps (Australia, China, Japan) |
| Model-Grade Laps | 2,049 (Miami, after filtering) |
| Median Pit-Loss | 14.34 s (Miami baseline, Phase 2E calibrated) |
| MEDIUM Degradation | -0.049 s/lap (corrected) |
| HARD Degradation | +0.003 s/lap (corrected) |
| Response Time | Sub-second |

---

## Features

### Completed phases

- Phase 1A: Data loading (Miami historical + 2026 pre-Miami races, schedule-driven)
- Phase 1B: Fuel correction (removes fuel-load confound from lap times)
- Phase 1C: Degradation modeling with cliff detection
- Phase 2A: Automatic strategy search and recommendation
- Phase 2B: Hybrid data context (Miami anchor + current-season recency support)
- Phase 2C: Strategy sensitivity analysis
- Phase 2D: Broader validation / robustness evaluation
- Phase 2E: Strategy search refinement / calibration
- Pre-3: Defensibility upgrade (support tiers, role-based hybrid predictions, held-out Miami backtest, stop-timing and pace-shape diagnostics)

### Product-shell phases

- Phase 3A: Race-state and dashboard groundwork
- Phase 3B: Replay-first dashboard shell and driver detail experience
- Planned next: Phase 3C competitor-gap / undercut-overcut context
- Planned next: Phase 3D richer race realism (SC/VSC, traffic, weather, stochastic branching)

Phase 3A and 3B are still intentionally limited. They add the canonical race-state model and a replay-first UI shell, and the latest Phase 3B.5 pass moves the primary race-control surface into a custom Streamlit component. They do not add competitor-aware recommendation logic yet and they do not redesign the underlying strategy engine.

### Unified pipeline

```python
from src.features.evaluate_degradation import evaluate_all_degradation

result = evaluate_all_degradation(
    model_laps,
    use_fuel_correction=True,
    use_piecewise=True,
)

lap_time = result.predict_lap_time(compound="MEDIUM", tyre_life=5)
```

---

## Project Structure

```text
f1-strategy-lab/
|-- app/
|   |-- streamlit_app.py
|   |-- demo_strategy.py
|   |-- demo_phase1b.py
|   `-- demo_phase1c.py
|-- assets/
|   |-- asset_manifest.json
|   |-- team_logos/
|   `-- driver_photos/
|-- components/
|   `-- race_control_board/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- notebooks/
|-- scripts/
|-- src/
|   |-- data/
|   |-- features/
|   |-- simulation/
|   `-- utils/
|-- tests/
|-- CONTRIBUTING.md
|-- requirements.txt
`-- README.md
```

Main UI: `app/streamlit_app.py`  
Main walkthrough: `app/demo_strategy.py`

---

## How To Run

### 1. Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/data/build_phase1_dataset.py
```

### 2. Interactive app

```bash
streamlit run app/streamlit_app.py
```

The canonical app now opens in a replay-first dashboard shell:

- a custom race-control board surface for circuit, leaderboard, and selected-driver interaction
- darker, tighter schematic circuit-view placeholder
- dense timing / order panel with row-style selection instead of repeated generic controls
- compact selected-driver tactical drawer integrated into the board
- optional local-first team logos and driver photos when supplied through `assets/asset_manifest.json`
- team-color-aware selected-state accents for the board, pit-call card, and analysis drawer
- strategy-strip stint history and race-call recommendation card
- brighter race-control header, lighter source sidebar, and a Race Control / Analysis split so deep analysis stays out of the primary tactical screen
- a native Analysis selected-car summary path that renders from structured Python data instead of component HTML fragments
- a final small color-separation polish where circuit markers use team color while timing-table tyre pills stay compound-colored

### 3. Core demos and validation

```bash
python app/demo_strategy.py
python scripts/build_race_pack.py --race canada_2026
python scripts/run_phase2d_validation.py
python scripts/run_pre3_backtest.py
python scripts/run_stop_timing_audit.py
python scripts/run_pace_shape_audit.py
python scripts/run_phase3a_readiness.py
python app/demo_phase1b.py
python app/demo_phase1c.py
```

### 4. Tests

```bash
pytest tests/
```

### 5. Utility scripts

```bash
python scripts/validate_fuel_correction.py
python scripts/test_phase1_integration.py
python scripts/run_phase2d_validation.py
python scripts/build_race_pack.py --race canada_2026
python scripts/run_phase3a_readiness.py
python scripts/verify_data_manifest.py
```

See [scripts/README.md](scripts/README.md) for more detail.

---

## Data

Generated data is not tracked in Git.

- Miami historical (2022-2025): about 4,311 laps
- 2026 pre-Miami races: about 3,038 laps
- Race activation packs, including Canada 2026 readiness artifacts, are generated under `data/processed/race_packs/`
- Ingestion manifest: `data/raw/manifest.json`

Build local data with:

```bash
python src/data/build_phase1_dataset.py
```

---

## Methodology

### Phase 1A: Data loading

Load and standardize race data from multiple F1 races. See [docs/phase1a_summary.md](docs/phase1a_summary.md).

### Phase 1B: Fuel correction

Remove fuel-load confound from lap times before degradation modeling.

See [docs/phase1b_fuel_correction.md](docs/phase1b_fuel_correction.md).

### Phase 1C: Degradation modeling with cliff detection

Detect and model mid-stint tyre-wear cliffs using piecewise regression.

See [docs/phase1c_degradation_modeling.md](docs/phase1c_degradation_modeling.md).

### Phase 2A: Automatic strategy recommendation

Search pit-window space and recommend the strategy minimizing total race time.

Key files: `src/simulation/strategy.py`, `src/simulation/strategy_engine.py`

### Phase 2B / Pre-3: Role-based hybrid modeling

Use Miami historical data as the circuit anchor and treat current-season 2026 races as bounded recency support rather than direct Miami truth.

### Phase 2C: Sensitivity analysis

Stress-test the baseline recommendation under pit-loss and degradation variations and label it as Stable, Moderately Sensitive, or Fragile.

### Phase 2D: Broader validation / robustness evaluation

Run a representative scenario suite across compounds, tyre ages, and remaining-race lengths to understand where the strategy system is robust versus brittle.

Artifacts:

- `data/processed/phase2d_validation_summary.json`
- `data/processed/phase2d_validation_summary.csv`

### Phase 2E: Strategy search refinement / calibration

Calibrate the strategy stack after broader validation by:

- fixing race-context leakage in pit-stop and fuel-progress grouping
- restoring a non-degenerate Miami pit-loss baseline
- cleaning up SOFT model-health behavior
- replacing the rough two-stop heuristic with a bounded search over valid pit-lap pairs

### Pre-3: Defensibility upgrade

Improve how the project presents and audits model trustworthiness by:

- adding explicit compound support tiers
- using a role-based Miami-anchor design
- expanding the honest held-out Miami decision-support backtest
- adding stop-timing diagnostics
- adding pace-shape diagnostics

### Phase 3A: Race-state and dashboard groundwork

Phase 3A adds the canonical race-state object model and extraction path needed for future replay-first dashboard work.

It supports:

- race/session identity
- driver/team identity when available
- current compound and tyre age
- stint-history reconstruction
- recommendation payload fields
- support/confidence/risk metadata
- placeholders for nearby competitors and future event status

It does not yet add:

- competitor-aware recommendations
- traffic logic
- SC/VSC strategy behavior
- weather logic
- Monte Carlo branching
- opponent modeling
- a full dashboard UI

See [docs/phase3a_race_state_groundwork.md](docs/phase3a_race_state_groundwork.md).

### Phase 3B: Replay-first dashboard shell

Phase 3B uses the Phase 3A checkpoint model inside the canonical Streamlit app.

It adds:

- a replay-first race-control layout in `app/streamlit_app.py`
- a custom Streamlit Components v2 race-control surface for the circuit, timing leaderboard, and selected-driver drawer
- a dark compact schematic circuit-view placeholder with tighter map occupancy for stable ordered marker placement
- a dense timing/order panel built from canonical race-state objects
- a compact selected-driver tactical panel
- a local-first asset manifest for optional team logos and driver photos with graceful fallback when assets are missing
- team-color accents on selected timing rows, track halos, and current-call surfaces
- a visual stint strip and cleaner race-call recommendation surface
- a Race Control / Analysis view switch that keeps alternatives, sensitivity, model context, and timing curves secondary and more compact

It still does not add:

- live timing ingestion
- exact interval-gap timing
- real telemetry coordinates
- competitor-aware strategy logic
- SC/VSC, weather, traffic, or Monte Carlo behavior
- exact live race-control behavior

See [docs/phase3b_dashboard_shell.md](docs/phase3b_dashboard_shell.md).
See [docs/phase3b_custom_component.md](docs/phase3b_custom_component.md).

---

## Pipeline Modes

| Mode | Command | Fuel Correction | Degradation | Use Case |
|------|---------|-----------------|-------------|----------|
| Integrated (Main) | `streamlit run app/streamlit_app.py` | yes | Piecewise | Production pit decisions |
| Strategy Demo | `python app/demo_strategy.py` | yes | Piecewise | Debug strategy logic |
| Phase 1C | `python app/demo_phase1c.py` | yes | Piecewise | Validate models |
| Phase 1B | `python app/demo_phase1b.py` | yes | Linear | Analyze fuel effects |

---

## Limitations

- Single circuit: the core recommendation model is still validated on Miami only.
- Canada 2026 activation: available as a race-weekend manual snapshot slice, but it is cautious unless local `canada_historical` FastF1 data has been generated.
- Deterministic: there is no uncertainty quantification or stochastic race branching yet.
- Current engine scope: recommendation logic is still single-car and not competitor-aware yet.
- Phase 3A/3B scope: the race-state model and dashboard shell are product-structure work, not new strategy behavior.
- No traffic model: the engine does not account for overtaking or position effects.
- No safety-car strategy: the engine does not respond to VSC or full Safety Car conditions.
- Hybrid modeling: 2026 non-Miami data is bounded recency support, not direct Miami degradation truth.
- Backtesting scope: the Pre-3 backtest is honest decision-support auditing, not a historical performance guarantee.
- Pace-shape confidence: repeated long-stint timing misses still mean the deterministic core needs more credibility work before richer Phase 3 logic.
- Validation scope: Phase 2D is representative scenario validation, not Monte Carlo race simulation.
- Dashboard future work: interval gaps, live track-map coordinates, tyre inventory, weather, and true live race control remain future phases. Logos and driver photos are now optional local assets only; nothing is auto-downloaded.

---

## Roadmap

### Completed

- Phase 1A
- Phase 1B
- Phase 1C
- Phase 2A
- Phase 2B
- Phase 2C
- Phase 2D
- Phase 2E
- Pre-3 diagnostics / defensibility work
- Phase 3A race-state and dashboard groundwork
- Phase 3B replay-first dashboard shell

### Planned next phases

- Phase 3C - Competitor-gap / undercut-overcut context
- Phase 3D - Richer race realism (SC/VSC, traffic, weather, stochastic branching)

The final product vision is a replay-first race-control dashboard that can later grow into a live product. The implemented system today now includes the Phase 3A checkpoint layer and the Phase 3B replay shell, but the strategy core is still deterministic and not yet competitor-aware.

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for code standards, workflow, testing guidance, and architecture notes.

---

## Documentation

- [Phase 1 Specification](docs/phase1_spec.md)
- [Phase 1A: Data Loading](docs/phase1a_summary.md)
- [Phase 1B: Fuel Correction](docs/phase1b_fuel_correction.md)
- [Phase 1C: Degradation Modeling](docs/phase1c_degradation_modeling.md)
- [Phase 2A: Strategy Engine](docs/phase2a_strategy_engine.md)
- [Phase 2B: Hybrid Modeling](docs/phase2b_hybrid_modeling.md)
- [Phase 2C: Sensitivity Analysis](docs/phase2c_sensitivity_analysis.md)
- [Phase 2D: Broader Validation](docs/phase2d_validation.md)
- [Phase 2E: Strategy Refinement](docs/phase2e_strategy_refinement.md)
- [Pre-3: Defensibility Upgrade](docs/pre3_defensibility_upgrade.md)
- [Pre-3: Stop-Timing Audit](docs/pre3_stop_timing_audit.md)
- [Pre-3: Pace-Shape Audit](docs/pre3_pace_shape_audit.md)
- [Phase 3A: Race-State Groundwork](docs/phase3a_race_state_groundwork.md)
- [Phase 3B: Dashboard Shell](docs/phase3b_dashboard_shell.md)
- [Phase 3B.5: Custom Race-Control Component](docs/phase3b_custom_component.md)
- [Assets Pipeline](assets/README.md)

---

## License

[See LICENSE file](LICENSE)
