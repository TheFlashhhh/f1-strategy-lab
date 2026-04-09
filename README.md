# F1 Strategy Lab

## 1. Project Overview
F1 Strategy Lab is a lap-level race-strategy modeling project that builds a deterministic decision engine for pit timing using real Formula 1 race data.
The project moves from exploratory analysis to a reusable simulation pipeline that outputs PIT vs STAY OUT decisions from current race state.
This mirrors how real F1 strategy teams evaluate pit decisions under time pressure, translating raw race-state data into actionable strategy decisions.
The project is developed in a research notebook and productionized into modular Python components for preprocessing, degradation modeling, and deterministic pit-window simulation.

Current scope:
- Detect pit stops from stint transitions
- Build per-compound tyre degradation models
- Estimate empirical pit-loss from observed race laps
- Optimize pit timing and return a PIT vs STAY OUT recommendation

## Live Demo

The project includes an interactive Streamlit interface for PIT vs STAY OUT recommendations.

```bash
streamlit run app/streamlit_app.py
```

![App Screenshot](assets/app_screenshot.png)

## 2. Data

**⚠️ Important:** Generated race data is **not tracked in Git** and must be built locally on first use (see "Data Setup" section in "How to Run" below). This keeps the repository lightweight and ensures data reproducibility.

### Primary Data Source: Phase 1A (Miami-focused, Parquet)

The project defaults to **Phase 1A canonical Parquet data** (built locally via `python src/data/build_phase1_dataset.py`):
- **Miami historical (2022–2025):** Combined race laps in `data/raw/miami_historical/combined.parquet`
  - ~4,311 laps from Miami GP races across 4 seasons
  - Canonical 20-field schema with metadata tags
- **2026 pre-Miami races (schedule-driven, completion-aware):** Individual Parquet files in `data/raw/season_2026_pre_miami/`
  - Races before Miami 2026 selected dynamically from FastF1 schedule
  - Only attempted if race date is on or before build date (completion-aware)
  - As of April 2026: Australia, China, Japan (3,038 laps)
  - Automatically includes new races when the schedule updates (no manual list maintenance)

### Metadata & Audit Trail
- **Ingestion manifest:** `data/raw/manifest.json`
  - Records all ingestion parameters, batch info, and data quality stats
  - Enables reproducibility and audit trail for Phase 1A builds

### Legacy CSV Data (Fallback)
- **Base dataset:** `data/raw/2020_abudhabi_race.csv` — original exploratory data
- **Alternate dataset:** `data/raw/2024_bahrain_race.csv`
- Automatically falls back to CSV if Phase 1A Parquet not available (with warning)

### Schema & Granularity
- **Canonical Phase 1A schema:** driver, lap_number, lap_time, compound, tyre_life, stint, pit_in_time, pit_out_time, is_accurate, deleted, track_status, position, team, + metadata (season, event_name, circuit_name, session_name, data_group, regulation_era, target_race_context)
- **Granularity:** Lap-level records (1 row per driver per lap)
- **Internal representation:** Converted to legacy PascalCase (Driver, LapNumber, LapTime, etc.) for backward compatibility with existing preprocessing and simulation logic

## Tech Stack
- Python
- pandas
- numpy
- matplotlib
- Streamlit
- Modular pipeline design (preprocessing, feature engineering, simulation)

## 3. Method

### Unified Phase 1 Pipeline: Integrated Evaluation

The main strategy pipeline now uses a unified degradation evaluation interface that orchestrates Phases 1A, 1B, and 1C:

```python
from src.features.evaluate_degradation import evaluate_all_degradation

# One-line integration of all three phases
result = evaluate_all_degradation(
    model_laps,
    use_fuel_correction=True,   # Phase 1B enabled
    use_piecewise=True,         # Phase 1C enabled with fallback
)

# Unified prediction API (abstracts model type)
lap_time = result.predict_lap_time(compound="MEDIUM", tyre_life=5)

# Model inspection
info = result.get_model_info("MEDIUM")
print(f"{info['model_type']} | {info['samples']} samples | cliff @{info['breakpoint_tyre_life']}")

# Backward compatibility: convert to legacy linear dict if needed
legacy_models = result.to_legacy_linear_models()
```

**Key benefit:** Strategy code doesn't need to know about model types—it just calls `predict_lap_time()` and gets the best available prediction (piecewise or linear).

### Phase 1A: Data Loading (Parquet-first)
Currently integrated via `src/data/loader.py`, using Miami historical and 2026 pre-Miami races.

### Phase 1: Step-by-step Method

1. **Pit-stop detection:**
Pit stops are identified when stint changes for a given driver between consecutive laps (with each driver's first lap explicitly not treated as a pit stop).

2. **Lap cleaning and model filtering:**
- Remove missing/non-numeric lap times and outlier laps above 150s
- Build a model-grade subset using non-pit laps, tyre life > 2, accurate laps, non-deleted laps, and green-flag track status

3. Degradation modeling:
Estimate compound-specific degradation by regressing lap time on tyre life.
$\text{LapTime} = \text{slope} \cdot \text{TyreLife} + \text{intercept}$

4. Empirical pit-loss estimation:
Use a pit-window proxy based on local before/after baselines around pit events to estimate pit penalty samples.

5. Pit-window optimization and policy:
For a given race state, exhaustively evaluate all feasible pit laps and select the strategy that minimizes total race time. A decision function returns either PIT or STAY OUT (with suggested laps to pit).
All strategy decisions are derived from full pit-window enumeration and total-race-time minimization.

### Phase 1B: Fuel Correction
**Goal:** Remove fuel-load confound from raw lap times before fitting degradation models.

**Method:** As a race progresses, fuel load decreases, making cars faster due to reduced weight. This masks the true tyre degradation effect. Phase 1B estimates and removes this effect:

1. **Estimate fuel-burn time effect:** Fit linear model on model-grade laps: $\text{LapTime} \approx \text{fuel\_effect} \times \text{race\_progress}$ per compound
2. **Apply correction:** Create `FuelCorrectedLapTime = LapTime - (fuel_effect × race_progress)`, normalizing to full-tank baseline
3. **Evaluate impact:** Compare degradation slopes before and after correction to quantify fuel confound

**Key Assumptions:**
- Fuel burn effect is approximately linear across a race
- Faster lap times late in race driven primarily by lower fuel load
- Mixing fuel and tyre effects confounds true degradation estimation

**Results on Miami 2022–2025:**
- Estimated fuel effect: MEDIUM -2.33 s/race, HARD -1.68 s/race, SOFT -0.59 s/race
  - Negative values mean lap times improve (get faster) as the race progresses due to fuel burn
- Impact on degradation: Raw slopes change 26–1450% after correction
- Example: MEDIUM raw slope (-0.046 s/lap) → corrected slope (+0.014 s/lap), indicating fuel was masking actual tyre degradation

**Run Phase 1B demo:**
```powershell
python app/demo_phase1b.py
```

Generates `data/processed/fuel_correction_summary.json` with detailed before/after comparison.

### Phase 1C: Degradation Modeling with Cliff Detection
**Goal:** Improve lap-time predictions by detecting and modeling tyre-wear cliffs—abrupt performance drop-offs that occur mid-stint.

**Problem:** Linear degradation models underestimate early wear and overestimate late-stint performance. Real tyres exhibit a cliff: manageable wear early, then sharp degradation after a threshold.

**Method:** Piecewise linear model with automatic breakpoint search:
1. Fit two line segments (pre-cliff and post-cliff) with an optimal breakpoint tyre-life value
2. Minimize total residual sum of squares (RSS) across both segments
3. Fall back to linear model if insufficient data (< 10 samples per segment)
4. Apply fuel correction before fitting (Phase 1B output)

**Results on Miami 2022–2025:**
- SOFT: 33 laps (insufficient) → linear fallback
- MEDIUM: 691 laps → cliff at tyre-life 8 (RSS improvement ~15%)
- HARD: 1,325 laps → cliff at tyre-life 10 (RSS improvement ~13%)

**Key insight:** MEDIUM and HARD compounds exhibit clear mid-stint cliffs, improving late-stint lap-time predictions and pit-window timing.

**Run Phase 1C demo:**
```powershell
python app/demo_phase1c.py
```

Generates `data/processed/degradation_model_comparison.json` with detailed breakpoint and RSS comparison.

**Full documentation:** See [docs/phase1c_degradation_modeling.md](docs/phase1c_degradation_modeling.md) for methodology, implementation, and caveats.

## 4. Key Findings (Phase 1A + Phase 1B Fuel Correction + Phase 1C Cliff Detection)
- **Dataset scale:** 7,349 total laps
  - Miami historical: 4,311 laps (2022–2025)
  - 2026 pre-Miami: 3,038 laps (Australia, China, Japan)
- **Model laps (Miami only):** 2,049 laps pass quality filters (pit-stop exclusion, tyre life > 2, accurate laps, green-flag track status)
- **Pit-loss estimation (Miami):** 1,728 pit-window samples; median empirical pit-loss ~0.30s
- **Degradation models (Miami):**
  - SOFT: -0.067 s/lap (strong degradation)
  - MEDIUM: -0.049 s/lap (moderate degradation)
  - HARD: +0.003 s/lap (minimal degradation/wear resilience)
- **Example baseline:** Driver on MEDIUM tyres with life 5 and 25 laps remaining receives **PIT** recommendation (optimal pit lap: 1)
- **Decision policy:** Deterministic, threshold-based; derived from full pit-window enumeration minimizing total race time
- **Response time:** Full optimization completes in near-instant for each scenario in local app
- **2026 data ready:** Three completed races (Australia, China, Japan) with 3,038 laps available for active-aero regulation calibration

## Results Snapshot
- **Total dataset:** 7,349 laps (4,311 Miami historical + 3,038 2026 pre-Miami)
- **Median pit-loss (Miami):** 0.30s (from 1,728 pit-window samples)
- **Model quality (Miami):** 2,049 laps pass quality filters (out of 4,311 raw)
- **Baseline optimization:** Driver on MEDIUM life-5, 25 laps remaining → recommend PIT, optimal lap 1
- **Decision time:** Sub-second response in local app
- **2026 status:** Schedule-driven ingestion successfully loaded 3 completed pre-Miami races (Australia, China, Japan)

The system processes Miami race data from 4 seasons plus 2026 pre-Miami data, evaluating ~20-25 candidate pit strategies per scenario with near-instant response time.


## 5. Project Structure
```
f1-strategy-lab/
├── app/
│   ├── demo_strategy.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── 2020_abudhabi_race.csv
│   ├── processed/
│   └── features/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data/
│   │   ├── ingest.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── simulation/
│   │   ├── simulator.py
│   │   └── strategy.py
│   ├── models/
│   ├── api/
│   └── utils/
├── requirements.txt
└── README.md
```

Clarification: strategy.py contains pit-window optimization and decision logic; simulator.py is reserved for future stochastic and multi-agent race simulation.

## 6. How to Run
### Install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Data Setup: Build Local Dataset

**Important:** Generated race data is **not stored in Git** (see `.gitignore`). Each user must build their own local dataset using the Phase 1A builder script:

```powershell
python src/data/build_phase1_dataset.py
```

**What this does:**
1. Downloads Miami GP races (2022–2025) from FastF1
2. Optionally downloads 2026 pre-Miami races (if available/completed)
3. Saves to Parquet format in `data/raw/miami_historical/`
4. Generates `data/raw/manifest.json` for reproducibility and audit trail
5. Creates processed artifacts in `data/processed/` as needed

**Output locations:**
- `data/raw/miami_historical/combined.parquet` — Combined Miami races (~4,311 laps)
- `data/raw/miami_historical/{year}_miami_grand_prix_race.parquet` — Individual years
- `data/raw/season_2026_pre_miami/*.parquet` — 2026 pre-Miami races (if available)
- `data/raw/manifest.json` — Ingestion audit trail
- `data/processed/*.json` — Generated model artifacts (fuel correction, degradation models)

**Subsequent runs:** Skip this step if Parquet files already exist in your local `data/raw/` directory.

**Why local-only?** 
- Generated data is large (~100+ MB with FastF1 cache) and circuit-specific
- Rebuilding locally ensures reproducibility and data freshness
- Users can optionally add their own race data or customize sources
- No dependency on shared data storage or version control of binary files

### Run the notebook
```powershell
jupyter notebook notebooks/eda.ipynb
```

Optional full execution check:
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

### Run the integrated Phase 1 demo
```powershell
python app/demo_strategy.py
```

**What it does (Integrated Phase 1 stack):**
1. **Phase 1A**: Loads Miami Parquet data (via `src/data/loader.py`)
2. **Phase 1B**: Applies fuel correction (automatic) to normalize fuel-load confound
3. **Phase 1C**: Fits piecewise degradation models with cliff detection (with linear fallback)
4. Estimates empirical pit-loss window
5. Optimizes pit strategy using improved degradation models
6. Reports which model type is active for each compound

**Key improvement:** Default now uses fuel-corrected + piecewise models, not just linear baselines.

**Output:** Example
```
================================================================================
F1 STRATEGY LAB - INTEGRATED PHASE 1 PIPELINE
================================================================================

Phase 1A: Loading data...
  Loaded 4311 raw laps
  Filtered to 2049 model-grade laps

Phase 1B + 1C: Evaluating degradation (fuel correction + piecewise)...

Model Status (by compound):
  SOFT       (  33 samples): LINEAR (piecewise fallback)
  MEDIUM     ( 691 samples): PIECEWISE                 [cliff at tyre-life 14]
  HARD       (1325 samples): PIECEWISE                 [cliff at tyre-life 26]

Estimating pit loss...
  Pit-loss samples: 1728
  Median pit-loss: 0.30 s

Optimizing pit strategy...
  Example scenario: MEDIUM compound, tyre-life 5, 25 laps remaining
    Optimal pit lap: 1
    Minimum total time: 2321.97 s
    Decision: PIT

Example lap-time predictions (MEDIUM compound, using active models):
  Tyre-life  1: 94.24 s
  Tyre-life  5: 94.07 s
  Tyre-life 10: 94.38 s
  Tyre-life 15: 94.79 s
  Tyre-life 20: 95.40 s
```

### Run Phase 1B fuel correction demo
```powershell
python app/demo_phase1b.py
```

**What it does:**
1. Loads Phase 1A Miami data
2. Estimates fuel-burn time effects per compound using race progress
3. Applies fuel correction to lap times
4. Compares degradation models before and after fuel correction
5. Saves detailed evaluation to `data/processed/fuel_correction_summary.json`

**Key output:**
- Estimated fuel effects (seconds per race progression per compound)
- Before/after degradation slopes showing impact of fuel confound removal
- Sample counts and stability assessment per compound

### Run Phase 1C cliff-detection demo
```powershell
python app/demo_phase1c.py
```

**What it does:**
1. Loads Phase 1A Miami data
2. Applies Phase 1B fuel correction to lap times
3. Fits piecewise-linear degradation models with automatic cliff-breakpoint search
4. Compares piecewise vs linear models (RSS improvement)
5. Saves detailed comparison to `data/processed/degradation_model_comparison.json`

**Key output:**
- Cliff breakpoint tyre-life for each compound
- Pre-cliff vs post-cliff degradation slopes
- RSS improvement (%) showing model accuracy gain
- Honest fallback reporting for compounds with insufficient data

### Run the interactive web app (with integrated Phase 1)
```powershell
streamlit run app/streamlit_app.py
```

**What it provides:**
- Real-time pit strategy calculator using integrated Phase 1 pipeline
- Adjustable race state sliders: current compound, tyre life, laps remaining, target compound
- Live optimization results with pit-lap candidate curve
- **Model status display**: Shows which model type (piecewise vs linear) is active for each compound
- Transparent fuel correction and degradation mode reporting

### Baseline comparison (Phase 1C isolated demo)
To see Phase 1C models in isolation (without active use in strategy):
```powershell
python app/demo_phase1c.py
```

**What it shows:**
- Linear baseline vs piecewise comparison
- Cliff breakpoints for each compound
- RSS improvement metrics
- Degradation model artifact (JSON)
- For research/validation purposes; main app uses piecewise integrated

## 7. Comparison: Pipeline Modes

## 7. Comparison: Pipeline Modes

| Mode | Command | Data | Fuel Correction | Degradation Model | Use Case |
|------|---------|------|---|---|---|
| **Integrated Phase 1 (Default)** | `python app/demo_strategy.py` | Phase 1A Miami | ✅ Automatic (1B) | Piecewise with fallback (1C) | Main app; production strategy |
| **Interactive Streamlit** | `streamlit run app/streamlit_app.py` | Phase 1A Miami | ✅ Automatic (1B) | Piecewise with fallback (1C) | Real-time UI; pit decisions |
| **Phase 1C Isolated** | `python app/demo_phase1c.py` | Phase 1A Miami | ✅ Automatic (1B) | Piecewise (full detail) | Research; model validation |
| **Phase 1B Fuel Correction** | `python app/demo_phase1b.py` | Phase 1A Miami | ✅ Detailed | Linear (for comparison) | Fuel effect analysis |

**Key insight:** The main pipeline (`demo_strategy.py` and `streamlit_app.py`) now use the **complete Phase 1 stack by default**. Baseline linear models are still available internally as fallback when piecewise data is insufficient.
- **Historical scope:** Miami GP only (2022–2025). Findings primarily validated on one circuit.
- **2026 pre-Miami data:** Currently includes only races completed before Miami 2026 (Australia, China, Japan). Schedule-driven selection will automatically include later races as they complete.
- **Deterministic modeling:** Degradation and strategy layers are deterministic and linear; uncertainty and traffic effects are not modeled.
- **Pit-loss estimation:** Proxy-based on lap windows, not full causal decomposition of in-lap, stationary, and out-lap components.
- **Deployment:** Local interactive Streamlit interface included; no production deployment or real-time serving layer currently implemented.
- **Baseline:** Current implementation serves as deterministic baseline for future stochastic and opponent-aware strategy modeling.

## 8. Current Development Focus

**Phase 1 - Nearly Complete:**
- ✅ Phase 1A: Data ingestion (Miami historical + 2026 pre-Miami races, schedule-driven)
- ✅ Phase 1B: Fuel correction (removes fuel-load confound from lap times)
- ✅ Phase 1C: Degradation modeling with cliff detection (piecewise-linear with automatic breakpoint search)

See [docs/phase1_spec.md](docs/phase1_spec.md) for Phase 1 specification and [docs/phase1c_degradation_modeling.md](docs/phase1c_degradation_modeling.md) for Phase 1C technical details.

**Phase 2 (Planned):**
- Uncertainty-aware simulation (pace variability, confidence intervals on cliff breakpoints)
- Circuit generalization (validate Phase 1C models on Bahrain, Singapore, Monaco, etc.)
- Multi-circuit regulation-era models (account for regulation changes 2022 vs 2024 vs 2026)

## 9. Next Steps
- Evaluate the same pipeline across multiple races and circuits.
- Add uncertainty-aware simulation (pace variability, safety-car states, and traffic interaction).
- Compare linear degradation against piecewise/nonlinear alternatives.
- Add unit tests for preprocessing, pit-loss estimation, and optimization components.
