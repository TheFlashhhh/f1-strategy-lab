# F1 Strategy Lab

Real-time pit strategy recommendations using Phase 1 empirical modeling and Phase 2 automatic strategy optimization. Built on real Formula 1 race data with hybrid degradation models blending historical and current-season information.

<div align="center">
  <pre>
  <b>Current State</b>: Driver on MEDIUM, tyre-life 5, 25 laps remaining
                 ↓
         [Degradation Models] ← Phase 1C (piecewise with cliff detection)
         [Fuel Correction]    ← Phase 1B (removes fuel-load confound)
         [Data Loading]       ← Phase 1A (Miami + 2026 pre-Miami races)
                 ↓
  <b>Output</b>: PIT in lap 1 → Switch to SOFT
         Est. time: 2321.97 s | Feasibility: ✓
  </pre>
</div>

**Quick Start:**
```bash
# Install and build data
pip install -r requirements.txt
python src/data/build_phase1_dataset.py

# Try the interactive app
streamlit run app/streamlit_app.py
```

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Total Dataset** | 7,349 laps |
| **Miami Historical** | 4,311 laps (2022–2025) |
| **2026 Pre-Miami** | 3,038 laps (Australia, China, Japan) |
| **Model-Grade Laps** | 2,049 (Miami, after filtering) |
| **Median Pit-Loss** | 14.34 s (Miami baseline, Phase 2E calibrated) |
| **MEDIUM Degradation** | -0.049 s/lap (corrected) |
| **HARD Degradation** | +0.003 s/lap (corrected) |
| **Response Time** |Sub-second |

---

## 🚀 Features

### Phase 1: Empirical Modeling (Complete)
- ✅ **Phase 1A:** Data loading (Miami historical + 2026 pre-Miami races, schedule-driven)
- ✅ **Phase 1B:** Fuel correction (removes fuel-load confound from lap times)
- ✅ **Phase 1C:** Degradation modeling with cliff detection (piecewise-linear with automatic breakpoint search)

### Phase 2: Strategy Optimization
- ✅ **Phase 2A:** Automatic strategy search and recommendation (one-stop vs two-stop, ranked by time)
- ✅ **Phase 2B:** Hybrid data blending (Miami-specific + current-season recency, 40%-60% weighting)
- ✅ **Phase 2C:** Strategy sensitivity analysis (pit-loss & degradation scenario testing, stability classification)
- ✅ **Phase 2D:** Broader validation / robustness evaluation across a representative scenario suite
- ✅ **Phase 2E:** Strategy search refinement / calibration (pit-loss fix, SOFT cleanup, bounded two-stop search)

### Unified Pipeline
```python
# One-line Phase 1 integration
from src.features.evaluate_degradation import evaluate_all_degradation

result = evaluate_all_degradation(
    model_laps,
    use_fuel_correction=True,   # Phase 1B enabled
    use_piecewise=True,         # Phase 1C enabled with fallback
)

# Consistent API (works with any model type)
lap_time = result.predict_lap_time(compound="MEDIUM", tyre_life=5)
```

---

## 📁 Project Structure

**High-level overview:**

```
f1-strategy-lab/
├── 📱 app/
│   ├── streamlit_app.py              ← Interactive pit app (main entry)
│   ├── demo_strategy.py              ← Phase 2A strategy demo
│   ├── demo_phase1b.py               ← Phase 1B fuel correction demo
│   └── demo_phase1c.py               ← Phase 1C degradation demo
├── 📚 src/
│   ├── data/           Data loading & preprocessing
│   ├── features/       Degradation & fuel correction models
│   ├── simulation/     Pit-window optimization & strategy
│   └── utils/          Shared utilities
├── 🧪 tests/
│   └── test_pipeline.py              ← Main test suite
├── 📓 notebooks/
│   └── eda.ipynb                     ← Exploratory analysis
├── 📖 docs/
│   ├── phase1_spec.md, phase1[abc]_*.md
│   └── PHASE_1_INTEGRATION.md
├── 🛠️ scripts/
│   ├── validate_fuel_correction.py
│   ├── test_phase1_integration.py
│   └── ...
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full structure and development guidelines.

---

## 🎯 How to Run

### 1. Setup (One-time)

```bash
# Create environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Build local data (REQUIRED on first run)
python src/data/build_phase1_dataset.py
```

**Why local data?** Generated files (~100+ MB) are not tracked in Git. This ensures reproducibility and lets users customize data sources.

### 2. Run Interactive App

```bash
streamlit run app/streamlit_app.py
```

**What you get:**
- 🎯 Automatic pit strategy recommendation
- 📊 Top 5 alternative strategies ranked by time
- 📈 Real-time model status (model type, sample count per compound)
- ✅ Optional visibility into the latest Phase 2D representative validation summary
- 🔧 Advanced mode: Manual pit-lap curves and deep-dive analysis
- ⚠️ Feasibility badges and limitations

### 3. Run Demo Scripts

```bash
# Phase 2A: Strategy recommendation
python app/demo_strategy.py

# Phase 2D: Broader validation / robustness evaluation
python scripts/run_phase2d_validation.py

# Phase 2E: Current calibrated strategy pipeline demo
python app/demo_strategy.py

# Phase 1B: Fuel correction impact
python app/demo_phase1b.py

# Phase 1C: Degradation modeling with cliff detection
python app/demo_phase1c.py
```

### 4. Run Tests

```bash
pytest tests/
```

### 5. Run Utility Scripts

```bash
python scripts/validate_fuel_correction.py
python scripts/test_phase1_integration.py
python scripts/run_phase2d_validation.py
python scripts/verify_data_manifest.py
```

See [scripts/README.md](scripts/README.md) for details.

---

## 📊 Data

### Primary Source: Phase 1A (Parquet, Local-Build)

Generated data is **not tracked in Git**:

1. **Miami Historical (2022–2025):** ~4,311 laps
2. **2026 Pre-Miami:** ~3,038 laps (Australia, China, Japan)
3. **Ingestion Manifest:** `data/raw/manifest.json`

### Building Data

```bash
python src/data/build_phase1_dataset.py
```

---

## 🔬 Methodology

### Phase 1A: Data Loading
Load and standardize race data from multiple F1 races. See [docs/phase1a_summary.md](docs/phase1a_summary.md).

### Phase 1B: Fuel Correction
Remove fuel-load confound from lap times before degradation modeling.

**Results on Miami 2022–2025:**
- MEDIUM: -2.33 s/race fuel effect
- HARD: -1.68 s/race
- SOFT: -0.59 s/race

**Full doc:** [docs/phase1b_fuel_correction.md](docs/phase1b_fuel_correction.md)

### Phase 1C: Degradation Modeling with Cliff Detection
Detect and model mid-stint tyre-wear cliffs using piecewise regression.

**Results on Miami 2022–2025:**
- SOFT: 33 laps → Linear fallback
- MEDIUM: 691 laps → Piecewise (cliff at tyre-life 8, RSS improvement ~15%)
- HARD: 1,325 laps → Piecewise (cliff at tyre-life 10, RSS improvement ~13%)

**Full doc:** [docs/phase1c_degradation_modeling.md](docs/phase1c_degradation_modeling.md)

### Phase 2A: Automatic Strategy Recommendation
Search pit-window space and recommend the strategy minimizing total race time.

**Key files:** `src/simulation/strategy.py`, `src/simulation/strategy_engine.py`

### Phase 2B: Hybrid Modeling
Blend Miami-specific historical data (40%) with current-season 2026 races (60%).

### Phase 2C: Sensitivity Analysis
Stress-test the baseline recommendation under pit-loss and degradation variations and label it as Stable, Moderately Sensitive, or Fragile.

### Phase 2D: Broader Validation / Robustness Evaluation
Run a compact representative scenario suite across compounds, tyre ages, and remaining-race lengths to understand where the strategy system is robust versus brittle.

**Artifacts:**
- `data/processed/phase2d_validation_summary.json`
- `data/processed/phase2d_validation_summary.csv`

### Phase 2E: Strategy Search Refinement / Calibration
Calibrate the strategy stack after broader validation by:
- fixing race-context leakage in pit-stop and fuel-progress grouping
- restoring a non-degenerate Miami pit-loss baseline
- cleaning up SOFT model-health behavior
- replacing the rough two-stop heuristic with a bounded search over valid pit-lap pairs

---

## ⚙️ Pipeline Modes

| Mode | Command | Fuel Correction | Degradation | Use Case |
|------|---------|---|---|---|
| **Integrated (Main)** | `streamlit run app/streamlit_app.py` | ✅ | Piecewise | Production pit decisions |
| **Strategy Demo** | `python app/demo_strategy.py` | ✅ | Piecewise | Debug strategy logic |
| **Phase 1C** | `python app/demo_phase1c.py` | ✅ | Piecewise | Validate models |
| **Phase 1B** | `python app/demo_phase1b.py` | ✅ | Linear | Analyze fuel effects |

---

## ⚠️ Limitations

- **Single circuit:** Model validated on Miami only
- **Deterministic:** No uncertainty quantification
- **No traffic model:** Doesn't account for overtaking or position effects
- **No safety cars:** Doesn't respond to VSCs or full-course yellows
- **SOFT compound:** Phase 2E removed the invalid-prediction path, but SOFT should still be treated as lower-confidence than MEDIUM/HARD
- **Validation scope:** Phase 2D is representative scenario validation, not historical backtesting or Monte Carlo race simulation

---

## 🔄 Roadmap

### Phase 1 (Complete) ✅
- ✅ Phase 1A: Data loading
- ✅ Phase 1B: Fuel correction
- ✅ Phase 1C: Degradation modeling with cliffs

### Phase 2 (In progress) 🚀
- ✅ Phase 2A: Automatic strategy search
- ✅ Phase 2B: Hybrid data blending
- ✅ Phase 2C: Scenario-based sensitivity analysis
- ✅ Phase 2D: Representative robustness validation
- ✅ Phase 2E: Strategy search refinement / calibration

---

## 🛠️ Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code standards and patterns
- Development workflow
- Testing guidelines
- Phase-specific architecture
- Pull request process

---

## 📚 Documentation

- [Phase 1 Specification](docs/phase1_spec.md)
- [Phase 1A: Data Loading](docs/phase1a_summary.md)
- [Phase 1B: Fuel Correction](docs/phase1b_fuel_correction.md)
- [Phase 1C: Degradation Modeling](docs/phase1c_degradation_modeling.md)
- [Phase 2A: Strategy Engine](docs/phase2a_strategy_engine.md)
- [Phase 2B: Hybrid Modeling](docs/phase2b_hybrid_modeling.md)
- [Phase 2C: Sensitivity Analysis](docs/phase2c_sensitivity_analysis.md)
- [Phase 2D: Broader Validation](docs/phase2d_validation.md)
- [Phase 2E: Strategy Refinement](docs/phase2e_strategy_refinement.md)

---

## 📝 License
[See LICENSE file](LICENSE)

---

**Happy strategizing!** 🏁
