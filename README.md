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

## 2. Data
- Primary dataset: [data/raw/2020_abudhabi_race.csv](data/raw/2020_abudhabi_race.csv)
- Granularity: lap-level records (driver, lap number, lap time, compound, tyre life, stint, accuracy and track-status fields)
- Notebook source: [notebooks/eda.ipynb](notebooks/eda.ipynb)

## Tech Stack
- Python
- pandas
- numpy
- matplotlib
- Streamlit
- Modular pipeline design (preprocessing, feature engineering, simulation)

## 3. Method
1. Pit-stop detection:
Pit stops are identified when stint changes for a given driver between consecutive laps (with each driver’s first lap explicitly not treated as a pit stop).

2. Lap cleaning and model filtering:
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

## 4. Key Findings
- A linear degradation model identified positive degradation on MEDIUM, while HARD behaved approximately flat in this sample.
- Empirical pit-loss estimation yields a median penalty of ~15.45s for this race sample.
- In the baseline deterministic simulation, a driver on MEDIUM tyres with tyre life 5 and 25 laps remaining yields an optimal switch to HARD at approximately lap 6.
- The model produces a monotonic decision policy: increasing tyre age shifts optimal pit timing earlier, with a threshold beyond which immediate pitting is optimal.
- The decision boundary emerges from the trade-off between degradation slope and pit-loss penalty, yielding an interpretable threshold policy.

## Results Snapshot
- Median pit-loss: ~15.45s
- Optimal pit (baseline scenario): Lap 6
- Decision policy: deterministic, threshold-based

The system processes ~1,000 lap-level observations and evaluates ~20-25 candidate pit strategies per scenario with near-instant response time in the local app.


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

### Run the notebook
```powershell
jupyter notebook notebooks/eda.ipynb
```

Optional full execution check:
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

### Run the module-based demo
```powershell
python app/demo_strategy.py
```

## 7. Limitations
- Single-race focus (Abu Dhabi 2020), so findings are not yet validated across multiple tracks or race conditions.
- Degradation and strategy layers are deterministic and linear; uncertainty and traffic effects are not modeled.
- Pit-loss estimation is a proxy based on lap windows, not a full causal decomposition of in-lap, stationary, and out-lap components.
- A local interactive Streamlit interface is included; no production deployment or real-time serving layer is currently implemented.
- The current implementation serves as a deterministic baseline for future stochastic and opponent-aware strategy modeling.

## 8. Next Steps
- Evaluate the same pipeline across multiple races and circuits.
- Add uncertainty-aware simulation (pace variability, safety-car states, and traffic interaction).
- Compare linear degradation against piecewise/nonlinear alternatives.
- Add unit tests for preprocessing, pit-loss estimation, and optimization components.
