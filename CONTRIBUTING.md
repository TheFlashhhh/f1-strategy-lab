# Contributing to F1 Strategy Lab

Thank you for your interest in contributing to the F1 Strategy Lab! This document provides guidelines for contributing code, documentation, and improvements.

## Code Organization

### Project Structure
```
f1-strategy-lab/
├── app/                          # Demos and applications
│   ├── streamlit_app.py         # Interactive pit strategy app
│   ├── streamlit_app_improved.py # Improved version (development)
│   ├── demo_strategy.py         # Phase 2A strategy demo
│   ├── demo_phase1b.py          # Phase 1B fuel correction demo
│   └── demo_phase1c.py          # Phase 1C degradation demo
├── src/                          # Main source code
│   ├── data/                    # Data loading and preprocessing
│   │   ├── loader.py            # DataLoader class (Phase 1A)
│   │   ├── preprocess.py        # Preprocessing (pit detection, cleaning)
│   │   ├── build_phase1_dataset.py  # Phase 1A builder
│   │   └── ingest_phase1.py     # Data ingestion utilities
│   ├── features/                # Feature engineering
│   │   ├── evaluate_degradation.py  # Unified Phase 1 interface
│   │   ├── fuel_correction.py   # Phase 1B fuel correction
│   │   ├── degradation_modeling.py  # Phase 1C cliff detection
│   │   └── evaluate_*.py        # Specialized evaluators
│   ├── simulation/              # Strategy and optimization
│   │   ├── strategy.py          # Pit-window optimization
│   │   ├── strategy_engine.py   # Automatic strategy recommendation
│   │   └── simulator.py         # Race simulation (future)
│   ├── models/                  # Model predictions (reserved)
│   ├── api/                     # API interface (reserved)
│   └── utils/                   # Utilities
├── scripts/                      # One-off utility scripts
│   ├── inspect_notebook_cells.py
│   ├── validate_fuel_correction.py
│   ├── test_phase1_integration.py
│   ├── cleanup_notebook_cells.py
│   ├── verify_data_manifest.py
│   └── README.md
├── tests/                        # Test suite
│   └── test_pipeline.py         # Main pipeline tests
├── notebooks/                    # Jupyter notebooks
│   └── eda.ipynb               # Exploratory data analysis
├── data/                         # Data (not tracked in Git)
│   ├── raw/                     # Raw data sources
│   ├── processed/               # Processed outputs
│   └── features/                # Feature data
├── docs/                         # Documentation
│   ├── phase1_spec.md
│   ├── phase1a_summary.md
│   ├── phase1b_fuel_correction.md
│   ├── phase1c_degradation_modeling.md
│   └── PHASE_1_INTEGRATION.md
└── README.md
```

## Development Workflow

### 1. Setting up the Environment

```bash
# Clone the repo
git clone https://github.com/your-fork/f1-strategy-lab.git
cd f1-strategy-lab

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (optional)
pip install pytest pytest-cov black isort mypy
```

### 2. Build Local Data

```bash
# Build Phase 1A data (required for first run)
python src/data/build_phase1_dataset.py
```

### 3. Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_pipeline.py::test_degradation_evaluation -v

# Run with coverage
pytest tests/ --cov=src
```

### 4. Making Changes

#### Style Guidelines
- **Code style:** Follow PEP 8
- **Formatting:** Use `black` for automatic formatting
- **Import sorting:** Use `isort` to organize imports
- **Type hints:** Add type annotations where possible
- **Docstrings:** Use numpy-style docstrings

#### Quick Format Check
```bash
# Format with black
black src/ app/ tests/

# Sort imports
isort src/ app/ tests/

# Type check (optional)
mypy src/ --ignore-missing-imports
```

#### Naming Conventions
- **Functions/methods:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private methods:** `_leading_underscore`
- **Internal attributes:** `_leading_underscore`

#### Phase 1 Integration Layer
The unified degradation evaluation interface provides abstraction over model types:

```python
from src.features.evaluate_degradation import evaluate_all_degradation

# Always returns a DegradationResult object with consistent API
result = evaluate_all_degradation(
    model_laps,
    use_fuel_correction=True,   # Phase 1B
    use_piecewise=True,         # Phase 1C (with linear fallback)
)

# Consistent prediction interface (works with any model type)
lap_time = result.predict_lap_time(compound="MEDIUM", tyre_life=5)

# Model inspection
info = result.get_model_info("MEDIUM")
print(f"Type: {info['model_type']}, Samples: {info['samples']}")
```

When adding new model types or degradation approaches:
1. Extend `DegradationResult` class
2. Implement `predict_lap_time()` with the unified interface
3. Add model type to `get_model_info()` reporting
4. Include fallback logic in case of insufficient data
5. Test backward compatibility with `to_legacy_linear_models()`

### 5. Working with Git

#### Commit Messages
```
Short summary (50 char max)

Optional longer explanation explaining the "why" behind
the change. Keep lines < 72 characters.

- Bullet points for multiple logical changes
- Reference issues when relevant: fixes #123
```

#### Branch Naming
- Feature: `feature/description-here`
- Bug fix: `fix/issue-description`
- Documentation: `docs/topic-here`
- Phase/milestone: `phase/<phase_name>`

### 6. Adding Features

#### Degradation Modeling
If adding a new degradation model type:
1. Create implementation in `src/features/degradation_*.py`
2. Add method to `DegradationResult` class
3. Implement evaluation logic
4. Add integration test in `tests/test_pipeline.py`
5. Document assumptions and limitations
6. Update `evaluate_all_degradation()` orchestrator

#### Data Preprocessing
If modifying data loading or preprocessing:
1. Test with both Parquet (Phase 1A) and CSV (legacy) data
2. Verify column handling (snake_case → PascalCase)
3. Update `src/data/preprocess.py` or `loader.py`
4. Add test case to `tests/test_pipeline.py`
5. Update README with any new fields or requirements

#### Strategy Optimization
If modifying `src/simulation/strategy.py`:
1. Test with multiple compounds and race states
2. Verify pit-window enumeration completeness
3. Test edge cases (final lap, low fuel, etc.)
4. Add benchmark for performance regression
5. Document assumptions about feasibility checking

### 7. Documentation

#### Code Comments
```python
def foo(x: int) -> int:
    """Brief one-line summary.
    
    Longer description if needed, explaining purpose and usage.
    
    Parameters
    ----------
    x : int
        Description of parameter
        
    Returns
    -------
    int
        Description of return value
        
    Raises
    ------
    ValueError
        When x is negative
        
    Notes
    -----
    Additional details about implementation or assumptions.
    
    Examples
    --------
    >>> foo(5)
    10
    """
```

#### README Updates
- Update main README if feature affects user workflows
- Add commands/examples if introducing new scripts
- Document data format changes
- Update architecture diagram if repo structure changes

#### Phase Documentation
Each phase should have a corresponding doc:
- `docs/phase1_spec.md` — Phase 1 specification
- `docs/phase1a_summary.md` — Phase 1A data loading
- `docs/phase1b_fuel_correction.md` — Phase 1B methodology
- `docs/phase1c_degradation_modeling.md` — Phase 1C modeling
- Create `docs/phase2_spec.md` when starting Phase 2

### 8. Pull Request Process

1. **Ensure tests pass:**
   ```bash
   pytest tests/
   ```

2. **Format code:**
   ```bash
   black src/ app/ tests/
   isort src/ app/ tests/
   ```

3. **Create descriptive PR title:**
   - `[Phase X] Feature description`
   - `[Fix] Issue description`
   - `[Docs] Documentation update`

4. **PR checklist:**
   - [ ] Tests added/updated
   - [ ] Code formatted with black
   - [ ] Imports sorted with isort
   - [ ] Documentation updated
   - [ ] No breaking changes (or justified)
   - [ ] Related issue referenced

5. **Code review:**
   - Changes reviewed for correctness
   - Tests verify new functionality
   - Documentation is clear
   - Performance impact assessed

## Key Principles

### 1. Backward Compatibility
- Maintain existing APIs when possible
- Use deprecation warnings for breaking changes
- Test legacy data paths (CSV fallback)
- Provide migration guides for interface changes

### 2. Data Integrity
- Generated data is **not tracked in Git**
- Each user builds local dataset
- Manifests enable reproducibility
- Version major data format changes clearly

### 3. Unified Interfaces
- Use `DegradationResult` abstraction for model access
- Abstract model type implementations from strategy code
- Provide fallback mechanisms (linear model for piecewise)
- Test backward compatibility

### 4. Phase Separation
- Phase 1: Pure empirical modeling (no optimization)
- Phase 2: Strategy and optimization logic
- Keep phases independent where possible
- Document cross-phase dependencies

### 5. Testing
- Test with real race data
- Test fallback mechanisms
- Verify performance benchmarks
- Include edge cases

## Release Process

1. **Milestone packaging:**
   - `v1.0` — Phase 1 complete (data + models)
   - `v2.0` — Phase 2A complete (automatic strategy)
   - `v2.1` — Phase 2B (hybrid modeling)
   - `v3.0` — Phase 2 complete (uncertainty + circuits)

2. **Changelog:**
   - Document new features
   - Note breaking changes
   - Include performance improvements
   - Credit contributors

3. **Tags:**
   ```bash
   git tag -a v1.0 -m "Phase 1 complete"
   git push origin v1.0
   ```

## Questions?

- Check existing issues and documentation
- Review docs/ folder for detailed explanations
- Look at test cases for usage examples
- Check app/ demos for integration patterns

## Code of Conduct

Be respectful, inclusive, and constructive. We welcome contributions from all backgrounds and experience levels.

---

Happy contributing! 🏁
