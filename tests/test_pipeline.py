#!/usr/bin/env python
"""Test suite for F1 Strategy Lab.

Run all tests: pytest tests/
Run specific module: pytest tests/test_degradation.py -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports():
    """Test that all main modules can be imported."""
    from src.data import loader, preprocess
    from src.features import build_features, fuel_correction, evaluate_degradation
    from src.simulation import strategy
    
    assert loader is not None
    assert preprocess is not None
    assert build_features is not None
    assert fuel_correction is not None
    assert evaluate_degradation is not None
    assert strategy is not None


def test_phase1_imports():
    """Test Phase 1-specific imports."""
    from src.features.evaluate_degradation import evaluate_all_degradation
    from src.data.loader import DataLoader
    from src.simulation.strategy_validation import build_representative_scenario_suite
    
    assert evaluate_all_degradation is not None
    assert DataLoader is not None
    assert build_representative_scenario_suite is not None


def test_phase2d_scenario_suite():
    """Test that the Phase 2D scenario suite is representative but compact."""
    from src.simulation.strategy_validation import build_representative_scenario_suite

    suite = build_representative_scenario_suite()
    assert 9 <= len(suite) <= 15

    compounds = {scenario.current_compound for scenario in suite}
    age_buckets = {scenario.tyre_age_bucket for scenario in suite}
    lap_buckets = {scenario.laps_remaining_bucket for scenario in suite}

    assert compounds == {"SOFT", "MEDIUM", "HARD"}
    assert age_buckets == {"low", "medium", "high"}
    assert lap_buckets == {"short", "medium", "long"}


def test_data_loading():
    """Test that Miami historical data can be loaded."""
    from src.data.loader import DataLoader
    
    loader = DataLoader(project_root=ROOT)
    df = loader.load_data(dataset="miami_historical")
    
    assert df is not None
    assert len(df) > 0
    assert "LapNumber" in df.columns or "lap_number" in df.columns.str.lower()


def test_preprocessing():
    """Test preprocessing pipeline."""
    from src.data.loader import DataLoader
    from src.data.preprocess import (
        select_relevant_columns,
        detect_pit_stops,
        clean_laps,
        build_model_df,
    )
    
    loader = DataLoader(project_root=ROOT)
    df_raw = loader.load_data(dataset="miami_historical")
    
    df = select_relevant_columns(df_raw)
    assert df is not None
    
    pit_df = detect_pit_stops(df)
    assert pit_df is not None
    
    clean_df = clean_laps(pit_df)
    assert clean_df is not None
    
    model_df = build_model_df(clean_df)
    assert model_df is not None
    assert len(model_df) > 0


def test_degradation_evaluation():
    """Test unified degradation evaluation interface."""
    from src.features.evaluate_degradation import evaluate_all_degradation
    from src.data.loader import DataLoader
    from src.data.preprocess import (
        select_relevant_columns,
        detect_pit_stops,
        clean_laps,
        build_model_df,
    )
    
    loader = DataLoader(project_root=ROOT)
    df_raw = loader.load_data(dataset="miami_historical")
    
    df = select_relevant_columns(df_raw)
    pit_df = detect_pit_stops(df)
    clean_df = clean_laps(pit_df)
    model_df = build_model_df(clean_df)
    
    result = evaluate_all_degradation(
        model_df,
        use_fuel_correction=True,
        use_piecewise=True,
    )
    
    assert result is not None
    assert len(result.compounds) > 0
    
    # Test prediction API
    for compound in result.compounds:
        info = result.get_model_info(compound)
        assert info is not None
        assert "model_type" in info
        assert "samples" in info
        
        # Test prediction
        lap_time = result.predict_lap_time(compound, tyre_life=5)
        assert lap_time is not None
        assert lap_time > 0


def test_backward_compatibility():
    """Test backward compatibility with legacy linear models."""
    from src.features.evaluate_degradation import evaluate_all_degradation
    from src.data.loader import DataLoader
    from src.data.preprocess import (
        select_relevant_columns,
        detect_pit_stops,
        clean_laps,
        build_model_df,
    )
    
    loader = DataLoader(project_root=ROOT)
    df_raw = loader.load_data(dataset="miami_historical")
    
    df = select_relevant_columns(df_raw)
    pit_df = detect_pit_stops(df)
    clean_df = clean_laps(pit_df)
    model_df = build_model_df(clean_df)
    
    result = evaluate_all_degradation(model_df)
    legacy = result.to_legacy_linear_models()
    
    assert legacy is not None
    assert isinstance(legacy, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
