#!/usr/bin/env python
"""Quick integration test for Phase 1 pipeline.

Run from project root: python scripts/test_phase1_integration.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.evaluate_degradation import evaluate_all_degradation
from src.data.loader import load_data
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns

print('Testing integrated Phase 1 pipeline...')
df_raw = load_data(dataset='miami_historical', project_root=ROOT)
df = select_relevant_columns(df_raw)
df = detect_pit_stops(df)
clean_df = clean_laps(df)
model_df = build_model_df(clean_df)

# Run full integration
result = evaluate_all_degradation(model_df, use_fuel_correction=True, use_piecewise=True)

print('\n✓ Unified interface works!')
print(f'  Compounds available: {len(result.compounds)}')

# Test backward compatibility
legacy = result.to_legacy_linear_models()
print(f'✓ Legacy format conversion: {len(legacy)} compounds')

# Test predictions
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    info = result.get_model_info(compound)
    pred = result.predict_lap_time(compound, 5)
    model_type = info['model_type']
    print(f'  {compound}: {model_type} | tyre-life 5 -> {pred:.2f}s')

print('\n✓ Full Phase 1 integration verified!')
