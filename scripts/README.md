# Scripts Directory

Utility scripts for data validation, notebook maintenance, and pipeline testing.

## Quick Reference

| Script | Purpose |
|--------|---------|
| `validate_fuel_correction.py` | Verify Phase 1B fuel correction implementation |
| `test_phase1_integration.py` | Quick test of complete Phase 1 pipeline |
| `verify_data_manifest.py` | Validate data ingestion manifest |
| `inspect_notebook_cells.py` | Inspect EDA notebook cell structure |
| `cleanup_notebook_cells.py` | Remove unnecessary cells from EDA notebook |
| `check_2026_files.py` | Check 2026 pre-Miami parquet file sizes |

## Running Scripts

All scripts should be run from the **project root**:

```bash
python scripts/<script_name>.py
```

## Script Descriptions

### validate_fuel_correction.py
Comprehensive validation of Phase 1B fuel correction:
- Verifies FuelCorrectedLapTime column is created
- Compares raw vs corrected degradation models
- Confirms fuel adjustment is applied correctly
- Validates error handling for missing columns

### test_phase1_integration.py
Quick integration test of the complete Phase 1 pipeline:
- Loads Miami historical data (Phase 1A)
- Applies fuel correction (Phase 1B)
- Tests piecewise degradation models (Phase 1C)
- Verifies backward compatibility for linear models
- Tests predictions for all compounds

### verify_data_manifest.py
Validates data ingestion manifest integrity:
- Checks total session counts
- Reports ingested row totals
- Lists 2026 pre-Miami records with success status

### inspect_notebook_cells.py
Inspect the structure of the EDA notebook:
- Shows total cell count
- Displays first 10 cells with their IDs and types
- Useful for understanding notebook organization

### cleanup_notebook_cells.py
Removes unnecessary cells from the EDA notebook:
- Keeps core cells (0-21) and final summary (35-38)
- Deletes redundant/debug cells (22-34)
- Preserves essential analyses

### check_2026_files.py
Checks individual parquet file sizes in the 2026 pre-Miami directory:
- Lists each file with row count
- Reports total laps loaded
- Useful for data validation before processing
