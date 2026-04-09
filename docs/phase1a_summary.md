# Phase 1A Implementation Summary

## Overview

Phase 1A successfully builds a **reliable, auditable data foundation** focused on Miami Grand Prix with optional 2026 pre-Miami data. This stage is purely about data ingestion, schema standardization, and metadata tracking—**no modeling changes**.

---

## What Was Built

### 1. Extended Ingestion Layer: `src/data/ingest_phase1.py`

A production-grade data ingestion pipeline with:

- **FastF1 integration:** Loads F1 race sessions with robust error handling
- **Schema standardization:** Maps raw FastF1 data to canonical 20-field schema
- **Schedule-driven, completion-aware 2026 pre-Miami selection:** Dynamically fetches the 2026 event schedule, identifies Miami by name, selects all race weekends before it, and only attempts races already completed (no manual race list, no future race attempts)
- **Metadata fields:** Adds `data_group`, `regulation_era`, `target_race_context` to every row
- **Defensive error handling:** Handles missing data gracefully without crashing
- **Complete manifest generation:** Full audit log of all ingestion attempts including discovered, ingested, and skipped races

**Key features:**
- Schedule-driven logic ensures all completed races before Miami are included (not a hardcoded list)
- Completion-aware filtering prevents attempting future races
- All discovered races logged to manifest (no silent skipping)
- Won't crash if 2026 data unavailable (expected)
- Records missing fields and errors in manifest
- Converts timedeltas to seconds for compatibility

### 2. Build Script: `src/data/build_phase1_dataset.py`

Orchestrates the full Phase 1A ingestion pipeline:

```bash
python src/data/build_phase1_dataset.py
```

Handles:
- Miami historical (2022–2025)
- 2026 pre-Miami races (optional, fails gracefully)
- Saves to Parquet (compressed, typed)
- Generates manifest.json

### 3. Data Plan Documentation: `docs/phase1_data_plan.md`

Comprehensive technical specification covering:
- Why Miami is chosen (consistent modern-regulation F1)
- Why 2026 data is attempted but not guaranteed
- Canonical 20-field schema definition
- Metadata field purposes and values
- Parquet storage rationale
- Manifest structure and audit trail
- Missing data handling rules
- Regulation-era differences (2022–2025 DRS vs. 2026 active-aero)
- Known limitations
- Usage examples

---

## Data Infrastructure

### Directory Structure

```
data/raw/
├── 2020_abudhabi_race.csv              (original)
├── 2024_bahrain_race.csv               (original)
├── fastf1_cache/                       (FastF1 cached data)
├── manifest.json                       (ingestion audit log)
├── miami_historical/
│   ├── 2022_miami_grand_prix_race.parquet   (1,057 laps)
│   ├── 2023_miami_grand_prix_race.parquet   (1,138 laps)
│   ├── 2024_miami_grand_prix_race.parquet   (1,111 laps)
│   ├── 2025_miami_grand_prix_race.parquet   (1,005 laps)
│   └── combined.parquet                     (4,311 laps total)
└── season_2026_pre_miami/
    ├── 2026_australian_grand_prix_race.parquet   (1,007 laps)
    ├── 2026_chinese_grand_prix_race.parquet      (924 laps)
    ├── 2026_japanese_grand_prix_race.parquet     (1,107 laps)
    └── combined.parquet                          (3,038 laps total)
```

### Data Volumes (Phase 1A Results)

| Dataset | Events | Total Rows | Format | Status |
|---------|--------|-----------|--------|--------|
| Miami Historical | 2022–2025 (4 years) | 4,311 | Parquet | ✅ Complete |
| 2026 Pre-Miami | Australia, China, Japan (schedule-driven) | 3,038 | Parquet | ✅ Complete |
| **Total** | — | **7,349** | — | — |

**Note:** 2026 pre-Miami races are selected dynamically from the FastF1 schedule. As of April 9, 2026, the three races before Miami (May 3, 2026) are Australia, China, and Japan.
      "success": true,
      "row_count": 1057,
      "missing_fields": [],
      "error_message": null
    },
    ...
  ]
}
```

---

## Canonical Schema (20 Fields)

### Base Fields
- `season`, `event_name`, `circuit_name`, `session_name`
- `driver`, `team`, `lap_number`, `lap_time`, `stint`
- `compound`, `tyre_life`, `track_status`, `position`
- `pit_in_time`, `pit_out_time`
- `is_accurate`, `deleted`

### Metadata Fields
- `data_group` — origin tag (`miami_historical`, `season_2026_pre_miami`)
- `regulation_era` — regulation version (`2022_2025`, `2026_active_aero`)
- `target_race_context` — always `miami` in Phase 1A

---

## Design Decisions

### Why Parquet?
- Efficient columnar compression (vs. CSV)
- Schema preservation and type safety
- Standard for data engineering pipelines
- Better I/O performance for phase 1B+ modeling

### Why Manifest?
- Complete transparency (no silent failures)
- Audit trail of ingestion attempts
- Records missing fields and errors
- Reproducibility across runs
- **Completion-aware:** Only re-downloads incomplete races; skips successful ones (efficient multi-run handling)

### Why Metadata Fields?
- **`data_group`:** Track data origin and temporal context
- **`regulation_era`:** Enable era-specific modeling (DRS vs. active-aero)
- **`target_race_context`:** Future-proof for multi-race expansion

### Why Miami?
1. Consistent modern-regulation F1 (since 2022)
2. Sufficient historical data (4 years available)
3. Smaller thermal window (predictable fuel behavior)
4. Known pit-stop patterns
5. Good test bed before 2026 active-aero transition

---

## Key Engineering Constraints Met

✅ **Modular, reusable code** — Pipeline can be extended for other circuits
✅ **No notebook-only logic** — Pure Python scripts, executable
✅ **No fake data** — All data from official F1 sources
✅ **Defensive error handling** — Missing 2026 data doesn't crash pipeline
✅ **Auditable ingestion** — Manifest logs every attempt
✅ **Metadata-rich** — All rows tagged with context
✅ **Type-safe storage** — Parquet preserves schema
✅ **Documentation-first** — phase1_data_plan.md specifies design upfront

---

## Next Steps (Phase 1B–1C)

This data foundation enables:

1. **Phase 1B: Fuel Correction**
   - Train hybrid physical-prior + empirical model on Miami data
   - Validate cross-season stability

2. **Phase 1C: Degradation Modeling**
   - Linear and piecewise degradation curves per compound
   - Tyre-cliff detection on Miami stints

3. **Phase 2: Multi-Circuit Generalization**
   - Add other circuits (Bahrain, Singapore, etc.)
   - Validate regulation-era differences
   - Prepare 2026 active-aero transition

---

## Known Limitations

1. **Miami-only:** Findings not yet validated on other circuits
2. **FastF1 dependency:** Relies on FIA timing data quality
3. **Missing fields possible:** Some sessions may lack `IsAccurate`, `Deleted`
4. **2026 data delay:** Unavailable until races complete
5. **Regulation differences:** 2022–2025 DRS ≠ 2026 active-aero (requires separate models)
6. **No real-time:** Data available only after race completion

---

## Usage in Phase 1B+ 

Load Miami combined dataset:
```python
import pandas as pd
miami = pd.read_parquet("data/raw/miami_historical/combined.parquet")
print(f"Miami: {len(miami)} laps across {miami['season'].nunique()} seasons")

# Filter by regulation era
drs_era = miami[miami['regulation_era'] == '2022_2025']

# Verify metadata
assert miami['data_group'].unique() == ['miami_historical']
assert miami['regulation_era'].isin(['2022_2025', '2026_active_aero']).all()
```

---

## Files Created/Modified

### Created
- `src/data/ingest_phase1.py` — Extended ingestion pipeline
- `src/data/build_phase1_dataset.py` — Build script and orchestration
- `docs/phase1_data_plan.md` — Technical specification

### Modified
- `README.md` — Added Phase 1A data references and build instructions
- `data/raw/manifest.json` — Ingestion audit log (auto-generated)

### Generated Data
- `data/raw/miami_historical/*.parquet` — 5 files (2022–2025 + combined)
- `data/raw/season_2026_pre_miami/*.parquet` — 2026 races (if available)

---

## Summary

Phase 1A delivers a **defensible, auditable, regulation-era-aware data layer** ready for fuel correction and degradation modeling. The pipeline is modular, handles missing data gracefully, and provides complete transparency through the manifest log. Miami historical data (4,311 laps, 2022–2025) is clean, standardized, and ready for Phase 1B analysis.
