# Phase 1A Data Plan: Miami-Focused Dataset Foundation

## Overview

Phase 1A builds a reliable, auditable data foundation focused on **Miami Grand Prix** and current-season context. This stage is strictly about **data ingestion, schema standardization, and metadata tracking**—not modeling changes.

---

## Design Rationale

### Why Miami?

**Miami Grand Prix (2022–present)** is chosen as the target race context because:

1. **Consistent modern-regulation F1** (2022–2025 with DRS, 2026 active-aero rules)
2. **Recent historical data available** (4 years of historical races)
3. **Smaller thermal window** vs. hot circuits (Abu Dhabi, Japan)
4. **Known pit-stop patterns** (predictable fuel and tyre management)
5. **Preparation for 2026 entry** (active-aero regulation change)

### Why Attempt 2026 Pre-Miami?

The 2026 season introduces **active-aero regulations**. To build predictive models that generalize to 2026:

- **Schedule-driven, completion-aware selection:** Automatically fetches the 2026 F1 schedule from FastF1, identifies the Miami event by name, selects all race weekends strictly before it, and only attempts races already completed as of the build date
- Use pre-Miami races as calibration data if available
- Fail gracefully if races haven't occurred yet or data unavailable

**Dynamic race selection ensures:**
- All completed race weekends before Miami are automatically included (e.g., Australia, China, Japan)
- No manual race list maintenance required
- Extensible to other years/contexts without code changes
- Only races with available data are ingested (completion-aware filtering prevents attempting future races)

**Critical:** If 2026 data is unavailable or incomplete:
- **Do NOT crash**
- **Do NOT fabricate data**
- **Log clearly** that data is unavailable
- **Record all discovered/skipped races in manifest**
- **Proceed with historical data only**

---

## Data Schema (Canonical)

All ingested sessions are standardized to this schema:

| Field | Type | Description |
|-------|------|-------------|
| `season` | int | Race year (e.g., 2022, 2026) |
| `event_name` | str | Grand Prix name (e.g., "Miami Grand Prix") |
| `circuit_name` | str | Circuit/venue name |
| `session_name` | str | Session type (e.g., "Race", "Qualifying") |
| `driver` | str | Driver name or code |
| `team` | str | Team/constructor name |
| `lap_number` | int | Lap count |
| `lap_time` | float | Lap time in seconds |
| `stint` | int | Stint number (pit-stop sequence) |
| `compound` | str | Tyre compound (SOFT, MEDIUM, HARD) |
| `tyre_life` | int | Laps on current tyre set |
| `track_status` | int/str | Track condition (1=green flag, etc.) |
| `position` | int | Driver position in lap |
| `pit_in_time` | float | Pit entry time (seconds) or null |
| `pit_out_time` | float | Pit exit time (seconds) or null |
| `is_accurate` | bool | Lap timing accurate (true if source reliable) |
| `deleted` | bool | Lap deleted from official records |

### Metadata Fields

**Three mandatory metadata fields** are added to every row:

1. **`data_group`** (str)
   - `"miami_historical"` — Miami GP 2022–2025
   - `"season_2026_pre_miami"` — 2026 races before Miami
   - **Purpose:** Track data origin and allow filtering by context

2. **`regulation_era`** (str)
   - `"pre_2022"` — Old regulations (not used in Phase 1A)
   - `"2022_2025"` — Current DRS-era regulations (Miami historical)
   - `"2026_active_aero"` — Active-aero (2026 pre-Miami)
   - **Purpose:** Enable regulation-aware filtering and modeling

3. **`target_race_context`** (str)
   - `"miami"` — All Phase 1A data
   - **Purpose:** Future expansion if other race contexts added

---

## Storage Format

**Parquet only** (no CSV):

```
data/raw/
├── miami_historical/
│   ├── 2022_miami_grand_prix_race.parquet
│   ├── 2023_miami_grand_prix_race.parquet
│   ├── 2024_miami_grand_prix_race.parquet
│   ├── 2025_miami_grand_prix_race.parquet
│   └── combined.parquet  (all Miami races concatenated)
├── season_2026_pre_miami/
│   ├── 2026_australian_grand_prix_race.parquet
│   ├── 2026_chinese_grand_prix_race.parquet
│   ├── 2026_japanese_grand_prix_race.parquet
│   └── combined.parquet  (all 2026 pre-Miami races)
└── manifest.json  (audit log)
```

**Note:** 2026 pre-Miami races are selected dynamically from the F1 schedule. The actual races included depend on the 2026 calendar and which races have been completed. As of April 2026, the races before Miami are Australia, China, and Japan.

**Why Parquet?**
- Efficient columnar compression
- Preserves schema and types
- Better performance for large datasets
- Standard for data pipelines

---

## Manifest: Auditable Ingestion Log

**File:** `data/raw/manifest.json`

Tracks every ingestion attempt (success or failure):

```json
{
  "generated_at": "2026-04-07T12:34:56.789Z",
  "total_sessions_attempted": 7,
  "total_sessions_succeeded": 4,
  "total_rows_ingested": 45230,
  "sessions": [
    {
      "season": 2022,
      "event_name": "Miami Grand Prix",
      "session_name": "Race",
      "data_group": "miami_historical",
      "regulation_era": "2022_2025",
      "target_race_context": "miami",
      "success": true,
      "row_count": 10456,
      "missing_fields": [],
      "error_message": null,
      "timestamp": "2026-04-07T12:34:00Z"
    },
    {
      "season": 2026,
      "event_name": "Bahrain Grand Prix",
      "session_name": "Race",
      "data_group": "season_2026_pre_miami",
      "regulation_era": "2026_active_aero",
      "target_race_context": "miami",
      "success": false,
      "row_count": 0,
      "missing_fields": [],
      "error_message": "2026 Bahrain race data not yet available",
      "timestamp": "2026-04-07T12:34:30Z"
    }
  ]
}
```

**Purpose:** Complete transparency on data completeness and quality.
- No silent failures
- All missing fields recorded
- Error messages logged
- Timestamp for reproducibility

---

## Missing Data Handling

### Explicit Rules

If **Miami year is missing** (e.g., 2024 unavailable):
- Log as failed session in manifest
- Record error message
- Continue with other years

If **2026 data is unavailable**:
- Expected (races may not have occurred)
- Log gracefully (not an error, just informational)
- Continue with historical data only
- Do NOT crash or fail the build

If **individual fields are missing** (e.g., no `IsAccurate` column):
- Fill with sensible defaults (e.g., assume `True`)
- Record missing field name in manifest
- Continue processing

If **session returns 0 laps**:
- Log as failure
- Skip to next session
- Continue execution

### Examples

**Example 1: Missing Miami 2024**
```
2024 Miami Race: FAILED
  Error: "No laps returned from FastF1 for 2024 Miami"
  Rows: 0
```
→ Recorded in manifest; next year processed

**Example 2: Missing 2026 pre-Miami data**
```
2026 Bahrain Race: SKIPPED
  Reason: "2026 Bahrain data not yet available"
```
→ Not treated as error; builds continues with Miami historical

**Example 3: Missing field in raw data**
```
2025 Miami Race: SUCCESS
  Rows: 9856
  Missing fields: ["IsAccurate", "Deleted"]
  Note: Filled with True (assume accurate/not deleted)
```
→ Recorded in manifest for transparency

---

## Regulation-Era Differences (Known Limitations)

### 2022–2025 (DRS Era)
- **Fuel effect on lap time:** 0.03–0.05 s/kg (Phase 1 calibration)
- **Tyre degradation:** Compound-specific, approximately linear over stint
- **Tyre cliff:** Age-dependent (5–10 laps depending on circuit)
- **Pit loss:** ~15–18s (empirical estimate from race data)

### 2026 (Active-Aero Era)
- **Active-aero systems** change downforce dynamically
- **Pace variation:** May be higher due to setup changes
- **Tyre behavior:** Potentially different due to chassis/weight changes
- **Degradation slopes:** Require recalibration
- **Undercut effectiveness:** May differ due to aero regulation

**Implication:** Models trained on 2022–2025 data may require adjustment for 2026. Phase 1A ingests both; Phase 2 will perform era-specific modeling.

---

## Data Sources

**Primary source:** [FastF1](https://github.com/theOehrly/Fast-F1)
- Official F1 timing data aggregator
- Cached locally in `data/raw/fastf1_cache/`
- Respects FIA and F1 terms of service

**Data granularity:** Lap-level (one row per driver per lap)

**Completeness:** Race sessions only (not qualifying or practice)

---

## Phase 1A Objectives

### Completed
- ✅ Design canonical lap-level schema
- ✅ Implement defensive ingestion (handle missing data without crashing)
- ✅ Create manifest for full auditability
- ✅ Store in Parquet format
- ✅ Add metadata fields (data_group, regulation_era, target_race_context)

### Outcome
- **Auditable, trackable dataset** ready for Phase 1B modeling
- **Clear documentation** of schema and limitations
- **Manifest logs** showing exactly what succeeded/failed
- **Regulation-era tagging** for future multi-era modeling

---

## Usage Example

### Build the dataset
```bash
python src/data/build_phase1_dataset.py
```

### Load Miami historical data
```python
import pandas as pd

miami_2022 = pd.read_parquet("data/raw/miami_historical/2022_miami_grand_prix_race.parquet")
miami_all = pd.read_parquet("data/raw/miami_historical/combined.parquet")

# Check data_group and regulation_era
print(miami_all['data_group'].unique())      # ["miami_historical"]
print(miami_all['regulation_era'].unique())  # ["2022_2025"]
```

### Review ingestion manifest
```bash
cat data/raw/manifest.json
```

---

## Next Steps (Phase 1B–1C)

Phases after Phase 1A will use this dataset to:

1. **Phase 1B: Fuel Correction**
   - Use Miami historical data to fit fuel-effect model
   - Validate on held-out Miami stints

2. **Phase 1C: Degradation Modeling**
   - Fit per-compound degradation curves
   - Detect tyre cliffs on Miami data

3. **Phase 2: Multi-Race Generalization**
   - Add more circuits beyond Miami
   - Validate across different track conditions
   - Prepare 2026 active-aero models

---

## Known Limitations

1. **Miami-only historical data:** Findings not validated on other circuits yet
2. **FastF1 source reliability:** Depends on FIA timing data quality
3. **Missing fields:** Some sessions may lack `IsAccurate` or `Deleted` flags
4. **2026 data delay:** If races haven't occurred, that era unavailable
5. **Regulation differences:** 2026 active-aero cannot be directly compared to 2022–2025 DRS
6. **No real-time ingestion:** Data available only after race completion

---

## Summary

Phase 1A builds a **defensible, auditable, metadata-rich dataset** focused on Miami from 2022–2025, with optional 2026 pre-Miami races. All ingestion attempts (success and failure) are logged in a manifest for full transparency. The standardized Parquet format and metadata fields enable clean, regulation-era-aware downstream modeling.
