"""Build Phase 1A dataset: Miami historical + 2026 pre-Miami races.

Run as: python src/data/build_phase1_dataset.py

This script:
1. Ingests Miami GP races (2022–2025)
2. Attempts 2026 pre-Miami races (fails gracefully if unavailable)
3. Saves to Parquet format
4. Generates manifest.json for auditing
"""

import logging
import sys
from pathlib import Path

from ingest_phase1 import DataIngestPipeline


def main():
    """Run Phase 1A ingestion pipeline."""
    # Setup
    project_root = Path(__file__).parents[2]
    sys.path.insert(0, str(project_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize pipeline
    print("\n" + "=" * 70)
    print("PHASE 1A DATA BUILD")
    print("=" * 70)
    print("Target: Miami + 2026 pre-Miami races")
    print("Format: Parquet with metadata")
    print()

    pipeline = DataIngestPipeline(cache_dir=project_root / "data" / "raw" / "fastf1_cache")

    # 1. Miami Historical (2022–2025)
    print("Step 1: Ingesting Miami historical (2022–2025)...")
    miami_datasets = pipeline.ingest_miami_historical()

    if miami_datasets:
        miami_paths = pipeline.save_datasets(
            miami_datasets,
            group_name="miami_historical",
            output_dir=project_root / "data" / "raw"
        )
        print(f"[OK] Saved {len(miami_paths)} Miami datasets\n")
    else:
        print("[X] No Miami datasets ingested\n")

    # 2. 2026 Pre-Miami (schedule-driven, completion-aware)
    print("Step 2: Ingesting 2026 pre-Miami races (schedule-driven, completion-aware)...")
    season_2026_datasets = pipeline.ingest_2026_pre_miami()

    # Count stats from manifest for 2026 pre-Miami (exclude batch-level placeholder records)
    all_2026_records = [r for r in pipeline.manifest if r.data_group == "season_2026_pre_miami"]
    # Actual race records (not batch-level failures like schedule/Miami not found)
    pre_miami_race_records = [r for r in all_2026_records if r.session_name == "Race"]
    # Batch-level failures (if any)
    batch_level_failures = [r for r in all_2026_records if r.session_name == "Batch"]
    
    pre_miami_successful = [r for r in pre_miami_race_records if r.success]
    pre_miami_failed = [r for r in pre_miami_race_records if not r.success]

    # Show batch-level failures if present
    if batch_level_failures:
        for r in batch_level_failures:
            if r.error_message:
                print(f"  [Batch-level] {r.event_name}: {r.error_message}")
    
    if pre_miami_race_records:
        print(f"  Discovered: {len(pre_miami_race_records)} race(s)")
        print(f"  Ingested:   {len(pre_miami_successful)} successfully with {sum(r.row_count for r in pre_miami_successful):,} laps")
        if pre_miami_failed:
            print(f"  Unavailable/Failed: {len(pre_miami_failed)} race(s)")
            for r in pre_miami_failed:
                if r.error_message:
                    print(f"    - {r.event_name}: {r.error_message}")
    else:
        print("  No 2026 pre-Miami races discovered")

    if season_2026_datasets:
        season_2026_paths = pipeline.save_datasets(
            season_2026_datasets,
            group_name="season_2026_pre_miami",
            output_dir=project_root / "data" / "raw"
        )
        print(f"[OK] Saved {len(season_2026_paths)} 2026 dataset file(s)\n")
    else:
        if pre_miami_records:
            print("→ No 2026 pre-Miami races completed/available for ingestion\n")
        else:
            print("→ No 2026 pre-Miami races in schedule\n")

    # 3. Save manifest
    print("Step 3: Saving manifest...")
    manifest_path = pipeline.save_manifest(project_root / "data" / "raw" / "manifest.json")
    print(f"[OK] Manifest saved to {manifest_path}\n")

    # 4. Print summary
    pipeline.print_summary()

    # 5. Output locations
    print("\nOutput locations:")
    print(f"  Miami historical: data/raw/miami_historical/")
    print(f"  2026 pre-Miami:   data/raw/season_2026_pre_miami/")
    print(f"  Manifest:         data/raw/manifest.json")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
