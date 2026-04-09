"""Phase 1A data ingestion for Miami historical + 2026 pre-Miami datasets.

This module:
- Loads race sessions via FastF1
- Standardizes schema with metadata
- Handles missing data defensively
- Tracks all successes/failures in manifest
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import fastf1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Manifest entry for a single session attempt."""
    season: int
    event_name: str
    session_name: str
    data_group: str
    regulation_era: str
    target_race_context: str
    success: bool
    row_count: int = 0
    missing_fields: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


class DataIngestPipeline:
    """Manages Phase 1A data ingestion and manifest tracking."""

    # Standard lap-level schema
    CANONICAL_SCHEMA = [
        "season",
        "event_name",
        "circuit_name",
        "session_name",
        "driver",
        "team",
        "lap_number",
        "lap_time",
        "stint",
        "compound",
        "tyre_life",
        "track_status",
        "position",
        "pit_in_time",
        "pit_out_time",
        "is_accurate",
        "deleted",
        # Metadata
        "data_group",
        "regulation_era",
        "target_race_context",
    ]

    def __init__(self, cache_dir: Path | str = "data/raw/fastf1_cache"):
        """Initialize pipeline with FastF1 cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        self.manifest: list[SessionRecord] = []

    def load_session(
        self,
        year: int,
        gp: str,
        session: str = "R",
    ) -> pd.DataFrame | None:
        """
        Load a single F1 session via FastF1.

        Args:
            year: Season year
            gp: Grand Prix name
            session: Session code (R=Race, Q=Qualifying, etc.)

        Returns:
            DataFrame with lap data, or None if load fails.
        """
        try:
            logger.info(f"Attempting to load {year} {gp} {session}...")
            session_obj = fastf1.get_session(year, gp, session)
            session_obj.load()
            laps = session_obj.laps.copy()

            if len(laps) == 0:
                logger.warning(f"No laps found for {year} {gp} {session}")
                return None

            logger.info(f"Loaded {len(laps)} laps from {laps['Driver'].nunique()} drivers")
            return laps

        except Exception as e:
            logger.error(f"Failed to load {year} {gp} {session}: {e}")
            return None

    def standardize_schema(
        self,
        laps: pd.DataFrame,
        year: int,
        event_name: str,
        circuit_name: str,
        session_name: str,
        data_group: str,
        regulation_era: str,
        target_race_context: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Standardize raw FastF1 laps into canonical schema.

        Returns:
            (standardized_df, missing_fields_list)
        """
        df = laps.copy()
        missing_fields = []

        # Map FastF1 columns to canonical names
        column_mapping = {
            "Driver": "driver",
            "Team": "team",
            "LapNumber": "lap_number",
            "LapTime": "lap_time",
            "Stint": "stint",
            "Compound": "compound",
            "TyreLife": "tyre_life",
            "TrackStatus": "track_status",
            "Position": "position",
            "PitInTime": "pit_in_time",
            "PitOutTime": "pit_out_time",
            "IsAccurate": "is_accurate",
            "Deleted": "deleted",
        }

        # Apply mapping
        for src, dst in column_mapping.items():
            if src in df.columns:
                df[dst] = df[src]
            else:
                missing_fields.append(dst)
                # Create empty column with sensible default
                if dst in ["is_accurate", "deleted"]:
                    df[dst] = True  # Assume accurate/not deleted if not available
                else:
                    df[dst] = None

        # Add metadata fields
        df["season"] = year
        df["event_name"] = event_name
        df["circuit_name"] = circuit_name
        df["session_name"] = session_name
        df["data_group"] = data_group
        df["regulation_era"] = regulation_era
        df["target_race_context"] = target_race_context

        # Convert timedeltas to seconds
        timedelta_cols = [col for col in df.columns if df[col].dtype == "timedelta64[ns]"]
        for col in timedelta_cols:
            df[col] = df[col].dt.total_seconds()

        # Select only canonical columns (in order)
        canonical_cols = [c for c in self.CANONICAL_SCHEMA if c in df.columns]
        df = df[canonical_cols].copy()

        return df, missing_fields

    def ingest_miami_historical(self) -> list[pd.DataFrame]:
        """
        Ingest Miami GP races from 2022–2025.

        Returns:
            List of standardized DataFrames (may be empty if all fail).
        """
        miami_years = [2022, 2023, 2024, 2025]
        datasets = []

        for year in miami_years:
            try:
                laps = self.load_session(year, "Miami", "R")
                if laps is None:
                    self.manifest.append(
                        SessionRecord(
                            season=year,
                            event_name="Miami Grand Prix",
                            session_name="Race",
                            data_group="miami_historical",
                            regulation_era="2022_2025",
                            target_race_context="miami",
                            success=False,
                            error_message=f"Failed to load laps for {year} Miami",
                        )
                    )
                    continue

                std_df, missing_fields = self.standardize_schema(
                    laps,
                    year=year,
                    event_name="Miami Grand Prix",
                    circuit_name="Miami International Autodrome",
                    session_name="Race",
                    data_group="miami_historical",
                    regulation_era="2022_2025",
                    target_race_context="miami",
                )

                self.manifest.append(
                    SessionRecord(
                        season=year,
                        event_name="Miami Grand Prix",
                        session_name="Race",
                        data_group="miami_historical",
                        regulation_era="2022_2025",
                        target_race_context="miami",
                        success=True,
                        row_count=len(std_df),
                        missing_fields=missing_fields,
                    )
                )

                datasets.append(std_df)
                logger.info(f"[OK] {year} Miami: {len(std_df)} laps ingested")

            except Exception as e:
                logger.error(f"Error processing {year} Miami: {e}")
                self.manifest.append(
                    SessionRecord(
                        season=year,
                        event_name="Miami Grand Prix",
                        session_name="Race",
                        data_group="miami_historical",
                        regulation_era="2022_2025",
                        target_race_context="miami",
                        success=False,
                        error_message=str(e),
                    )
                )

        return datasets

    def ingest_2026_pre_miami(self) -> list[pd.DataFrame]:
        """
        Ingest 2026 races BEFORE Miami using schedule-driven, completion-aware logic.

        Dynamically identifies the Miami event from the 2026 schedule,
        then selects all race weekends strictly before it AND already completed.
        Only attempts races that have a race date strictly before today (not on/after today).
        All discovered races (including skipped/failed) are recorded in manifest.
        Fails gracefully if 2026 data is unavailable.

        Returns:
            List of standardized DataFrames (may be empty if no races completed,
            data unavailable, or schedule unavailable).
        """
        try:
            # Get 2026 schedule from FastF1
            schedule_2026 = fastf1.get_event_schedule(2026)
            logger.info("[OK] Retrieved 2026 F1 schedule")
        except Exception as e:
            logger.warning(f"Failed to retrieve 2026 schedule: {e}. Skipping 2026 pre-Miami ingestion.")
            self.manifest.append(
                SessionRecord(
                    season=2026,
                    event_name="2026 Pre-Miami Batch",
                    session_name="Batch",
                    data_group="season_2026_pre_miami",
                    regulation_era="2026_active_aero",
                    target_race_context="miami",
                    success=False,
                    error_message=f"Failed to retrieve 2026 schedule: {e}",
                )
            )
            return []

        # Find Miami event in schedule
        miami_mask = schedule_2026['EventName'].str.contains('Miami', case=False, na=False)
        miami_rows = schedule_2026[miami_mask]

        if miami_rows.empty:
            logger.warning("Miami not found in 2026 schedule. Skipping 2026 pre-Miami ingestion.")
            self.manifest.append(
                SessionRecord(
                    season=2026,
                    event_name="2026 Pre-Miami Batch",
                    session_name="Batch",
                    data_group="season_2026_pre_miami",
                    regulation_era="2026_active_aero",
                    target_race_context="miami",
                    success=False,
                    error_message="Miami not found in 2026 schedule",
                )
            )
            return []

        miami_date = miami_rows.iloc[0]['EventDate']
        logger.info(f"[OK] Found Miami in 2026 schedule: {miami_date}")

        # Get current date (as pandas Timestamp for consistent comparison)
        today = pd.Timestamp('today').normalize()
        logger.info(f"Build date: {today.date()}")

        # Get all races before Miami AND strictly before today (exclude testing sessions)
        pre_miami_races_schedule = schedule_2026[
            (schedule_2026['EventDate'] < miami_date) &
            (~schedule_2026['EventName'].str.contains('Testing', case=False, na=False))
        ]

        if pre_miami_races_schedule.empty:
            logger.info("No races found before Miami in 2026 schedule.")
            return []

        logger.info(f"[OK] Found {len(pre_miami_races_schedule)} race(s) in 2026 pre-Miami schedule:")
        for idx, row in pre_miami_races_schedule.iterrows():
            logger.info(f"    - {row['EventName']} ({row['EventDate'].date()})")

        # Separate races into completed (before today) and future (today or later)
        completed_races = pre_miami_races_schedule[
            pre_miami_races_schedule['EventDate'] < today
        ]
        future_races = pre_miami_races_schedule[
            pre_miami_races_schedule['EventDate'] >= today
        ]

        logger.info(f"  {len(completed_races)} completed by build date")
        logger.info(f"  {len(future_races)} not yet completed (will skip)")

        # Process each completed pre-Miami race
        datasets = []
        for idx, race_row in completed_races.iterrows():
            event_name = race_row['EventName']
            event_date = race_row['EventDate'].date()
            circuit_name = race_row.get('Location', 'Unknown Circuit')

            logger.info(f"Attempting to load 2026 {event_name} ({event_date}) race session...")
            try:
                laps = self.load_session(2026, event_name, "R")
                if laps is None:
                    logger.warning(
                        f"2026 {event_name} ({event_date}): No laps found (data unavailable)"
                    )
                    self.manifest.append(
                        SessionRecord(
                            season=2026,
                            event_name=event_name,
                            session_name="Race",
                            data_group="season_2026_pre_miami",
                            regulation_era="2026_active_aero",
                            target_race_context="miami",
                            success=False,
                            error_message="No laps found / data unavailable",
                        )
                    )
                    continue

                std_df, missing_fields = self.standardize_schema(
                    laps,
                    year=2026,
                    event_name=event_name,
                    circuit_name=circuit_name,
                    session_name="Race",
                    data_group="season_2026_pre_miami",
                    regulation_era="2026_active_aero",
                    target_race_context="miami",
                )

                self.manifest.append(
                    SessionRecord(
                        season=2026,
                        event_name=event_name,
                        session_name="Race",
                        data_group="season_2026_pre_miami",
                        regulation_era="2026_active_aero",
                        target_race_context="miami",
                        success=True,
                        row_count=len(std_df),
                        missing_fields=missing_fields,
                    )
                )

                datasets.append(std_df)
                logger.info(f"[OK] 2026 {event_name}: {len(std_df)} laps ingested")

            except Exception as e:
                logger.warning(
                    f"2026 {event_name} ({event_date}): Failed to ingest. Error: {e}"
                )
                self.manifest.append(
                    SessionRecord(
                        season=2026,
                        event_name=event_name,
                        session_name="Race",
                        data_group="season_2026_pre_miami",
                        regulation_era="2026_active_aero",
                        target_race_context="miami",
                        success=False,
                        error_message=str(e),
                    )
                )
                # Don't block on missing 2026 data—continue

        # Log all future races as skipped
        for idx, race_row in future_races.iterrows():
            event_name = race_row['EventName']
            event_date = race_row['EventDate'].date()
            logger.info(
                f"2026 {event_name} ({event_date}): Skipping (race not completed yet as of build date)"
            )
            self.manifest.append(
                SessionRecord(
                    season=2026,
                    event_name=event_name,
                    session_name="Race",
                    data_group="season_2026_pre_miami",
                    regulation_era="2026_active_aero",
                    target_race_context="miami",
                    success=False,
                    error_message="Race not completed yet as of build date",
                )
            )

        if not datasets:
            logger.info("No 2026 pre-Miami data ingested (expected if races haven't occurred yet)")

        return datasets

    def save_datasets(
        self,
        datasets: list[pd.DataFrame],
        group_name: str,
        output_dir: Path | str = "data/raw",
    ) -> list[Path]:
        """
        Save datasets to Parquet format.

        Args:
            datasets: List of DataFrames to save
            group_name: Parent directory name (miami_historical, season_2026_pre_miami)
            output_dir: Root output directory

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir) / group_name
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, df in enumerate(datasets):
            season = int(df["season"].iloc[0])
            event = df["event_name"].iloc[0].replace(" ", "_").lower()
            filename = f"{season}_{event}_race.parquet"
            filepath = output_dir / filename

            df.to_parquet(filepath, index=False, compression="snappy")
            logger.info(f"Saved {len(df)} laps to {filepath}")
            saved_paths.append(filepath)

        # Also save combined dataset if multiple
        if len(datasets) > 1:
            combined = pd.concat(datasets, ignore_index=True)
            combined_path = output_dir / "combined.parquet"
            combined.to_parquet(combined_path, index=False, compression="snappy")
            logger.info(f"Saved combined {len(combined)} laps to {combined_path}")
            saved_paths.append(combined_path)

        return saved_paths

    def save_manifest(self, output_path: Path | str = "data/raw/manifest.json") -> Path:
        """
        Save ingestion manifest to JSON.

        Args:
            output_path: Path to manifest JSON file

        Returns:
            Path to saved manifest.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_sessions_attempted": len(self.manifest),
            "total_sessions_succeeded": sum(1 for r in self.manifest if r.success),
            "total_rows_ingested": sum(r.row_count for r in self.manifest),
            "sessions": [record.to_dict() for record in self.manifest],
        }

        with open(output_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Manifest saved to {output_path}")
        return output_path

    def print_summary(self):
        """Print ingestion summary to console."""
        successful = sum(1 for r in self.manifest if r.success)
        failed = sum(1 for r in self.manifest if not r.success)
        total_rows = sum(r.row_count for r in self.manifest)

        print("\n" + "=" * 70)
        print("PHASE 1A DATA INGESTION SUMMARY")
        print("=" * 70)
        print(f"Sessions attempted:  {len(self.manifest)}")
        print(f"Sessions succeeded:  {successful}")
        print(f"Sessions failed:     {failed}")
        print(f"Total rows ingested: {total_rows:,}")
        print()

        print("Miami Historical (2022–2025):")
        miami_records = [r for r in self.manifest if r.data_group == "miami_historical"]
        for r in miami_records:
            status = "[OK]" if r.success else "[X]"
            print(f"  {status} {r.season} {r.event_name}: "
                  f"{r.row_count} rows" if r.success else f"{r.error_message}")

        print()
        print("2026 Pre-Miami:")
        season_2026_records = [r for r in self.manifest if r.data_group == "season_2026_pre_miami"]
        if season_2026_records:
            for r in season_2026_records:
                status = "[OK]" if r.success else "[X]"
                print(f"  {status} {r.season} {r.event_name}: {r.row_count} rows")
        else:
            print("  (No 2026 data available yet—races may not have occurred)")

        print("\n" + "=" * 70)
