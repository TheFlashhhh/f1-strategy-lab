"""Central data loading interface for Phase 1A Parquet-first architecture.

This module provides a unified entry point for loading F1 race data, with
Parquet-first preference and CSV fallback for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Define the canonical Phase 1A schema to expected preprocess column mapping
COLUMN_MAPPING = {
    # Phase 1A canonical → Preprocess expected
    "driver": "Driver",
    "lap_number": "LapNumber",
    "lap_time": "LapTime",
    "compound": "Compound",
    "tyre_life": "TyreLife",
    "pit_out_time": "PitOutTime",
    "pit_in_time": "PitInTime",
    "stint": "Stint",
    "is_accurate": "IsAccurate",
    "deleted": "Deleted",
    "track_status": "TrackStatus",
}


class DataLoader:
    """Central data loader with Parquet-first preference and CSV fallback."""

    def __init__(self, project_root: Path | str = "."):
        """Initialize loader with project root path.

        Args:
            project_root: Root directory of the F1 Strategy Lab project.
        """
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "raw"

    def load_data(
        self,
        dataset: str = "miami_historical",
        fallback: bool = True,
    ) -> pd.DataFrame:
        """Load race data, preferring Parquet over CSV.

        Args:
            dataset: Name of dataset to load. Options:
                - "miami_historical" (default): Miami combined Parquet
                - "season_2026_pre_miami": 2026 pre-Miami races
                - Any CSV filename in data/raw/ (e.g., "2020_abudhabi_race.csv")
            fallback: If True, fall back to CSV if Parquet not found.

        Returns:
            DataFrame with standardized column names.

        Raises:
            FileNotFoundError: If neither Parquet nor CSV found (and fallback=True).
            ValueError: If dataset specification is invalid.
        """
        # Try Parquet first
        if dataset == "miami_historical":
            return self._load_miami_historical()
        elif dataset == "season_2026_pre_miami":
            return self._load_2026_pre_miami()
        else:
            # Try as CSV filename
            return self._load_csv_fallback(dataset, fallback)

    def _load_miami_historical(self) -> pd.DataFrame:
        """Load Miami historical combined Parquet (2022–2025)."""
        parquet_path = self.data_dir / "miami_historical" / "combined.parquet"

        if parquet_path.exists():
            logger.info(f"Loading Miami historical from Parquet: {parquet_path}")
            try:
                df = pd.read_parquet(parquet_path)
                logger.info(f"✓ Loaded {len(df)} laps from Miami historical")
                return self._normalize_schema(df)
            except Exception as e:
                logger.error(f"Failed to load Parquet: {e}")
                raise

        # Fallback to individual year files
        logger.info("Miami combined.parquet not found, attempting individual year files...")
        dfs = []
        for year in [2022, 2023, 2024, 2025]:
            year_path = self.data_dir / "miami_historical" / f"{year}_miami_grand_prix_race.parquet"
            if year_path.exists():
                try:
                    df = pd.read_parquet(year_path)
                    dfs.append(df)
                    logger.info(f"  Loaded {len(df)} laps from {year}")
                except Exception as e:
                    logger.warning(f"Failed to load {year_path}: {e}")

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"✓ Loaded {len(df)} laps from Miami individual files")
            return self._normalize_schema(df)

        # CSV fallback
        csv_path = self.data_dir / "2020_abudhabi_race.csv"
        if csv_path.exists():
            warnings.warn(
                "No Miami Parquet data found. Falling back to Abu Dhabi 2020 CSV. "
                "Run 'python src/data/build_phase1_dataset.py' to build Phase 1A dataset."
            )
            return self._load_csv_fallback("2020_abudhabi_race.csv", fallback=False)

        raise FileNotFoundError(
            f"No Miami data found. Expected: {parquet_path}\n"
            "Run 'python src/data/build_phase1_dataset.py' to build Phase 1A dataset."
        )

    def _load_2026_pre_miami(self) -> pd.DataFrame:
        """Load 2026 pre-Miami races (if available)."""
        pre_miami_dir = self.data_dir / "season_2026_pre_miami"

        if not pre_miami_dir.exists():
            logger.warning("2026 pre-Miami directory not found.")
            return pd.DataFrame()

        parquet_files = list(pre_miami_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No Parquet files in season_2026_pre_miami directory.")
            return pd.DataFrame()

        dfs = []
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} laps from {pf.stem}")
            except Exception as e:
                logger.warning(f"Failed to load {pf}: {e}")

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"✓ Loaded {len(df)} laps from 2026 pre-Miami")
            return self._normalize_schema(df)

        logger.warning("No 2026 pre-Miami data could be loaded.")
        return pd.DataFrame()

    def _load_csv_fallback(
        self,
        csv_filename: str,
        fallback: bool = True,
    ) -> pd.DataFrame:
        """Load a CSV file as fallback.

        Args:
            csv_filename: Filename in data/raw/ (e.g., "2020_abudhabi_race.csv").
            fallback: If False, raise error if file doesn't exist.

        Returns:
            DataFrame with standardized column names.
        """
        csv_path = self.data_dir / csv_filename

        if not csv_path.exists():
            if not fallback:
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            logger.warning(f"CSV {csv_filename} not found.")
            return pd.DataFrame()

        logger.info(f"Loading CSV: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✓ Loaded {len(df)} laps from CSV")
            return self._normalize_schema(df)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to preprocess-expected column names.

        Handles both Phase 1A Parquet schema (lowercase) and legacy CSV schema.

        Args:
            df: Raw DataFrame with possibly mixed column naming.

        Returns:
            DataFrame with standardized column names expected by preprocess.py.
        """
        df = df.copy()

        # Check what schema we have
        has_phase1a_columns = any(col.lower() in df.columns.str.lower() for col in COLUMN_MAPPING.keys())
        has_legacy_columns = any(col in df.columns for col in COLUMN_MAPPING.values())

        if not has_phase1a_columns and not has_legacy_columns:
            logger.warning(
                f"DataFrame columns: {list(df.columns)}. "
                f"Expected Phase 1A: {list(COLUMN_MAPPING.keys())} or "
                f"Legacy: {list(COLUMN_MAPPING.values())}"
            )

        # Rename Phase 1A columns to preprocess names (case-insensitive)
        for phase1a_col, preprocess_col in COLUMN_MAPPING.items():
            for df_col in df.columns:
                if df_col.lower() == phase1a_col.lower() and df_col != preprocess_col:
                    df.rename(columns={df_col: preprocess_col}, inplace=True)
                    logger.debug(f"Mapped column {df_col} → {preprocess_col}")
                    break

        # Validate required columns
        required_cols = [
            "Driver", "LapNumber", "LapTime", "Compound", "TyreLife",
            "Stint", "IsAccurate", "Deleted", "TrackStatus",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        # Type conversions
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").fillna(0).astype(int)
        df["LapTime"] = pd.to_numeric(df["LapTime"], errors="coerce")
        df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0).astype(int)
        df["Stint"] = pd.to_numeric(df["Stint"], errors="coerce").fillna(0).astype(int)
        df["TrackStatus"] = df["TrackStatus"].astype(str)
        df["IsAccurate"] = df["IsAccurate"].astype(bool)
        df["Deleted"] = df["Deleted"].astype(bool)

        return df

    def get_metadata(self) -> dict:
        """Get ingestion metadata from manifest.json if available.

        Returns:
            Metadata dict or empty dict if manifest not available.
        """
        manifest_path = self.data_dir / "manifest.json"

        if not manifest_path.exists():
            return {}

        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read manifest: {e}")
            return {}


def load_data(
    dataset: str = "miami_historical",
    project_root: Path | str = ".",
) -> pd.DataFrame:
    """
    Convenience function to load data using default loader.

    Args:
        dataset: Name of dataset to load (see DataLoader.load_data).
        project_root: Root directory of project.

    Returns:
        DataFrame with standardized columns.
    """
    loader = DataLoader(project_root)
    return loader.load_data(dataset)
