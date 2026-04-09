"""Data ingestion and preprocessing module.

Key modules:
- loader: Central data loading interface (Parquet-first, CSV fallback)
- preprocess: Data cleaning and feature preparation
- ingest: FastF1 race data ingestion
- ingest_phase1: Phase 1A Miami-focused data ingestion
"""

from src.data.loader import DataLoader, load_data

__all__ = ["DataLoader", "load_data"]
