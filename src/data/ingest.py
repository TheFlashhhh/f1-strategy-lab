"""Load F1 race session data using FastF1."""

from pathlib import Path

import pandas as pd
import fastf1

# Enable FastF1 cache to data/raw/fastf1_cache
cache_dir = Path("data/raw/fastf1_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))


def load_race_session(year: int, gp: str, session: str = "R") -> pd.DataFrame:
    """
    Load a historical F1 race session and extract lap-level data.

    Args:
        year: Season year (e.g., 2022).
        gp: Grand Prix name (e.g., 'Australia', 'Bahrain').
        session: Session code (default 'R'). Options: 'R', 'Q', 'P1', 'P2', 'P3', 'S'.

    Returns:
        DataFrame with full lap-level data for all drivers.
    """
    print(f"Loading {year} {gp} {session}...")
    session_obj = fastf1.get_session(year, gp, session)
    session_obj.load()

    # Get full lap-level dataframe
    laps = session_obj.laps.copy()

    # Convert timedelta columns to seconds
    timedelta_cols = [col for col in laps.columns if laps[col].dtype == "timedelta64[ns]"]
    for col in timedelta_cols:
        laps[col] = laps[col].dt.total_seconds()

    print(f"Loaded {len(laps)} laps from {laps['Driver'].nunique()} drivers")
    return laps


def save_laps(laps: pd.DataFrame, filename: str) -> Path:
    """
    Save lap data to data/raw/.

    Args:
        laps: DataFrame to save.
        filename: Output filename (e.g., '2023_bahrain_race.csv').

    Returns:
        Path to saved file.
    """
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    filepath = raw_dir / filename
    laps.to_csv(filepath, index=False)
    print(f"Saved {len(laps)} laps to {filepath}")
    return filepath


if __name__ == "__main__":
    # Load and save 2020 Abu Dhabi race lap data
    df = load_race_session(2020, "Abu Dhabi", "R")
    save_laps(df, "2020_abudhabi_race.csv")
