# src/fetchweather.py
from typing import Iterable, Optional, List
import pandas as pd

DEFAULT_STATES_FILE = "data/processed/waob_features_states.csv"

WEATHER_COLS: List[str] = [
    "trend", "jun_shortfall", "temp_JA", "prec_JA", "prec_JA_sq", "dummy_2003"
]

BASE_COLS: List[str] = [
    "state", "year", "yield_bu_acre", "harvest_ha", "acres_harvested"
]

def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["state"] = out["state"].astype(str).str.strip().str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out

def load_weather_features(
    states_file: str = DEFAULT_STATES_FILE,
    states: Optional[Iterable[str]] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load WAOB-style weather + yield features from the consolidated state file.
    Returns a tidy DataFrame with one row per (state, year).
    Columns returned: BASE_COLS + WEATHER_COLS.
    """
    df = pd.read_csv(states_file)
    df = _normalize_keys(df)

    keep = BASE_COLS + WEATHER_COLS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {states_file}: {missing}")
    df = df[keep].drop_duplicates(subset=["state","year"]).copy()

    if states:
        states = [s.strip().upper() for s in states]
        df = df[df["state"].isin(states)]
    if year_from is not None:
        df = df[df["year"].astype("Int64") >= int(year_from)]
    if year_to is not None:
        df = df[df["year"].astype("Int64") <= int(year_to)]

    return df.reset_index(drop=True)
