"""Raw-data ingestion for the ufcpred project.

Loads the two CSVs vendored from https://github.com/komaksym/UFC-DataLab
(cloned into vendor/UFC-DataLab/, see data/ingest_meta.json for the pinned
commit) and returns them as pandas DataFrames with light, non-destructive
normalisation applied:

- stats_raw.csv (";"-separated): one row per fight, red/blue fighter stats.
- raw_fighter_details.csv (","-separated): one row per fighter with
  Height/Weight/Reach/Stance/DOB plus UFC.com's own career-average stats
  (SLpM, Str_Acc, SApM, Str_Def, TD_Avg, TD_Acc, TD_Def, Sub_Avg).

IMPORTANT (downstream note): only Height, Weight, Reach, Stance, DOB (and
the name) from raw_fighter_details.csv are safe to use for modelling. The
career-average columns are snapshots as of scrape time (leak future data
into past fights) and must NOT be used as model features, even though
load_raw() returns them in fighters_df for convenience/inspection.

Do NOT load stats_processed.csv / stats_processed_all_bouts.csv anywhere in
this project -- both contain career-snapshot columns that leak information
from after the fight into pre-fight features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# src/ufcpred/ingest.py -> parents[0]=src/ufcpred, parents[1]=src, parents[2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "vendor" / "UFC-DataLab"

STATS_RAW_PATH = VENDOR_DIR / "data" / "stats" / "stats_raw.csv"
FIGHTER_DETAILS_PATH = VENDOR_DIR / "data" / "external_data" / "raw_fighter_details.csv"


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every string/object column in place."""
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the vendored UFC-DataLab CSVs into (fights_df, fighters_df).

    fights_df: one row per fight (stats_raw.csv), ";"-separated, with
        red_name_upper / blue_name_upper helper columns added.
    fighters_df: one row per fighter (raw_fighter_details.csv), ","-separated,
        with a name_upper helper column added.

    Original name columns are preserved untouched; the *_upper columns are
    purely additive, intended for joining fights to fighters.
    """
    if not STATS_RAW_PATH.exists():
        raise FileNotFoundError(
            f"stats_raw.csv not found at {STATS_RAW_PATH}. "
            "Did you clone vendor/UFC-DataLab? "
            "(git clone --depth 1 https://github.com/komaksym/UFC-DataLab.git vendor/UFC-DataLab)"
        )
    if not FIGHTER_DETAILS_PATH.exists():
        raise FileNotFoundError(
            f"raw_fighter_details.csv not found at {FIGHTER_DETAILS_PATH}. "
            "Did you clone vendor/UFC-DataLab? "
            "(git clone --depth 1 https://github.com/komaksym/UFC-DataLab.git vendor/UFC-DataLab)"
        )

    fights_df = pd.read_csv(STATS_RAW_PATH, sep=";", low_memory=False)
    fighters_df = pd.read_csv(FIGHTER_DETAILS_PATH, sep=",", low_memory=False)

    fights_df = _strip_string_columns(fights_df)
    fighters_df = _strip_string_columns(fighters_df)

    fights_df["red_name_upper"] = fights_df["red_fighter_name"].str.upper()
    fights_df["blue_name_upper"] = fights_df["blue_fighter_name"].str.upper()
    fighters_df["name_upper"] = fighters_df["fighter_name"].str.upper()

    return fights_df, fighters_df
