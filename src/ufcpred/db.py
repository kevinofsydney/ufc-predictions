"""SQLite database helpers for the ufcpred project.

DB_PATH is anchored to the repository root so it resolves correctly
regardless of the current working directory the code is invoked from.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# src/ufcpred/db.py -> parents[0]=src/ufcpred, parents[1]=src, parents[2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "ufc.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,      -- hex id from URL
    name        TEXT NOT NULL,
    event_date  TEXT NOT NULL,         -- ISO 8601 'YYYY-MM-DD'
    location    TEXT
);
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id  TEXT PRIMARY KEY,      -- hex id from URL
    name        TEXT NOT NULL,
    height_cm   REAL,                  -- NULL if unknown
    reach_cm    REAL,
    stance      TEXT,                  -- 'Orthodox','Southpaw','Switch', or NULL
    dob         TEXT                   -- ISO 8601 or NULL
);
CREATE TABLE IF NOT EXISTS fights (
    fight_id     TEXT PRIMARY KEY,     -- hex id from URL
    event_id     TEXT NOT NULL REFERENCES events(event_id),
    fighter_a_id TEXT NOT NULL REFERENCES fighters(fighter_id),
    fighter_b_id TEXT NOT NULL REFERENCES fighters(fighter_id),
    winner_id    TEXT,                 -- NULL for draw/no-contest
    weight_class TEXT,
    method       TEXT,                 -- e.g. 'KO/TKO','SUB','U-DEC','S-DEC','M-DEC','DQ','NC'
    end_round    INTEGER,
    end_time_sec INTEGER,              -- seconds into end_round
    is_title     INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS fight_stats (
    fight_id     TEXT NOT NULL REFERENCES fights(fight_id),
    fighter_id   TEXT NOT NULL REFERENCES fighters(fighter_id),
    knockdowns   INTEGER,
    sig_strikes_landed    INTEGER,
    sig_strikes_attempted INTEGER,
    total_strikes_landed  INTEGER,
    total_strikes_attempted INTEGER,
    takedowns_landed      INTEGER,
    takedowns_attempted   INTEGER,
    sub_attempts INTEGER,
    reversals    INTEGER,
    control_time_sec INTEGER,
    PRIMARY KEY (fight_id, fighter_id)
);
"""


def get_conn() -> sqlite3.Connection:
    """Return a sqlite3 connection to the project database with FK enforcement on."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create the project tables if they do not already exist."""
    conn = get_conn()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
