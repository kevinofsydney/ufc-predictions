"""Tests for non-destructive incremental update behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ufcpred.db import SCHEMA
from ufcpred.update import ingest_new_fights, spider_command


def _refresh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "red_fighter_name": "New Fighter One",
                "blue_fighter_name": "New Fighter Two",
                "event_date": "July 11, 2026",
                "event_name": "UFC Test Event",
                "event_location": "Sydney, Australia",
                "fight_outcome": "red_win",
                "method": "Decision - Unanimous",
                "round": "3",
                "time": "5:00",
                "bout_type": "Lightweight Bout",
                "red_fighter_KD": "0",
                "blue_fighter_KD": "0",
                "red_fighter_sig_str": "30 of 60",
                "blue_fighter_sig_str": "20 of 55",
                "red_fighter_total_str": "40 of 70",
                "blue_fighter_total_str": "25 of 60",
                "red_fighter_TD": "1 of 2",
                "blue_fighter_TD": "0 of 1",
                "red_fighter_sub_att": "0",
                "blue_fighter_sub_att": "0",
                "red_fighter_rev": "0",
                "blue_fighter_rev": "0",
                "red_fighter_ctrl": "2:00",
                "blue_fighter_ctrl": "0:30",
            }
        ]
    )


def test_incremental_ingest_second_run_adds_nothing():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    try:
        first = ingest_new_fights(conn, _refresh_frame())
        counts_after_first = tuple(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("fighters", "events", "fights", "fight_stats")
        )
        second = ingest_new_fights(conn, _refresh_frame())
        counts_after_second = tuple(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("fighters", "events", "fights", "fight_stats")
        )
        assert first["new_fights"] == 1
        assert first["new_fighters"] == 2
        assert second["new_fights"] == 0
        assert second["new_fighters"] == 0
        assert counts_after_first == (2, 1, 1, 2)
        assert counts_after_second == counts_after_first
    finally:
        conn.close()


def test_spider_command_is_incremental_and_polite():
    command = spider_command("2026-06-27", Path("refresh.csv"))
    rendered = " ".join(command)
    assert "since=27/06/2026" in rendered
    assert "CONCURRENT_REQUESTS=1" in rendered
    assert "CONCURRENT_REQUESTS_PER_DOMAIN=1" in rendered
    assert "DOWNLOAD_DELAY=2" in rendered
    assert "RANDOMIZE_DOWNLOAD_DELAY=False" in rendered
    assert "USER_AGENT=ufcpred/0.1" in rendered
