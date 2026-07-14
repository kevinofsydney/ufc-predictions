"""Tests for leakage-safe point-in-time feature construction."""

from __future__ import annotations

import sqlite3

import pytest

from ufcpred.db import SCHEMA
from ufcpred.features import build_feature_rows, fight_duration_sec, store_feature_rows
from ufcpred.ratings import ELO_SCHEMA


def _database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.executescript(ELO_SCHEMA)
    conn.executemany(
        "INSERT INTO fighters (fighter_id, name, height_cm, reach_cm, dob) VALUES (?, ?, ?, ?, ?)",
        [
            ("A", "Fighter A", 180.0, 185.0, "1990-01-01"),
            ("B", "Fighter B", 175.0, 180.0, "1992-01-01"),
        ],
    )
    return conn


def _add_fight(
    conn: sqlite3.Connection,
    number: int,
    event_date: str,
    a_sig: tuple[int, int],
    b_sig: tuple[int, int],
    *,
    winner: str = "A",
) -> None:
    event_id = f"e{number}"
    fight_id = f"f{number}"
    conn.execute(
        "INSERT INTO events (event_id, name, event_date) VALUES (?, ?, ?)",
        (event_id, f"Event {number}", event_date),
    )
    conn.execute(
        """
        INSERT INTO fights
            (fight_id, event_id, fighter_a_id, fighter_b_id, winner_id,
             method, end_round, end_time_sec)
        VALUES (?, ?, 'A', 'B', ?, 'U-DEC', 1, 300)
        """,
        (fight_id, event_id, winner),
    )
    conn.execute(
        "INSERT INTO fight_elo VALUES (?, 1500, 1500, ?, ?)",
        (fight_id, number - 1, number - 1),
    )
    for fighter_id, sig, opponent_sig in (("A", a_sig, b_sig), ("B", b_sig, a_sig)):
        conn.execute(
            """
            INSERT INTO fight_stats
                (fight_id, fighter_id, sig_strikes_landed, sig_strikes_attempted,
                 takedowns_landed, takedowns_attempted, sub_attempts, control_time_sec)
            VALUES (?, ?, ?, ?, 0, 0, 0, ?)
            """,
            (fight_id, fighter_id, sig[0], sig[1], 60 if fighter_id == "A" else 30),
        )
    conn.commit()


def test_fight_duration_sec():
    assert fight_duration_sec(1, 188) == 188
    assert fight_duration_sec(3, 120) == 720
    assert fight_duration_sec(None, 120) is None


def test_fight_three_uses_exactly_fights_one_and_two():
    """The Phase 4 leakage guard: fight 3 must not contribute to itself."""
    conn = _database()
    try:
        _add_fight(conn, 1, "2020-01-01", (10, 20), (5, 20))
        _add_fight(conn, 2, "2020-02-01", (20, 40), (10, 20))
        # These extreme fight-3 values would materially change its own row if leaked.
        _add_fight(conn, 3, "2020-03-01", (100, 100), (1, 100), winner="B")

        rows = build_feature_rows(conn, random_swap=False)
        third = next(row for row in rows if row["fight_id"] == "f3")

        # Prior 10 minutes: A landed 30 (3.0/min), B landed 15 (1.5/min).
        assert third["slpm_diff"] == pytest.approx(1.5)
        assert third["sapm_diff"] == pytest.approx(-1.5)
        assert third["str_acc_diff"] == pytest.approx(0.5 - 0.375)
        assert third["label"] == 0
    finally:
        conn.close()


def test_same_date_fights_do_not_see_each_other():
    conn = _database()
    try:
        _add_fight(conn, 1, "2020-01-01", (10, 20), (5, 20))
        _add_fight(conn, 2, "2020-02-01", (20, 40), (10, 20))
        _add_fight(conn, 3, "2020-02-01", (100, 100), (1, 100))

        rows = build_feature_rows(conn, random_swap=False)
        second = next(row for row in rows if row["fight_id"] == "f2")
        third = next(row for row in rows if row["fight_id"] == "f3")
        assert second["slpm_diff"] == pytest.approx(1.0)
        assert third["slpm_diff"] == pytest.approx(1.0)
    finally:
        conn.close()


def test_fixed_swap_is_reproducible_and_symmetric():
    conn = _database()
    try:
        _add_fight(conn, 1, "2020-01-01", (10, 20), (5, 20))
        _add_fight(conn, 2, "2020-02-01", (20, 40), (10, 20))
        unswapped = build_feature_rows(conn, random_swap=False)
        swapped_1 = build_feature_rows(conn, seed=42)
        swapped_2 = build_feature_rows(conn, seed=42)
        assert swapped_1 == swapped_2
        for original, transformed in zip(unswapped, swapped_1):
            if transformed["label"] != original["label"]:
                assert transformed["height_diff"] == -original["height_diff"]
    finally:
        conn.close()


def test_store_is_non_destructive_upsert():
    conn = _database()
    try:
        _add_fight(conn, 1, "2020-01-01", (10, 20), (5, 20))
        rows = build_feature_rows(conn, random_swap=False)
        assert store_feature_rows(conn, rows) == 1
        assert store_feature_rows(conn, rows) == 1
        assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 1
    finally:
        conn.close()
