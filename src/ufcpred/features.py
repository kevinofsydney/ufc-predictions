"""Leakage-safe, point-in-time fight features.

Run as:  python -m ufcpred.features

For every dated fight, career aggregates are calculated from fights on strictly
earlier dates. Fights on the same date are evaluated as a batch before any of
that date's results are added to fighter state. Winnerless fights still update
career history, but are excluded from the materialised training rows.

The training representation is symmetric: with a fixed random seed, half of
the rows swap fighter A and B, negate every differential, and flip the label.
This prevents the model from learning the source dataset's red-corner ordering.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from itertools import groupby
from typing import Iterable

from .db import get_conn

SWAP_SEED = 42

DIFF_COLUMNS = [
    "elo_diff",
    "age_diff",
    "reach_diff",
    "height_diff",
    "slpm_diff",
    "sapm_diff",
    "str_acc_diff",
    "str_def_diff",
    "td_avg_diff",
    "td_acc_diff",
    "sub_avg_diff",
    "ctrl_pct_diff",
    "finish_rate_diff",
    "win_streak_diff",
    "n_fights_diff",
    "layoff_diff",
]

FEATURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    fight_id         TEXT PRIMARY KEY REFERENCES fights(fight_id),
    event_date       TEXT NOT NULL,
    elo_diff         REAL,
    age_diff         REAL,
    reach_diff       REAL,
    height_diff      REAL,
    slpm_diff        REAL,
    sapm_diff        REAL,
    str_acc_diff     REAL,
    str_def_diff     REAL,
    td_avg_diff      REAL,
    td_acc_diff      REAL,
    sub_avg_diff     REAL,
    ctrl_pct_diff    REAL,
    finish_rate_diff REAL,
    win_streak_diff  INTEGER,
    n_fights_diff    INTEGER,
    layoff_diff      INTEGER,
    label             INTEGER NOT NULL CHECK (label IN (0, 1))
);
"""


@dataclass
class CareerState:
    """Additive history for one fighter, containing completed prior fights."""

    n_fights: int = 0
    wins: int = 0
    finish_wins: int = 0
    win_streak: int = 0
    last_fight_date: date | None = None

    sig_landed: float = 0.0
    sig_landed_time_sec: float = 0.0
    sig_accuracy_landed: float = 0.0
    sig_absorbed: float = 0.0
    sig_absorbed_time_sec: float = 0.0
    sig_attempted: float = 0.0
    opp_sig_landed: float = 0.0
    opp_sig_attempted: float = 0.0

    td_landed: float = 0.0
    td_landed_time_sec: float = 0.0
    td_accuracy_landed: float = 0.0
    td_attempted: float = 0.0
    sub_attempts: float = 0.0
    sub_time_sec: float = 0.0
    control_time_sec: float = 0.0
    control_fight_time_sec: float = 0.0


def _ratio(numerator: float, denominator: float, scale: float = 1.0) -> float | None:
    if denominator <= 0:
        return None
    return scale * numerator / denominator


def _parse_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _difference(a: float | int | None, b: float | int | None) -> float | int | None:
    if a is None or b is None:
        return None
    return a - b


def fight_duration_sec(end_round: int | None, end_time_sec: int | None) -> int | None:
    """Return elapsed fight seconds using the pinned five-minute-round rule."""
    if end_round is None or end_time_sec is None or end_round < 1 or end_time_sec < 0:
        return None
    return (end_round - 1) * 5 * 60 + end_time_sec


def snapshot(state: CareerState, as_of: date, fighter: dict) -> dict:
    """Return one fighter's career metrics before *as_of*."""
    dob = _parse_iso(fighter.get("dob"))
    age = (as_of - dob).days / 365.2425 if dob is not None and dob <= as_of else None
    layoff = (
        (as_of - state.last_fight_date).days
        if state.last_fight_date is not None
        else None
    )
    return {
        "age": age,
        "reach": fighter.get("reach_cm"),
        "height": fighter.get("height_cm"),
        "slpm": _ratio(state.sig_landed, state.sig_landed_time_sec, 60.0),
        "sapm": _ratio(state.sig_absorbed, state.sig_absorbed_time_sec, 60.0),
        "str_acc": _ratio(state.sig_accuracy_landed, state.sig_attempted),
        "str_def": (
            None
            if state.opp_sig_attempted <= 0
            else 1.0 - state.opp_sig_landed / state.opp_sig_attempted
        ),
        "td_avg": _ratio(state.td_landed, state.td_landed_time_sec, 15.0 * 60.0),
        "td_acc": _ratio(state.td_accuracy_landed, state.td_attempted),
        "sub_avg": _ratio(state.sub_attempts, state.sub_time_sec, 15.0 * 60.0),
        "ctrl_pct": _ratio(state.control_time_sec, state.control_fight_time_sec),
        "finish_rate": _ratio(state.finish_wins, state.wins),
        "win_streak": state.win_streak,
        "n_fights": state.n_fights,
        "layoff": layoff,
    }


def _add_if_observed(
    state: CareerState,
    numerator_attr: str,
    denominator_attr: str,
    value: int | float | None,
    denominator: int | float | None,
) -> None:
    if value is None or denominator is None or denominator <= 0:
        return
    setattr(state, numerator_attr, getattr(state, numerator_attr) + value)
    setattr(state, denominator_attr, getattr(state, denominator_attr) + denominator)


def update_state(
    state: CareerState,
    own_stats: dict,
    opponent_stats: dict,
    fight: dict,
    fighter_id: str,
) -> None:
    """Add one completed fight to a fighter's career state."""
    duration = fight_duration_sec(fight.get("end_round"), fight.get("end_time_sec"))

    _add_if_observed(
        state,
        "sig_landed",
        "sig_landed_time_sec",
        own_stats.get("sig_strikes_landed"),
        duration,
    )
    _add_if_observed(
        state,
        "sig_absorbed",
        "sig_absorbed_time_sec",
        opponent_stats.get("sig_strikes_landed"),
        duration,
    )
    if own_stats.get("sig_strikes_landed") is not None and own_stats.get("sig_strikes_attempted") is not None:
        state.sig_accuracy_landed += own_stats["sig_strikes_landed"]
        state.sig_attempted += own_stats["sig_strikes_attempted"]
    if opponent_stats.get("sig_strikes_landed") is not None and opponent_stats.get("sig_strikes_attempted") is not None:
        state.opp_sig_landed += opponent_stats["sig_strikes_landed"]
        state.opp_sig_attempted += opponent_stats["sig_strikes_attempted"]

    _add_if_observed(
        state,
        "td_landed",
        "td_landed_time_sec",
        own_stats.get("takedowns_landed"),
        duration,
    )
    if own_stats.get("takedowns_landed") is not None and own_stats.get("takedowns_attempted") is not None:
        state.td_accuracy_landed += own_stats["takedowns_landed"]
        state.td_attempted += own_stats["takedowns_attempted"]
    _add_if_observed(
        state,
        "sub_attempts",
        "sub_time_sec",
        own_stats.get("sub_attempts"),
        duration,
    )
    _add_if_observed(
        state,
        "control_time_sec",
        "control_fight_time_sec",
        own_stats.get("control_time_sec"),
        duration,
    )

    won = fight.get("winner_id") == fighter_id
    if won:
        state.wins += 1
        state.win_streak += 1
        if fight.get("method") in {"KO/TKO", "SUB"}:
            state.finish_wins += 1
    else:
        state.win_streak = 0
    state.n_fights += 1
    state.last_fight_date = _parse_iso(fight.get("event_date"))


def _row_dicts(cursor: sqlite3.Cursor) -> list[dict]:
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def load_fights(
    conn: sqlite3.Connection, before_date: str | None = None
) -> list[dict]:
    """Load all fights, stats, Elo, and safe static fighter attributes."""
    query = """
        SELECT
            f.fight_id, e.event_date, f.fighter_a_id, f.fighter_b_id,
            f.winner_id, f.method, f.end_round, f.end_time_sec,
            fe.elo_a_pre, fe.elo_b_pre,
            a.height_cm AS a_height_cm, a.reach_cm AS a_reach_cm, a.dob AS a_dob,
            b.height_cm AS b_height_cm, b.reach_cm AS b_reach_cm, b.dob AS b_dob,
            sa.sig_strikes_landed AS a_sig_strikes_landed,
            sa.sig_strikes_attempted AS a_sig_strikes_attempted,
            sa.takedowns_landed AS a_takedowns_landed,
            sa.takedowns_attempted AS a_takedowns_attempted,
            sa.sub_attempts AS a_sub_attempts,
            sa.control_time_sec AS a_control_time_sec,
            sb.sig_strikes_landed AS b_sig_strikes_landed,
            sb.sig_strikes_attempted AS b_sig_strikes_attempted,
            sb.takedowns_landed AS b_takedowns_landed,
            sb.takedowns_attempted AS b_takedowns_attempted,
            sb.sub_attempts AS b_sub_attempts,
            sb.control_time_sec AS b_control_time_sec
        FROM fights f
        JOIN events e ON e.event_id = f.event_id
        JOIN fight_elo fe ON fe.fight_id = f.fight_id
        JOIN fighters a ON a.fighter_id = f.fighter_a_id
        JOIN fighters b ON b.fighter_id = f.fighter_b_id
        LEFT JOIN fight_stats sa
            ON sa.fight_id = f.fight_id AND sa.fighter_id = f.fighter_a_id
        LEFT JOIN fight_stats sb
            ON sb.fight_id = f.fight_id AND sb.fighter_id = f.fighter_b_id
    """
    params: tuple[str, ...] = ()
    if before_date is not None:
        query += " WHERE e.event_date < ?"
        params = (before_date,)
    query += " ORDER BY e.event_date ASC, f.fight_id ASC"
    cursor = conn.execute(query, params)
    return _row_dicts(cursor)


def _fighter_record(fight: dict, corner: str) -> dict:
    return {
        "height_cm": fight.get(f"{corner}_height_cm"),
        "reach_cm": fight.get(f"{corner}_reach_cm"),
        "dob": fight.get(f"{corner}_dob"),
    }


def _stats_record(fight: dict, corner: str) -> dict:
    return {
        "sig_strikes_landed": fight.get(f"{corner}_sig_strikes_landed"),
        "sig_strikes_attempted": fight.get(f"{corner}_sig_strikes_attempted"),
        "takedowns_landed": fight.get(f"{corner}_takedowns_landed"),
        "takedowns_attempted": fight.get(f"{corner}_takedowns_attempted"),
        "sub_attempts": fight.get(f"{corner}_sub_attempts"),
        "control_time_sec": fight.get(f"{corner}_control_time_sec"),
    }


def _feature_row(fight: dict, state_a: CareerState, state_b: CareerState) -> dict | None:
    if fight.get("winner_id") not in {fight["fighter_a_id"], fight["fighter_b_id"]}:
        return None
    event_day = _parse_iso(fight.get("event_date"))
    if event_day is None:
        raise ValueError(f"invalid event date for fight {fight['fight_id']}: {fight.get('event_date')!r}")
    a = snapshot(state_a, event_day, _fighter_record(fight, "a"))
    b = snapshot(state_b, event_day, _fighter_record(fight, "b"))
    return {
        "fight_id": fight["fight_id"],
        "event_date": fight["event_date"],
        "elo_diff": _difference(fight.get("elo_a_pre"), fight.get("elo_b_pre")),
        "age_diff": _difference(a["age"], b["age"]),
        "reach_diff": _difference(a["reach"], b["reach"]),
        "height_diff": _difference(a["height"], b["height"]),
        "slpm_diff": _difference(a["slpm"], b["slpm"]),
        "sapm_diff": _difference(a["sapm"], b["sapm"]),
        "str_acc_diff": _difference(a["str_acc"], b["str_acc"]),
        "str_def_diff": _difference(a["str_def"], b["str_def"]),
        "td_avg_diff": _difference(a["td_avg"], b["td_avg"]),
        "td_acc_diff": _difference(a["td_acc"], b["td_acc"]),
        "sub_avg_diff": _difference(a["sub_avg"], b["sub_avg"]),
        "ctrl_pct_diff": _difference(a["ctrl_pct"], b["ctrl_pct"]),
        "finish_rate_diff": _difference(a["finish_rate"], b["finish_rate"]),
        "win_streak_diff": _difference(a["win_streak"], b["win_streak"]),
        "n_fights_diff": _difference(a["n_fights"], b["n_fights"]),
        "layoff_diff": _difference(a["layoff"], b["layoff"]),
        "label": 1 if fight["winner_id"] == fight["fighter_a_id"] else 0,
    }


def build_feature_rows(
    conn: sqlite3.Connection,
    *,
    random_swap: bool = True,
    seed: int = SWAP_SEED,
) -> list[dict]:
    """Build one leakage-safe feature row per fight with a known winner."""
    states: dict[str, CareerState] = {}
    rows: list[dict] = []
    rng = random.Random(seed)
    fights = load_fights(conn)

    for _, same_day_iter in groupby(fights, key=lambda fight: fight["event_date"]):
        same_day = list(same_day_iter)

        # Snapshot every fight first: no result from this date can enter a row
        # for another fight on the same date.
        for fight in same_day:
            state_a = states.setdefault(fight["fighter_a_id"], CareerState())
            state_b = states.setdefault(fight["fighter_b_id"], CareerState())
            row = _feature_row(fight, state_a, state_b)
            if row is None:
                continue
            if random_swap and rng.random() < 0.5:
                for column in DIFF_COLUMNS:
                    if row[column] is not None:
                        row[column] = -row[column]
                row["label"] = 1 - row["label"]
            rows.append(row)

        # Only after every snapshot for the date has been recorded do these
        # fights become prior history for future dates.
        for fight in same_day:
            state_a = states[fight["fighter_a_id"]]
            state_b = states[fight["fighter_b_id"]]
            stats_a = _stats_record(fight, "a")
            stats_b = _stats_record(fight, "b")
            update_state(state_a, stats_a, stats_b, fight, fight["fighter_a_id"])
            update_state(state_b, stats_b, stats_a, fight, fight["fighter_b_id"])

    return rows


def build_career_states_as_of(
    conn: sqlite3.Connection, as_of_date: str
) -> dict[str, CareerState]:
    """Build fighter career state using fights strictly before *as_of_date*."""
    as_of = _parse_iso(as_of_date)
    if as_of is None or as_of.isoformat() != as_of_date:
        raise ValueError(f"invalid as-of date: {as_of_date!r}; expected YYYY-MM-DD")
    states: dict[str, CareerState] = {}
    for fight in load_fights(conn, before_date=as_of_date):
        state_a = states.setdefault(fight["fighter_a_id"], CareerState())
        state_b = states.setdefault(fight["fighter_b_id"], CareerState())
        stats_a = _stats_record(fight, "a")
        stats_b = _stats_record(fight, "b")
        update_state(state_a, stats_a, stats_b, fight, fight["fighter_a_id"])
        update_state(state_b, stats_b, stats_a, fight, fight["fighter_b_id"])
    return states


def build_matchup_features(
    conn: sqlite3.Connection,
    fighter_a_id: str,
    fighter_b_id: str,
    as_of_date: str,
) -> tuple[dict[str, float | int | None], dict[str, float | int]]:
    """Build an unswapped model vector and diagnostics for an upcoming matchup."""
    if fighter_a_id == fighter_b_id:
        raise ValueError("a fighter cannot be matched against themselves")
    as_of = _parse_iso(as_of_date)
    if as_of is None or as_of.isoformat() != as_of_date:
        raise ValueError(f"invalid as-of date: {as_of_date!r}; expected YYYY-MM-DD")

    fighters = {}
    for fighter_id in (fighter_a_id, fighter_b_id):
        row = conn.execute(
            "SELECT fighter_id, height_cm, reach_cm, dob FROM fighters WHERE fighter_id=?",
            (fighter_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown fighter id: {fighter_id}")
        fighters[fighter_id] = {
            "height_cm": row[1],
            "reach_cm": row[2],
            "dob": row[3],
        }

    states = build_career_states_as_of(conn, as_of_date)
    state_a = states.get(fighter_a_id, CareerState())
    state_b = states.get(fighter_b_id, CareerState())
    a = snapshot(state_a, as_of, fighters[fighter_a_id])
    b = snapshot(state_b, as_of, fighters[fighter_b_id])

    from .ratings import START_RATING, rating_state_as_of

    ratings, _ = rating_state_as_of(conn, as_of_date)
    elo_a = ratings.get(fighter_a_id, START_RATING)
    elo_b = ratings.get(fighter_b_id, START_RATING)
    feature_values = {
        "elo_diff": _difference(elo_a, elo_b),
        "age_diff": _difference(a["age"], b["age"]),
        "reach_diff": _difference(a["reach"], b["reach"]),
        "height_diff": _difference(a["height"], b["height"]),
        "slpm_diff": _difference(a["slpm"], b["slpm"]),
        "sapm_diff": _difference(a["sapm"], b["sapm"]),
        "str_acc_diff": _difference(a["str_acc"], b["str_acc"]),
        "str_def_diff": _difference(a["str_def"], b["str_def"]),
        "td_avg_diff": _difference(a["td_avg"], b["td_avg"]),
        "td_acc_diff": _difference(a["td_acc"], b["td_acc"]),
        "sub_avg_diff": _difference(a["sub_avg"], b["sub_avg"]),
        "ctrl_pct_diff": _difference(a["ctrl_pct"], b["ctrl_pct"]),
        "finish_rate_diff": _difference(a["finish_rate"], b["finish_rate"]),
        "win_streak_diff": _difference(a["win_streak"], b["win_streak"]),
        "n_fights_diff": _difference(a["n_fights"], b["n_fights"]),
        "layoff_diff": _difference(a["layoff"], b["layoff"]),
    }
    diagnostics = {
        "elo_a": elo_a,
        "elo_b": elo_b,
        "n_fights_a": state_a.n_fights,
        "n_fights_b": state_b.n_fights,
    }
    return feature_values, diagnostics


def store_feature_rows(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Create/update feature rows without deleting existing database content."""
    rows = list(rows)
    conn.executescript(FEATURE_SCHEMA)
    columns = ["fight_id", "event_date", *DIFF_COLUMNS, "label"]
    names = ", ".join(columns)
    values = ", ".join(f":{column}" for column in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}" for column in columns if column != "fight_id"
    )
    conn.executemany(
        f"INSERT INTO features ({names}) VALUES ({values}) "
        f"ON CONFLICT(fight_id) DO UPDATE SET {updates}",
        rows,
    )
    conn.commit()
    return len(rows)


def null_rates(rows: list[dict]) -> dict[str, float]:
    """Percentage NULL for each model feature, for the Phase 4 audit."""
    if not rows:
        return {column: 0.0 for column in DIFF_COLUMNS}
    return {
        column: 100.0 * sum(row[column] is None for row in rows) / len(rows)
        for column in DIFF_COLUMNS
    }


def run() -> dict:
    """Build and non-destructively materialise all training features."""
    conn = get_conn()
    try:
        rows = build_feature_rows(conn)
        stored = store_feature_rows(conn, rows)
        winner_fights = conn.execute(
            "SELECT COUNT(*) FROM fights WHERE winner_id IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "rows": stored,
        "winner_fights": winner_fights,
        "label_mean": sum(row["label"] for row in rows) / len(rows) if rows else None,
        "null_rates": null_rates(rows),
    }


def main() -> None:
    result = run()
    print("=== ufcpred.features complete ===")
    print(f"feature rows : {result['rows']} (winner fights: {result['winner_fights']})")
    print(f"label mean   : {result['label_mean']:.4f}")
    print("NULL rates:")
    for column, rate in result["null_rates"].items():
        print(f"  {column:18s} {rate:6.2f}%")


if __name__ == "__main__":
    main()
