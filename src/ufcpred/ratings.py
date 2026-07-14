"""Elo rating engine.

Run as:  python -m ufcpred.ratings

Computes a pre-fight Elo rating for both fighters of every fight, processing
fights in deterministic chronological order (event_date ASC, fight_id ASC),
and stores the results in the ``fight_elo`` table (full recompute each run).

Only PRE-fight ratings are ever stored — downstream models must never see a
rating that already includes the fight being predicted.
"""

from __future__ import annotations

import sys
from typing import Iterable

from .db import get_conn

START_RATING = 1500.0
BASE_K = 36.0
NEW_FIGHTER_BOOST = 1.5   # applied for a fighter's first 5 UFC fights
METHOD_MULT = {
    "KO/TKO": 1.25, "SUB": 1.25,
    "U-DEC": 1.0, "M-DEC": 0.9, "S-DEC": 0.8,
    "DQ": 0.7, "DRAW": 1.0, "NC": 0.0,   # NC: no rating change
}

ELO_SCHEMA = """
CREATE TABLE IF NOT EXISTS fight_elo (
    fight_id   TEXT PRIMARY KEY REFERENCES fights(fight_id),
    elo_a_pre  REAL NOT NULL,
    elo_b_pre  REAL NOT NULL,
    n_fights_a INTEGER NOT NULL,   -- prior UFC fights, pre-fight
    n_fights_b INTEGER NOT NULL
);
"""


def expected_score(ra: float, rb: float) -> float:
    """Expected score of the first fighter under the Elo formula."""
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def run_elo(
    fights: Iterable[dict],
    ratings: dict[str, float] | None = None,
    n_fights: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, float], dict[str, int]]:
    """Run the Elo procedure over *fights* (already in chronological order).

    Each fight is a dict with fight_id, fighter_a_id, fighter_b_id, winner_id
    (None for draw/NC) and method. Optional *ratings* / *n_fights* seed the
    state (used by tests and as-of-date prediction). Pure: does not touch the
    DB. Returns (per-fight pre-rating rows, final ratings, final n_fights).
    """
    ratings = dict(ratings) if ratings else {}
    n_fights = dict(n_fights) if n_fights else {}
    rows: list[dict] = []

    for f in fights:
        a, b = f["fighter_a_id"], f["fighter_b_id"]
        ra = ratings.get(a, START_RATING)
        rb = ratings.get(b, START_RATING)
        na = n_fights.get(a, 0)
        nb = n_fights.get(b, 0)
        rows.append(
            {
                "fight_id": f["fight_id"],
                "elo_a_pre": ra,
                "elo_b_pre": rb,
                "n_fights_a": na,
                "n_fights_b": nb,
            }
        )

        method = f.get("method")
        winner = f.get("winner_id")
        if method == "NC":
            s_a = None  # no rating change at all
        elif winner == a:
            s_a = 1.0
        elif winner == b:
            s_a = 0.0
        elif method == "DRAW":
            s_a = 0.5
        else:
            # No winner and not an explicit draw: treat like NC.
            s_a = None

        if s_a is not None:
            mult = METHOD_MULT.get(method)
            if mult is None:
                print(
                    f"warning: unknown method {method!r} for fight {f['fight_id']}; "
                    "using K multiplier 1.0",
                    file=sys.stderr,
                )
                mult = 1.0
            k = BASE_K * mult
            k_a = k * (NEW_FIGHTER_BOOST if na < 5 else 1.0)
            k_b = k * (NEW_FIGHTER_BOOST if nb < 5 else 1.0)
            e_a = expected_score(ra, rb)
            ratings[a] = ra + k_a * (s_a - e_a)
            ratings[b] = rb + k_b * ((1 - s_a) - (1 - e_a))
        else:
            ratings.setdefault(a, ra)
            ratings.setdefault(b, rb)

        n_fights[a] = na + 1
        n_fights[b] = nb + 1

    return rows, ratings, n_fights


def _load_fights_chronological(conn, before_date: str | None = None) -> list[dict]:
    query = """
        SELECT f.fight_id, f.fighter_a_id, f.fighter_b_id, f.winner_id, f.method
        FROM fights f
        JOIN events e ON e.event_id = f.event_id
    """
    params: tuple[str, ...] = ()
    if before_date is not None:
        query += " WHERE e.event_date < ?"
        params = (before_date,)
    query += " ORDER BY e.event_date ASC, f.fight_id ASC"
    cur = conn.execute(query, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def compute_and_store() -> int:
    """Full recompute of the fight_elo table. Returns the row count."""
    conn = get_conn()
    try:
        conn.executescript(ELO_SCHEMA)
        rows, _, _ = run_elo(_load_fights_chronological(conn))
        conn.executemany(
            """
            INSERT INTO fight_elo
                (fight_id, elo_a_pre, elo_b_pre, n_fights_a, n_fights_b)
            VALUES
                (:fight_id, :elo_a_pre, :elo_b_pre, :n_fights_a, :n_fights_b)
            ON CONFLICT(fight_id) DO UPDATE SET
                elo_a_pre=excluded.elo_a_pre,
                elo_b_pre=excluded.elo_b_pre,
                n_fights_a=excluded.n_fights_a,
                n_fights_b=excluded.n_fights_b
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def current_ratings() -> list[tuple[str, str, float, int]]:
    """Every fighter's latest rating: (fighter_id, name, rating, n_fights), desc."""
    conn = get_conn()
    try:
        _, ratings, n_fights = run_elo(_load_fights_chronological(conn))
        names = dict(conn.execute("SELECT fighter_id, name FROM fighters"))
    finally:
        conn.close()
    out = [
        (fid, names.get(fid, "?"), rating, n_fights.get(fid, 0))
        for fid, rating in ratings.items()
    ]
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def rating_state_as_of(conn, as_of_date: str) -> tuple[dict[str, float], dict[str, int]]:
    """Return post-history ratings and fight counts strictly before a date."""
    _, ratings, n_fights = run_elo(
        _load_fights_chronological(conn, before_date=as_of_date)
    )
    return ratings, n_fights


def main() -> None:
    n = compute_and_store()
    print(f"fight_elo recomputed: {n} rows")
    print("\nTop 15 by current rating:")
    for fid, name, rating, nf in current_ratings()[:15]:
        print(f"  {rating:7.1f}  {name}  ({nf} fights)")


if __name__ == "__main__":
    main()
