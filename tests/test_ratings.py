"""Unit tests for the Elo engine (spec Phase 3)."""

from ufcpred.ratings import expected_score, run_elo


def _fight(fid="f1", winner=None, method="U-DEC"):
    return {
        "fight_id": fid,
        "fighter_a_id": "A",
        "fighter_b_id": "B",
        "winner_id": winner,
        "method": method,
    }


def test_udec_win_between_established_1500s():
    # Both at 1500 with >=5 prior fights: K = 36 * 1.0, expected 0.5 -> +/-18.
    rows, ratings, n_fights = run_elo(
        [_fight(winner="A", method="U-DEC")],
        ratings={"A": 1500.0, "B": 1500.0},
        n_fights={"A": 5, "B": 5},
    )
    assert ratings["A"] == 1518.0
    assert ratings["B"] == 1482.0
    assert rows[0]["elo_a_pre"] == 1500.0 and rows[0]["elo_b_pre"] == 1500.0
    assert n_fights == {"A": 6, "B": 6}


def test_expected_score_1600_vs_1400():
    assert round(expected_score(1600.0, 1400.0), 4) == 0.7597


def test_nc_changes_nothing():
    rows, ratings, n_fights = run_elo(
        [_fight(winner=None, method="NC")],
        ratings={"A": 1550.0, "B": 1450.0},
        n_fights={"A": 7, "B": 2},
    )
    assert ratings == {"A": 1550.0, "B": 1450.0}
    # Pre-fight values are still recorded, and n_fights still increments.
    assert rows[0] == {
        "fight_id": "f1",
        "elo_a_pre": 1550.0,
        "elo_b_pre": 1450.0,
        "n_fights_a": 7,
        "n_fights_b": 2,
    }
    assert n_fights == {"A": 8, "B": 3}
