"""Unit tests for the pure converter functions and one full row transform.

These tests import ufcpred.parse and exercise the pure functions only; they do
not touch the SQLite DB or the vendored CSVs.
"""

from __future__ import annotations

import math

from ufcpred.parse import (
    build_fight_and_stats,
    clean_weight_class,
    norm_method,
    parse_date,
    parse_height,
    parse_mmss,
    parse_of,
    parse_reach,
)


# --------------------------------------------------------------------------- #
# parse_of
# --------------------------------------------------------------------------- #
def test_parse_of_basic():
    assert parse_of("23 of 63") == (23, 63)
    assert parse_of("0 of 0") == (0, 0)
    assert parse_of("6 of 8") == (6, 8)


def test_parse_of_sentinels():
    assert parse_of("--") == (None, None)
    assert parse_of("---") == (None, None)
    assert parse_of("") == (None, None)
    assert parse_of(None) == (None, None)
    assert parse_of(float("nan")) == (None, None)
    assert parse_of("garbage") == (None, None)


# --------------------------------------------------------------------------- #
# parse_mmss
# --------------------------------------------------------------------------- #
def test_parse_mmss_basic():
    assert parse_mmss("3:08") == 188
    assert parse_mmss("0:00") == 0
    assert parse_mmss("8:48") == 528
    assert parse_mmss("10:00") == 600


def test_parse_mmss_sentinels():
    assert parse_mmss("--") is None
    assert parse_mmss("---") is None
    assert parse_mmss("") is None
    assert parse_mmss(None) is None
    assert parse_mmss(float("nan")) is None
    assert parse_mmss("not a time") is None


# --------------------------------------------------------------------------- #
# parse_height / parse_reach
# --------------------------------------------------------------------------- #
def test_parse_height_basic():
    assert parse_height("5' 11\"") == round((5 * 12 + 11) * 2.54, 1)  # 180.3
    assert parse_height("6' 0\"") == round(72 * 2.54, 1)  # 182.9
    assert parse_height("5' 11\"") == 180.3


def test_parse_height_sentinels():
    assert parse_height("--") is None
    assert parse_height("") is None
    assert parse_height(None) is None
    assert parse_height(float("nan")) is None


def test_parse_reach_basic():
    assert parse_reach('72"') == round(72 * 2.54, 1)  # 182.9
    assert parse_reach("78\"") == 198.1
    assert parse_reach("70") == round(70 * 2.54, 1)


def test_parse_reach_sentinels():
    assert parse_reach("--") is None
    assert parse_reach("---") is None
    assert parse_reach("") is None
    assert parse_reach(None) is None
    assert parse_reach(float("nan")) is None


# --------------------------------------------------------------------------- #
# parse_date
# --------------------------------------------------------------------------- #
def test_parse_date_formats():
    assert parse_date("27/06/2026") == "2026-06-27"  # DD/MM/YYYY
    assert parse_date("Sep 20, 1989") == "1989-09-20"
    assert parse_date("July 11, 2026") == "2026-07-11"
    assert parse_date("2026-07-11") == "2026-07-11"  # ISO passthrough


def test_parse_date_dd_mm_disambiguation():
    # 13 can only be a month if MM/DD; DD/MM/YYYY gives day=13, month=06.
    assert parse_date("13/06/2020") == "2020-06-13"


def test_parse_date_sentinels():
    assert parse_date("--") is None
    assert parse_date("") is None
    assert parse_date(None) is None
    assert parse_date(float("nan")) is None
    assert parse_date("not a date") is None


# --------------------------------------------------------------------------- #
# norm_method
# --------------------------------------------------------------------------- #
def test_norm_method_mapping():
    assert norm_method("Decision - Unanimous", "red_win") == "U-DEC"
    assert norm_method("Decision - Split", "blue_win") == "S-DEC"
    assert norm_method("Decision - Majority", "red_win") == "M-DEC"
    assert norm_method("KO/TKO", "red_win") == "KO/TKO"
    assert norm_method("TKO - Doctor's Stoppage", "red_win") == "KO/TKO"
    assert norm_method("Submission", "blue_win") == "SUB"
    assert norm_method("DQ", "red_win") == "DQ"
    assert norm_method("Overturned", "red_win") == "NC"
    assert norm_method("Could Not Continue", "red_win") == "NC"
    assert norm_method("Other", "red_win") == "NC"


def test_norm_method_outcome_override():
    # draw / no_contest override whatever the method column says.
    assert norm_method("Decision - Unanimous", "draw") == "DRAW"
    assert norm_method("KO/TKO", "no_contest") == "NC"
    assert norm_method("Submission", "draw") == "DRAW"


def test_norm_method_unknown_and_null():
    assert norm_method("Some New Method", "red_win") == "NC"
    assert norm_method(None, "red_win") == "NC"
    assert norm_method(float("nan"), "red_win") == "NC"


# --------------------------------------------------------------------------- #
# clean_weight_class
# --------------------------------------------------------------------------- #
def test_clean_weight_class():
    assert clean_weight_class("Lightweight Bout") == ("Lightweight", True)
    assert clean_weight_class("UFC Lightweight Title Bout") == ("Lightweight", True)
    assert clean_weight_class("UFC Interim Heavyweight Title Bout") == ("Heavyweight", True)
    assert clean_weight_class("Light Heavyweight Bout") == ("Light Heavyweight", True)
    assert clean_weight_class("Women's Strawweight Bout") == ("Women's Strawweight", True)
    assert clean_weight_class("Women's Bantamweight Bout") == ("Women's Bantamweight", True)
    assert clean_weight_class("Catch Weight Bout") == ("Catch Weight", True)
    assert clean_weight_class("Road to UFC 4 Featherweight Tournament Title Bout") == (
        "Featherweight",
        True,
    )


def test_clean_weight_class_miss():
    wc, ok = clean_weight_class("UFC Superfight Championship Bout")
    assert ok is False
    assert wc == "UFC Superfight Championship Bout"  # raw returned on miss


# --------------------------------------------------------------------------- #
# Full row transform
# --------------------------------------------------------------------------- #
def _sample_row() -> dict:
    return {
        "red_fighter_name": "Abus Magomedov",
        "blue_fighter_name": "Michal Oleksiejczuk",
        "red_name_upper": "ABUS MAGOMEDOV",
        "blue_name_upper": "MICHAL OLEKSIEJCZUK",
        "event_date": "27/06/2026",
        "event_name": "UFC Fight Night: Fiziev vs. Torres",
        "event_location": "Baku, Azerbaijan",
        "fight_outcome": "red_win",
        "method": "Submission",
        "round": "1",
        "time": "3:25",
        "bout_type": "Middleweight Bout",
        "red_fighter_KD": "1",
        "blue_fighter_KD": "0",
        "red_fighter_sig_str": "10 of 27",
        "blue_fighter_sig_str": "5 of 19",
        "red_fighter_total_str": "10 of 27",
        "blue_fighter_total_str": "5 of 19",
        "red_fighter_TD": "0 of 0",
        "blue_fighter_TD": "0 of 0",
        "red_fighter_sub_att": "1",
        "blue_fighter_sub_att": "0",
        "red_fighter_rev": "0",
        "blue_fighter_rev": "0",
        "red_fighter_ctrl": "0:36",
        "blue_fighter_ctrl": "0:03",
    }


def test_build_fight_and_stats():
    from ufcpred.parse import event_id, fight_id, fighter_id

    fight, stats = build_fight_and_stats(_sample_row())

    expected_eid = event_id("UFC Fight Night: Fiziev vs. Torres", "2026-06-27")
    expected_fid = fight_id(expected_eid, "ABUS MAGOMEDOV", "MICHAL OLEKSIEJCZUK")
    red_id = fighter_id("ABUS MAGOMEDOV")
    blue_id = fighter_id("MICHAL OLEKSIEJCZUK")

    assert fight["fight_id"] == expected_fid
    assert fight["event_id"] == expected_eid
    assert fight["fighter_a_id"] == red_id
    assert fight["fighter_b_id"] == blue_id
    assert fight["winner_id"] == red_id  # red_win
    assert fight["weight_class"] == "Middleweight"
    assert fight["method"] == "SUB"
    assert fight["end_round"] == 1
    assert fight["end_time_sec"] == 205  # 3:25
    assert fight["is_title"] == 0

    assert len(stats) == 2
    red_stat, blue_stat = stats
    assert red_stat["fighter_id"] == red_id
    assert red_stat["knockdowns"] == 1
    assert red_stat["sig_strikes_landed"] == 10
    assert red_stat["sig_strikes_attempted"] == 27
    assert red_stat["total_strikes_landed"] == 10
    assert red_stat["total_strikes_attempted"] == 27
    assert red_stat["takedowns_landed"] == 0
    assert red_stat["takedowns_attempted"] == 0
    assert red_stat["sub_attempts"] == 1
    assert red_stat["reversals"] == 0
    assert red_stat["control_time_sec"] == 36  # 0:36

    assert blue_stat["fighter_id"] == blue_id
    assert blue_stat["knockdowns"] == 0
    assert blue_stat["sig_strikes_landed"] == 5
    assert blue_stat["control_time_sec"] == 3  # 0:03


def test_build_fight_and_stats_title_and_draw():
    row = _sample_row()
    row["bout_type"] = "UFC Lightweight Title Bout"
    row["fight_outcome"] = "draw"
    fight, _ = build_fight_and_stats(row)
    assert fight["is_title"] == 1
    assert fight["weight_class"] == "Lightweight"
    assert fight["method"] == "DRAW"
    assert fight["winner_id"] is None


def test_build_fight_and_stats_null_stats():
    row = _sample_row()
    row["red_fighter_TD"] = "---"
    row["red_fighter_ctrl"] = "--"
    row["red_fighter_KD"] = "--"
    fight, stats = build_fight_and_stats(row)
    red_stat = stats[0]
    assert red_stat["takedowns_landed"] is None
    assert red_stat["takedowns_attempted"] is None
    assert red_stat["control_time_sec"] is None
    assert red_stat["knockdowns"] is None
