"""Parse the raw UFC-DataLab CSVs into the normalised SQLite schema.

Run as:  python -m ufcpred.parse

Reads the two dataframes produced by ``ufcpred.ingest.load_raw`` and populates
the four tables defined in ``ufcpred.db`` (events, fighters, fights,
fight_stats). All writes use ``INSERT OR REPLACE`` keyed on deterministic,
name-derived surrogate ids, so re-running the module is idempotent.

Surrogate keys (all sha1 hexdigests sliced to 16 chars):
    fighter_id = sha1(UPPER(name))[:16]
    event_id   = sha1(event_name + '|' + iso_date)[:16]
    fight_id   = sha1(event_id + '|' + UPPER(red) + '|' + UPPER(blue))[:16]

Known limitation (logged, not fixed): keys are name-based, and MMA has
duplicate fighter names. Join anomalies -- fight_id collisions, fighters that
appear in fights but not in the fighter-details CSV, and unparseable values --
are written to ``data/ingest_warnings.log``.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime

import pandas as pd

from .db import DB_PATH, get_conn, init_db

REPO_ROOT = DB_PATH.parent.parent
WARNINGS_PATH = DB_PATH.parent / "ingest_warnings.log"

# Null sentinels seen across the raw data. Any of these (or NaN / empty) -> NULL.
_NULL_TOKENS = {"", "--", "---", "nan", "n/a", "na"}

# Weight classes ordered so that the most specific / qualified name is matched
# first (Women's variants before their open-division counterparts, and
# "Light Heavyweight" before "Heavyweight"). Extraction from bout_type is a
# substring search against this ordered list.
WEIGHT_CLASSES = [
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Light Heavyweight",
    "Catch Weight",
    "Open Weight",
    "Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Strawweight",
]

# Raw method column -> normalised code. Outcome (draw / no_contest) overrides
# this mapping; see norm_method.
_METHOD_MAP = {
    "Decision - Unanimous": "U-DEC",
    "Decision - Split": "S-DEC",
    "Decision - Majority": "M-DEC",
    "KO/TKO": "KO/TKO",
    "TKO - Doctor's Stoppage": "KO/TKO",
    "Submission": "SUB",
    "DQ": "DQ",
    "Overturned": "NC",
    "Could Not Continue": "NC",
    "Other": "NC",
}


# --------------------------------------------------------------------------- #
# Null handling
# --------------------------------------------------------------------------- #
def _is_null(s) -> bool:
    """True if *s* should be treated as a missing value (NaN, None, sentinel)."""
    if s is None:
        return True
    if isinstance(s, float) and math.isnan(s):
        return True
    return str(s).strip().lower() in _NULL_TOKENS


# --------------------------------------------------------------------------- #
# Pure converter functions (unit-tested in tests/test_parse.py)
# --------------------------------------------------------------------------- #
def parse_of(s) -> tuple[int | None, int | None]:
    """'23 of 63' -> (23, 63); sentinels / NaN -> (None, None)."""
    if _is_null(s):
        return (None, None)
    m = re.match(r"^\s*(\d+)\s+of\s+(\d+)\s*$", str(s))
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


def parse_mmss(s) -> int | None:
    """'3:08' -> 188 seconds; sentinels / NaN -> None."""
    if _is_null(s):
        return None
    m = re.match(r"^\s*(\d+):([0-5]?\d)\s*$", str(s))
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_height(s) -> float | None:
    """\"6' 0\\\"\" -> centimetres (1 dp); sentinels / NaN -> None."""
    if _is_null(s):
        return None
    m = re.match(r"^\s*(\d+)\s*'\s*(\d+)\s*\"?\s*$", str(s))
    if not m:
        return None
    total_inches = int(m.group(1)) * 12 + int(m.group(2))
    return round(total_inches * 2.54, 1)


def parse_reach(s) -> float | None:
    """'72\"' -> centimetres (1 dp); sentinels / NaN -> None."""
    if _is_null(s):
        return None
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*\"?\s*$", str(s))
    if not m:
        return None
    return round(float(m.group(1)) * 2.54, 1)


def parse_date(s) -> str | None:
    """Parse the several date formats in the raw data to ISO 'YYYY-MM-DD'.

    Handles DD/MM/YYYY ('27/06/2026'), 'Sep 20, 1989', 'July 11, 2026', and
    ISO '2026-07-11'. Unparseable / missing -> None.
    """
    if _is_null(s):
        return None
    text = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def norm_method(method, fight_outcome) -> str:
    """Normalise the method string, letting outcome override for draw / NC.

    draw -> 'DRAW', no_contest -> 'NC'. Otherwise the method column is mapped
    via _METHOD_MAP; anything unrecognised falls back to 'NC'.
    """
    outcome = "" if _is_null(fight_outcome) else str(fight_outcome).strip().lower()
    if outcome == "draw":
        return "DRAW"
    if outcome == "no_contest":
        return "NC"
    if _is_null(method):
        return "NC"
    return _METHOD_MAP.get(str(method).strip(), "NC")


def clean_weight_class(bout_type) -> tuple[str | None, bool]:
    """Extract the weight class from a bout_type string.

    Returns (weight_class, ok). ``ok`` is False when no known class could be
    found; in that case the raw bout_type is returned so the caller can store
    it verbatim and log the miss.
    """
    if _is_null(bout_type):
        return (None, False)
    text = str(bout_type)
    for wc in WEIGHT_CLASSES:
        if wc in text:
            return (wc, True)
    return (text.strip(), False)


# --------------------------------------------------------------------------- #
# Surrogate ids
# --------------------------------------------------------------------------- #
def _sha16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def fighter_id(name: str) -> str:
    return _sha16(name.upper())


def event_id(event_name: str, iso_date: str) -> str:
    return _sha16(f"{event_name}|{iso_date}")


def fight_id(evt_id: str, red_name: str, blue_name: str) -> str:
    return _sha16(f"{evt_id}|{red_name.upper()}|{blue_name.upper()}")


# --------------------------------------------------------------------------- #
# Row transforms
# --------------------------------------------------------------------------- #
def build_fight_and_stats(row: dict, warnings: list[str] | None = None) -> tuple[dict, list[dict]]:
    """Transform one raw fights row (dict) into a fight dict + two stat dicts.

    Pure aside from the optional *warnings* list it appends to. The row is keyed
    by the stats_raw.csv column names; ``red_name_upper`` / ``blue_name_upper``
    helper columns are expected to be present (added by load_raw).
    """
    if warnings is None:
        warnings = []

    red_upper = str(row["red_name_upper"])
    blue_upper = str(row["blue_name_upper"])
    iso_date = parse_date(row.get("event_date"))
    if iso_date is None:
        warnings.append(
            f"unparseable event_date '{row.get('event_date')}' for "
            f"{red_upper} vs {blue_upper} at '{row.get('event_name')}'"
        )
        iso_date = ""

    evt_id = event_id(str(row.get("event_name", "")), iso_date)
    fid = fight_id(evt_id, red_upper, blue_upper)
    red_id = fighter_id(red_upper)
    blue_id = fighter_id(blue_upper)

    outcome = "" if _is_null(row.get("fight_outcome")) else str(row["fight_outcome"]).strip().lower()
    if outcome == "red_win":
        winner_id = red_id
    elif outcome == "blue_win":
        winner_id = blue_id
    else:  # draw / no_contest / unknown
        winner_id = None

    weight_class, wc_ok = clean_weight_class(row.get("bout_type"))
    if not wc_ok and weight_class is not None:
        warnings.append(
            f"could not extract weight class from bout_type '{row.get('bout_type')}' "
            f"({red_upper} vs {blue_upper}); stored raw"
        )

    bout_type_str = "" if _is_null(row.get("bout_type")) else str(row["bout_type"])
    is_title = 1 if "Title" in bout_type_str else 0

    end_round = None
    if not _is_null(row.get("round")):
        try:
            end_round = int(str(row["round"]).strip())
        except ValueError:
            end_round = None

    fight = {
        "fight_id": fid,
        "event_id": evt_id,
        "fighter_a_id": red_id,
        "fighter_b_id": blue_id,
        "winner_id": winner_id,
        "weight_class": weight_class,
        "method": norm_method(row.get("method"), row.get("fight_outcome")),
        "end_round": end_round,
        "end_time_sec": parse_mmss(row.get("time")),
        "is_title": is_title,
    }

    stats = []
    for corner, fid_val in (("red", red_id), ("blue", blue_id)):
        sig_l, sig_a = parse_of(row.get(f"{corner}_fighter_sig_str"))
        tot_l, tot_a = parse_of(row.get(f"{corner}_fighter_total_str"))
        td_l, td_a = parse_of(row.get(f"{corner}_fighter_TD"))
        kd = row.get(f"{corner}_fighter_KD")
        sub = row.get(f"{corner}_fighter_sub_att")
        rev = row.get(f"{corner}_fighter_rev")
        stats.append(
            {
                "fight_id": fid,
                "fighter_id": fid_val,
                "knockdowns": None if _is_null(kd) else int(float(kd)),
                "sig_strikes_landed": sig_l,
                "sig_strikes_attempted": sig_a,
                "total_strikes_landed": tot_l,
                "total_strikes_attempted": tot_a,
                "takedowns_landed": td_l,
                "takedowns_attempted": td_a,
                "sub_attempts": None if _is_null(sub) else int(float(sub)),
                "reversals": None if _is_null(rev) else int(float(rev)),
                "control_time_sec": parse_mmss(row.get(f"{corner}_fighter_ctrl")),
            }
        )

    return fight, stats


def build_fighter_row(name_as_in_fights: str, details: dict | None) -> dict:
    """Build a fighters-table row.

    If *details* (a raw_fighter_details row dict) is provided, use its
    mixed-case name and parsed physicals; otherwise fall back to the
    (uppercase) name from the fights table with NULL physicals.
    """
    if details is None:
        return {
            "fighter_id": fighter_id(name_as_in_fights),
            "name": name_as_in_fights,
            "height_cm": None,
            "reach_cm": None,
            "stance": None,
            "dob": None,
        }
    stance = details.get("Stance")
    return {
        "fighter_id": fighter_id(name_as_in_fights),
        "name": details.get("fighter_name") or name_as_in_fights,
        "height_cm": parse_height(details.get("Height")),
        "reach_cm": parse_reach(details.get("Reach")),
        "stance": None if _is_null(stance) else str(stance).strip(),
        "dob": parse_date(details.get("DOB")),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    """Load raw data, transform, and populate the DB. Returns a counts dict."""
    from .ingest import load_raw  # local import: keeps converters DB/CSV-free

    fights_df, fighters_df = load_raw()
    warnings: list[str] = []

    # Fighter details lookup by uppercase name (last one wins on dup names).
    details_by_upper: dict[str, dict] = {}
    dup_detail_names = 0
    for rec in fighters_df.to_dict("records"):
        key = str(rec["name_upper"])
        if key in details_by_upper:
            dup_detail_names += 1
        details_by_upper[key] = rec

    fight_rows: list[dict] = []
    stat_rows: list[dict] = []
    fight_id_seen: dict[str, int] = {}
    fight_id_collisions = 0
    fighters_upper: dict[str, str] = {}  # upper name -> name as in fights

    for row in fights_df.to_dict("records"):
        fight, stats = build_fight_and_stats(row, warnings)
        if fight["fight_id"] in fight_id_seen:
            fight_id_collisions += 1
            fight_id_seen[fight["fight_id"]] += 1
            warnings.append(
                f"duplicate fight_id {fight['fight_id']} "
                f"({row['red_name_upper']} vs {row['blue_name_upper']} @ "
                f"'{row.get('event_name')}'); keeping last"
            )
        else:
            fight_id_seen[fight["fight_id"]] = 1
        fight_rows.append(fight)
        stat_rows.extend(stats)
        for upper in (str(row["red_name_upper"]), str(row["blue_name_upper"])):
            fighters_upper.setdefault(upper, upper)

    # Events derived from the (deduped) fight rows.
    events_by_id: dict[str, dict] = {}
    for row in fights_df.to_dict("records"):
        iso_date = parse_date(row.get("event_date")) or ""
        ename = str(row.get("event_name", ""))
        eid = event_id(ename, iso_date)
        events_by_id[eid] = {
            "event_id": eid,
            "name": ename,
            "event_date": iso_date,
            "location": None if _is_null(row.get("event_location")) else str(row["event_location"]).strip(),
        }

    # Fighter rows: every fighter appearing in a fight.
    fighter_rows: list[dict] = []
    missing_details = 0
    for upper, name_in_fights in fighters_upper.items():
        details = details_by_upper.get(upper)
        if details is None:
            missing_details += 1
            warnings.append(f"fighter '{name_in_fights}' has no fighter-details row; physicals NULL")
        fighter_rows.append(build_fighter_row(name_in_fights, details))

    # Fighters present in the details file but never appearing in a fight —
    # kept so name resolution (predict CLI) covers every known fighter.
    for upper, details in details_by_upper.items():
        if upper not in fighters_upper:
            fighter_rows.append(build_fighter_row(str(details.get("fighter_name") or upper), details))

    # ------------------------------------------------------------------- #
    # Write to the DB (INSERT OR REPLACE; order respects FKs).
    # ------------------------------------------------------------------- #
    init_db()
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO fighters "
            "(fighter_id, name, height_cm, reach_cm, stance, dob) "
            "VALUES (:fighter_id, :name, :height_cm, :reach_cm, :stance, :dob)",
            fighter_rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO events (event_id, name, event_date, location) "
            "VALUES (:event_id, :name, :event_date, :location)",
            list(events_by_id.values()),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO fights "
            "(fight_id, event_id, fighter_a_id, fighter_b_id, winner_id, weight_class, "
            "method, end_round, end_time_sec, is_title) VALUES "
            "(:fight_id, :event_id, :fighter_a_id, :fighter_b_id, :winner_id, :weight_class, "
            ":method, :end_round, :end_time_sec, :is_title)",
            fight_rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO fight_stats "
            "(fight_id, fighter_id, knockdowns, sig_strikes_landed, sig_strikes_attempted, "
            "total_strikes_landed, total_strikes_attempted, takedowns_landed, takedowns_attempted, "
            "sub_attempts, reversals, control_time_sec) VALUES "
            "(:fight_id, :fighter_id, :knockdowns, :sig_strikes_landed, :sig_strikes_attempted, "
            ":total_strikes_landed, :total_strikes_attempted, :takedowns_landed, :takedowns_attempted, "
            ":sub_attempts, :reversals, :control_time_sec)",
            stat_rows,
        )
        conn.commit()

        counts = {
            "fighters": conn.execute("SELECT COUNT(*) FROM fighters").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "fights": conn.execute("SELECT COUNT(*) FROM fights").fetchone()[0],
            "fight_stats": conn.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0],
            "winner_null": conn.execute("SELECT COUNT(*) FROM fights WHERE winner_id IS NULL").fetchone()[0],
        }
    finally:
        conn.close()

    # Warnings summary header + detail lines.
    header = [
        f"# ingest warnings ({datetime.now().isoformat(timespec='seconds')})",
        f"# source rows: {len(fights_df)}; unique fight_ids: {len(fight_id_seen)}",
        f"# fight_id collisions (duplicate source rows, last kept): {fight_id_collisions}",
        f"# fighters missing detail rows: {missing_details}",
        f"# duplicate fighter-detail names (last kept): {dup_detail_names}",
    ]
    WARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WARNINGS_PATH.write_text("\n".join(header + warnings) + "\n", encoding="utf-8")

    counts["fight_id_collisions"] = fight_id_collisions
    counts["missing_details"] = missing_details
    counts["warnings_lines"] = len(header) + len(warnings)
    return counts


def main() -> None:
    counts = run()
    total_fights = counts["fights"]
    null_pct = 100.0 * counts["winner_null"] / total_fights if total_fights else 0.0
    print("=== ufcpred.parse complete ===")
    print(f"fighters   : {counts['fighters']}")
    print(f"events     : {counts['events']}")
    print(f"fights     : {counts['fights']}")
    print(f"fight_stats: {counts['fight_stats']}  (expected ~2x fights)")
    print(f"winner NULL: {counts['winner_null']} ({null_pct:.2f}%)  -> non-NULL {100 - null_pct:.2f}%")
    print(f"fight_id collisions (dup source rows, last kept): {counts['fight_id_collisions']}")
    print(f"fighters missing detail rows: {counts['missing_details']}")
    print(f"warnings written to {WARNINGS_PATH} ({counts['warnings_lines']} lines)")


if __name__ == "__main__":
    main()
