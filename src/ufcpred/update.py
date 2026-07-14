"""Preservation-safe incremental UFC data updates.

By default this runs the vendored UFC-DataLab Scrapy spider for events after the
latest database date, ingests only unseen fight ids, and rebuilds downstream
state when new fights arrive. Every refresh export and log is retained under
``data/refreshes``; update history is append-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .db import DB_PATH, REPO_ROOT, get_conn
from .features import run as run_features
from .ingest import _strip_string_columns
from .parse import build_fight_and_stats, event_id, fight_id, parse_date
from .ratings import compute_and_store
from .train import MODEL_PATH, train

VENDOR_DIR = REPO_ROOT / "vendor" / "UFC-DataLab"
SPIDER_DIR = VENDOR_DIR / "src" / "scraping" / "ufc_stats"
REFRESH_ROOT = DB_PATH.parent / "refreshes"
SCRAPY_COMPAT_DIR = REPO_ROOT / ".venv" / "scrapy_compat"
UPDATE_HISTORY_PATH = DB_PATH.parent / "update_history.jsonl"
UPDATE_WARNINGS_PATH = DB_PATH.parent / "update_warnings.log"
USER_AGENT = "ufcpred/0.1 (historical UFC research; one request at a time)"

REQUIRED_REFRESH_COLUMNS = {
    "red_fighter_name",
    "blue_fighter_name",
    "event_date",
    "event_name",
    "fight_outcome",
}


class UpdateError(RuntimeError):
    """A safe, user-facing update failure."""


def _unique_refresh_dir() -> Path:
    REFRESH_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = REFRESH_ROOT / stamp
    counter = 1
    while candidate.exists():
        candidate = REFRESH_ROOT / f"{stamp}-{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


def latest_event_date(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(event_date) FROM events").fetchone()
    return row[0] if row else None


def spider_command(since_iso: str, output_path: Path) -> list[str]:
    """Build the pinned, deliberately polite Scrapy command."""
    parsed = datetime.strptime(since_iso, "%Y-%m-%d")
    since = parsed.strftime("%d/%m/%Y")
    return [
        sys.executable,
        "-m",
        "scrapy",
        "crawl",
        "stats_spider",
        "-a",
        f"since={since}",
        "-O",
        str(output_path.resolve()),
        "-s",
        "CONCURRENT_REQUESTS=1",
        "-s",
        "CONCURRENT_REQUESTS_PER_DOMAIN=1",
        "-s",
        "DOWNLOAD_DELAY=2",
        "-s",
        "RANDOMIZE_DOWNLOAD_DELAY=False",
        "-s",
        f"USER_AGENT={USER_AGENT}",
    ]


def run_spider(since_iso: str) -> Path:
    """Run the incremental spider and retain its export and complete process log."""
    if not (SPIDER_DIR / "scrapy.cfg").exists():
        raise UpdateError(
            f"vendored spider not found at {SPIDER_DIR}; restore vendor/UFC-DataLab"
        )
    refresh_dir = _unique_refresh_dir()
    output_path = refresh_dir / "ufc_stats_refresh.csv"
    command = spider_command(since_iso, output_path)
    environment = os.environ.copy()
    if SCRAPY_COMPAT_DIR.exists():
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(SCRAPY_COMPAT_DIR) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
    completed = subprocess.run(
        command,
        cwd=SPIDER_DIR,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = (
        "command: "
        + subprocess.list2cmdline(command)
        + f"\nexit_code: {completed.returncode}\n\nSTDOUT\n{completed.stdout}"
        + f"\n\nSTDERR\n{completed.stderr}"
    )
    (refresh_dir / "scrapy.log").write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        raise UpdateError(
            f"Scrapy refresh failed with exit code {completed.returncode}; "
            f"details preserved in {refresh_dir / 'scrapy.log'}"
        )
    if not output_path.exists():
        output_path.touch()
    return output_path


def load_refresh_csv(path: Path) -> pd.DataFrame:
    """Load either Scrapy comma CSV or UFC-DataLab semicolon raw CSV."""
    if not path.exists():
        raise UpdateError(f"refresh CSV not found: {path}")
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=sorted(REQUIRED_REFRESH_COLUMNS))
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=sorted(REQUIRED_REFRESH_COLUMNS))
    missing = REQUIRED_REFRESH_COLUMNS.difference(frame.columns)
    if missing:
        raise UpdateError(f"refresh CSV is missing required columns: {sorted(missing)}")
    frame = _strip_string_columns(frame)
    frame["red_name_upper"] = frame["red_fighter_name"].str.upper()
    frame["blue_name_upper"] = frame["blue_fighter_name"].str.upper()
    return frame


def _append_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    UPDATE_WARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UPDATE_WARNINGS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"# update warnings {datetime.now(timezone.utc).isoformat()}\n")
        for warning in warnings:
            handle.write(warning + "\n")


def ingest_new_fights(conn: sqlite3.Connection, frame: pd.DataFrame) -> dict:
    """Insert unseen fight ids and dependencies; never remove or replace a row."""
    existing_fights = {row[0] for row in conn.execute("SELECT fight_id FROM fights")}
    existing_fighters = {row[0] for row in conn.execute("SELECT fighter_id FROM fighters")}
    new_fighter_ids: set[str] = set()
    seen_refresh_ids: set[str] = set()
    warnings: list[str] = []
    inserted_fights = 0

    for raw in frame.to_dict("records"):
        iso_date = parse_date(raw.get("event_date"))
        if iso_date is None:
            warnings.append(
                f"skipped row with unparseable date {raw.get('event_date')!r}: "
                f"{raw.get('red_fighter_name')} vs {raw.get('blue_fighter_name')}"
            )
            continue
        raw["event_date"] = iso_date
        raw["red_name_upper"] = str(raw["red_fighter_name"]).strip().upper()
        raw["blue_name_upper"] = str(raw["blue_fighter_name"]).strip().upper()
        prospective_event_id = event_id(str(raw.get("event_name", "")), iso_date)
        fight_key = fight_id(
            prospective_event_id, raw["red_name_upper"], raw["blue_name_upper"]
        )
        if fight_key in existing_fights:
            continue
        if fight_key in seen_refresh_ids:
            warnings.append(
                f"duplicate refresh fight_id {fight_key}; first row preserved, later row skipped"
            )
            continue
        seen_refresh_ids.add(fight_key)

        row_warnings: list[str] = []
        fight, stats = build_fight_and_stats(raw, row_warnings)
        warnings.extend(row_warnings)

        for corner, id_key in (("red", "fighter_a_id"), ("blue", "fighter_b_id")):
            fighter_key = fight[id_key]
            fighter_name = str(raw[f"{corner}_fighter_name"]).strip()
            conn.execute(
                """
                INSERT INTO fighters
                    (fighter_id, name, height_cm, reach_cm, stance, dob)
                VALUES (?, ?, NULL, NULL, NULL, NULL)
                ON CONFLICT(fighter_id) DO NOTHING
                """,
                (fighter_key, fighter_name),
            )
            if fighter_key not in existing_fighters:
                new_fighter_ids.add(fighter_key)
                existing_fighters.add(fighter_key)

        event_name = str(raw.get("event_name", "")).strip()
        location = raw.get("event_location")
        if pd.isna(location) or str(location).strip() in {"", "--", "---", "-"}:
            location = None
        else:
            location = str(location).strip()
        conn.execute(
            """
            INSERT INTO events (event_id, name, event_date, location)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id) DO NOTHING
            """,
            (fight["event_id"], event_name, iso_date, location),
        )
        conn.execute(
            """
            INSERT INTO fights
                (fight_id, event_id, fighter_a_id, fighter_b_id, winner_id,
                 weight_class, method, end_round, end_time_sec, is_title)
            VALUES
                (:fight_id, :event_id, :fighter_a_id, :fighter_b_id, :winner_id,
                 :weight_class, :method, :end_round, :end_time_sec, :is_title)
            ON CONFLICT(fight_id) DO NOTHING
            """,
            fight,
        )
        conn.executemany(
            """
            INSERT INTO fight_stats
                (fight_id, fighter_id, knockdowns, sig_strikes_landed,
                 sig_strikes_attempted, total_strikes_landed,
                 total_strikes_attempted, takedowns_landed,
                 takedowns_attempted, sub_attempts, reversals, control_time_sec)
            VALUES
                (:fight_id, :fighter_id, :knockdowns, :sig_strikes_landed,
                 :sig_strikes_attempted, :total_strikes_landed,
                 :total_strikes_attempted, :takedowns_landed,
                 :takedowns_attempted, :sub_attempts, :reversals,
                 :control_time_sec)
            ON CONFLICT(fight_id, fighter_id) DO NOTHING
            """,
            stats,
        )
        existing_fights.add(fight_key)
        inserted_fights += 1

    conn.commit()
    _append_warnings(warnings)
    return {
        "new_fights": inserted_fights,
        "new_fighters": len(new_fighter_ids),
        "warnings": len(warnings),
    }


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(VENDOR_DIR), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise UpdateError(
            f"vendor git command failed: git {' '.join(args)}\n{completed.stderr.strip()}"
        )
    return completed


def update_vendor(*, allow_upstream_removals: bool = False) -> dict:
    """Fetch and fast-forward upstream after auditing path removals/renames."""
    if not (VENDOR_DIR / ".git").exists():
        raise UpdateError(f"vendored git checkout not found at {VENDOR_DIR}")
    before = _run_git(["rev-parse", "HEAD"]).stdout.strip()
    _run_git(["fetch", "origin"])
    remote = _run_git(
        ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"]
    ).stdout.strip()
    changes = _run_git(["diff", "--name-status", "HEAD", remote]).stdout.splitlines()
    removals = [line for line in changes if line.startswith(("D\t", "R"))]
    if removals and not allow_upstream_removals:
        detail = "\n".join(removals)
        raise UpdateError(
            "upstream contains deletions or renames; no merge was performed. "
            "Review these paths and rerun with --allow-upstream-removals only after "
            f"explicit human approval:\n{detail}"
        )
    if changes:
        _run_git(["merge", "--ff-only", remote])
    after = _run_git(["rev-parse", "HEAD"]).stdout.strip()
    return {"before": before, "after": after, "changed": before != after}


def _training_due() -> bool:
    if not MODEL_PATH.exists():
        return True
    try:
        artifact = joblib.load(MODEL_PATH)
        trained = datetime.fromisoformat(artifact["trained_at_utc"])
    except (KeyError, TypeError, ValueError):
        return True
    now = datetime.now(timezone.utc)
    return (trained.year, trained.month) != (now.year, now.month)


def _append_history(record: dict) -> None:
    UPDATE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UPDATE_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_update(
    *,
    refresh_csv: Path | None = None,
    pull_upstream: bool = False,
    allow_upstream_removals: bool = False,
    force_retrain: bool = False,
) -> dict:
    """Run one incremental update and append its audit record."""
    started = datetime.now(timezone.utc)
    vendor_result = None
    if pull_upstream:
        vendor_result = update_vendor(
            allow_upstream_removals=allow_upstream_removals
        )

    conn = get_conn()
    try:
        latest = latest_event_date(conn)
    finally:
        conn.close()
    if latest is None:
        raise UpdateError("database has no events; run the bootstrap ingest first")

    source_path = Path(refresh_csv).resolve() if refresh_csv else run_spider(latest)
    frame = load_refresh_csv(source_path)
    conn = get_conn()
    try:
        ingest_result = ingest_new_fights(conn, frame)
    finally:
        conn.close()

    elo_rows = None
    feature_rows = None
    retrained = False
    if ingest_result["new_fights"]:
        elo_rows = compute_and_store()
        feature_rows = run_features()["rows"]
    if force_retrain or (ingest_result["new_fights"] and _training_due()):
        train()
        retrained = True

    result = {
        "timestamp_utc": started.isoformat(),
        "source_csv": str(source_path),
        "source_rows": len(frame),
        "latest_date_before": latest,
        "new_fights": ingest_result["new_fights"],
        "new_fighters": ingest_result["new_fighters"],
        "warnings": ingest_result["warnings"],
        "elo_rows": elo_rows,
        "feature_rows": feature_rows,
        "retrained": retrained,
        "vendor": vendor_result,
    }
    _append_history(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incrementally refresh UFC fight data.")
    parser.add_argument(
        "--refresh-csv",
        type=Path,
        help="use a preserved CSV instead of running the network spider",
    )
    parser.add_argument(
        "--pull-upstream",
        action="store_true",
        help="fetch and fast-forward the vendored UFC-DataLab checkout",
    )
    parser.add_argument(
        "--allow-upstream-removals",
        action="store_true",
        help="explicit human approval for an upstream merge containing deletions/renames",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="retrain even when the current monthly model is already present",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.allow_upstream_removals and not args.pull_upstream:
        print("error: --allow-upstream-removals requires --pull-upstream", file=sys.stderr)
        raise SystemExit(2)
    try:
        result = run_update(
            refresh_csv=args.refresh_csv,
            pull_upstream=args.pull_upstream,
            allow_upstream_removals=args.allow_upstream_removals,
            force_retrain=args.force_retrain,
        )
    except UpdateError as exc:
        failure = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": str(exc),
        }
        _append_history(failure)
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        f"update: {result['new_fights']} new fights, "
        f"{result['new_fighters']} new fighters"
    )
    print(f"source: {result['source_csv']} ({result['source_rows']} rows)")
    if result["elo_rows"] is not None:
        print(
            f"rebuilt: {result['elo_rows']} Elo rows, "
            f"{result['feature_rows']} feature rows"
        )
    print(f"retrained: {'yes' if result['retrained'] else 'no'}")


if __name__ == "__main__":
    main()
