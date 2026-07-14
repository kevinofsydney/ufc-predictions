"""Command-line predictions for an upcoming UFC matchup.

Example:
    python -m ufcpred.predict "Jon Jones" "Stipe Miocic" --date 2026-08-01
"""

from __future__ import annotations

import argparse
import difflib
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .db import get_conn
from .features import build_matchup_features
from .train import MODEL_PATH


class PredictionError(RuntimeError):
    """A user-facing prediction input or artifact error."""


def resolve_fighter(conn: sqlite3.Connection, requested_name: str) -> tuple[str, str]:
    """Resolve a Unicode-aware, case-insensitive exact fighter name."""
    target = requested_name.strip().casefold()
    all_fighters = conn.execute(
        "SELECT fighter_id, name FROM fighters ORDER BY name, fighter_id"
    ).fetchall()
    matches = [(fighter_id, name) for fighter_id, name in all_fighters if name.casefold() == target]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        candidates = "\n".join(f"  - {name} [{fighter_id}]" for fighter_id, name in matches)
        raise PredictionError(
            f"fighter name {requested_name!r} is ambiguous; candidates:\n{candidates}"
        )

    names = [name for _, name in all_fighters]
    suggestions = difflib.get_close_matches(requested_name, names, n=5, cutoff=0.6)
    suffix = ""
    if suggestions:
        suffix = "\nClose matches:\n" + "\n".join(f"  - {name}" for name in suggestions)
    raise PredictionError(f"fighter not found: {requested_name!r}{suffix}")


def load_model_artifact(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise PredictionError(
            f"model artifact not found at {path}; run python -m ufcpred.train first"
        )
    artifact = joblib.load(path)
    required = {"model", "model_type", "feature_columns", "imputation_medians"}
    missing = required.difference(artifact)
    if missing:
        raise PredictionError(f"model artifact is missing keys: {sorted(missing)}")
    return artifact


def probability_from_artifact(artifact: dict, features: dict) -> float:
    """Apply artifact-specific preprocessing and return fighter-A probability."""
    columns = artifact["feature_columns"]
    missing = [column for column in columns if column not in features]
    if missing:
        raise PredictionError(f"matchup features are missing columns: {missing}")
    frame = pd.DataFrame(
        [{column: features[column] for column in columns}],
        columns=columns,
        dtype=float,
    )
    if str(artifact["model_type"]).startswith("logistic"):
        frame = frame.fillna(pd.Series(artifact["imputation_medians"]))
    probability = float(artifact["model"].predict_proba(frame)[:, 1][0])
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise PredictionError(f"model returned an invalid probability: {probability}")
    return probability


def symmetric_probability_from_artifact(artifact: dict, features: dict) -> float:
    """Average both fighter orientations so input order cannot change the matchup."""
    forward = probability_from_artifact(artifact, features)
    reversed_features = {
        column: None if value is None else -value for column, value in features.items()
    }
    reverse = probability_from_artifact(artifact, reversed_features)
    return 0.5 * (forward + (1.0 - reverse))


def predict_matchup(
    fighter_one: str,
    fighter_two: str,
    as_of_date: str,
    *,
    model_path: Path = MODEL_PATH,
) -> dict:
    """Resolve fighters, build point-in-time features, and predict both sides."""
    try:
        parsed_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise PredictionError(f"invalid date {as_of_date!r}; expected YYYY-MM-DD") from exc
    if parsed_date.isoformat() != as_of_date:
        raise PredictionError(f"invalid date {as_of_date!r}; expected YYYY-MM-DD")

    conn = get_conn()
    try:
        fighter_a_id, canonical_a = resolve_fighter(conn, fighter_one)
        fighter_b_id, canonical_b = resolve_fighter(conn, fighter_two)
        if fighter_a_id == fighter_b_id:
            raise PredictionError("choose two different fighters")
        features, diagnostics = build_matchup_features(
            conn, fighter_a_id, fighter_b_id, as_of_date
        )
    finally:
        conn.close()

    artifact = load_model_artifact(model_path)
    probability_a = symmetric_probability_from_artifact(artifact, features)
    return {
        "fighter_a": canonical_a,
        "fighter_b": canonical_b,
        "probability_a": probability_a,
        "probability_b": 1.0 - probability_a,
        "as_of_date": as_of_date,
        "model_type": artifact["model_type"],
        **diagnostics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict win probabilities for an upcoming UFC matchup."
    )
    parser.add_argument("fighter_one", help="case-insensitive exact fighter name")
    parser.add_argument("fighter_two", help="case-insensitive exact fighter name")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="fight date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"model artifact path (default: {MODEL_PATH})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        result = predict_matchup(
            args.fighter_one, args.fighter_two, args.date, model_path=args.model
        )
    except (PredictionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"{result['fighter_a']}:  {100.0 * result['probability_a']:.1f}%")
    print(f"{result['fighter_b']}:  {100.0 * result['probability_b']:.1f}%")
    print(
        f"(elo: {result['elo_a']:.0f} vs {result['elo_b']:.0f} | "
        f"model: {result['model_type']} | "
        f"fights in db: {result['n_fights_a']} vs {result['n_fights_b']} | "
        f"as of: {result['as_of_date']})"
    )
    if result["n_fights_a"] < 3 or result["n_fights_b"] < 3:
        print("warning: low-confidence prediction; a fighter has fewer than 3 prior UFC fights")


if __name__ == "__main__":
    main()
