"""Unit tests for prediction name resolution and artifact inference."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from ufcpred.predict import (
    PredictionError,
    probability_from_artifact,
    resolve_fighter,
    symmetric_probability_from_artifact,
)


class _FixedModel:
    def predict_proba(self, frame):
        return np.array([[0.3, 0.7] for _ in range(len(frame))])


class _DirectionalModel:
    def predict_proba(self, frame):
        probability = 0.5 + frame.iloc[:, 0].to_numpy(dtype=float) / 1000.0
        return np.column_stack([1.0 - probability, probability])


def _fighters() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE fighters (fighter_id TEXT PRIMARY KEY, name TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO fighters VALUES (?, ?)",
        [("j", "Jon Jones"), ("i", "Islam Makhachev"), ("x", "José Aldo")],
    )
    return conn


def test_resolve_fighter_is_case_insensitive_and_unicode_aware():
    conn = _fighters()
    try:
        assert resolve_fighter(conn, "  JON JONES ") == ("j", "Jon Jones")
        assert resolve_fighter(conn, "josé aldo") == ("x", "José Aldo")
    finally:
        conn.close()


def test_resolve_fighter_gives_close_matches():
    conn = _fighters()
    try:
        with pytest.raises(PredictionError, match="Jon Jones"):
            resolve_fighter(conn, "John Jones")
    finally:
        conn.close()


def test_probability_from_artifact():
    artifact = {
        "model": _FixedModel(),
        "model_type": "lightgbm",
        "feature_columns": ["elo_diff", "age_diff"],
        "imputation_medians": {"elo_diff": 0.0, "age_diff": 0.0},
    }
    probability = probability_from_artifact(
        artifact, {"elo_diff": 100.0, "age_diff": None}
    )
    assert probability == pytest.approx(0.7)
    assert probability + (1.0 - probability) == pytest.approx(1.0)


def test_symmetric_probability_is_invariant_to_fighter_order():
    artifact = {
        "model": _DirectionalModel(),
        "model_type": "lightgbm",
        "feature_columns": ["elo_diff"],
        "imputation_medians": {"elo_diff": 0.0},
    }
    forward = symmetric_probability_from_artifact(artifact, {"elo_diff": 100.0})
    reverse = symmetric_probability_from_artifact(artifact, {"elo_diff": -100.0})
    assert forward == pytest.approx(0.6)
    assert reverse == pytest.approx(0.4)
    assert forward + reverse == pytest.approx(1.0)
