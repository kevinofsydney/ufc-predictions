"""Small deterministic tests for Phase 5 training helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufcpred.train import chronological_split, elo_probabilities, evaluate


def test_chronological_split_boundaries():
    frame = pd.DataFrame(
        {
            "event_date": [
                "2022-12-31",
                "2023-01-01",
                "2023-12-31",
                "2024-01-01",
            ],
            "label": [0, 1, 0, 1],
        }
    )
    train, validation, test = chronological_split(frame)
    assert train["event_date"].tolist() == ["2022-12-31"]
    assert validation["event_date"].tolist() == ["2023-01-01", "2023-12-31"]
    assert test["event_date"].tolist() == ["2024-01-01"]


def test_elo_probabilities():
    probabilities = elo_probabilities(np.array([0.0, 200.0, -200.0]))
    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] == pytest.approx(0.7597, abs=0.0001)
    assert probabilities[2] == pytest.approx(1.0 - probabilities[1])


def test_coin_flip_metrics():
    result = evaluate(np.array([0, 1, 0, 1]), np.full(4, 0.5))
    assert result["log_loss"] == pytest.approx(0.693147)
    assert result["brier"] == pytest.approx(0.25)
    assert result["accuracy"] == pytest.approx(0.5)
