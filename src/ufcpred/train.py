"""Chronological model training and evaluation for UFC fight predictions.

Run as:  python -m ufcpred.train

The data split is fixed by event date. All preprocessing statistics are learned
from the training period only. Existing output artifacts are copied to a dated
archive before a new version is written, so a training run never discards an
older model or report.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .db import REPO_ROOT, get_conn
from .features import DIFF_COLUMNS

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

TRAIN_END = "2023-01-01"
VALIDATION_END = "2024-01-01"
CALIBRATION_ECE_THRESHOLD = 0.05

FEATURE_COLUMNS = list(DIFF_COLUMNS)
MODEL_PATH = REPO_ROOT / "models" / "model.pkl"
REPORTS_DIR = REPO_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.md"
CALIBRATION_PATH = REPORTS_DIR / "calibration.png"
IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.md"


def load_features() -> pd.DataFrame:
    """Load materialised Phase 4 features in deterministic order."""
    conn = get_conn()
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='features'"
        ).fetchone()
        if exists is None:
            raise RuntimeError("features table not found; run python -m ufcpred.features first")
        columns = ["fight_id", "event_date", *FEATURE_COLUMNS, "label"]
        frame = pd.read_sql_query(
            f"SELECT {', '.join(columns)} FROM features ORDER BY event_date, fight_id",
            conn,
        )
    finally:
        conn.close()
    if frame.empty:
        raise RuntimeError("features table is empty; run python -m ufcpred.features first")
    return frame


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the pinned train, validation, and test date partitions."""
    train = frame.loc[frame["event_date"] < TRAIN_END].copy()
    validation = frame.loc[
        (frame["event_date"] >= TRAIN_END) & (frame["event_date"] < VALIDATION_END)
    ].copy()
    test = frame.loc[frame["event_date"] >= VALIDATION_END].copy()
    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            "chronological split produced an empty partition: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )
    return train, validation, test


def evaluate(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate the three pinned binary-probability metrics."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)
    labels = np.asarray(y_true, dtype=int)
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
    }


def elo_probabilities(elo_diff: pd.Series | np.ndarray) -> np.ndarray:
    """Convert Elo rating differences into fighter-A win probabilities."""
    diff = np.asarray(elo_diff, dtype=float)
    return 1.0 / (1.0 + np.power(10.0, -diff / 400.0))


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted absolute calibration error over fixed-width probability bins."""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    bin_ids = np.digitize(probabilities, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
        )
    return error


def _preserve_existing(path: Path, run_stamp: str) -> Path | None:
    """Archive an existing artifact before replacement; never remove it."""
    if not path.exists():
        return None
    archive = path.with_name(f"{path.stem}.{run_stamp}{path.suffix}")
    counter = 1
    while archive.exists():
        archive = path.with_name(f"{path.stem}.{run_stamp}.{counter}{path.suffix}")
        counter += 1
    shutil.copy2(path, archive)
    return archive


def _write_text(path: Path, content: str, run_stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _preserve_existing(path, run_stamp)
    path.write_text(content, encoding="utf-8")


def _metrics_markdown(
    metrics: dict[str, dict[str, dict[str, float]]],
    split_sizes: dict[str, int],
    final_model_name: str,
    validation_ece: float,
    calibrated: bool,
) -> str:
    lines = [
        "# UFC prediction model metrics",
        "",
        "Chronological split: train before 2023, validation during 2023, test from 2024 onward.",
        "",
        f"- Train rows: {split_sizes['train']:,}",
        f"- Validation rows: {split_sizes['validation']:,}",
        f"- Test rows: {split_sizes['test']:,}",
        f"- Final persisted model: `{final_model_name}`",
        f"- Uncalibrated LightGBM validation ECE (10 bins): {validation_ece:.4f}",
        f"- Isotonic calibration applied: {'yes' if calibrated else 'no'} "
        f"(threshold {CALIBRATION_ECE_THRESHOLD:.2f})",
        "",
        "| Model | Split | Log loss | Brier | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, split_results in metrics.items():
        for split_name in ("validation", "test"):
            result = split_results[split_name]
            lines.append(
                f"| {model_name} | {split_name} | {result['log_loss']:.4f} | "
                f"{result['brier']:.4f} | {result['accuracy']:.4f} |"
            )
    lines.extend(
        [
            "",
            "The final model is selected without consulting test outcomes. Calibration is triggered "
            "only by validation-set ECE, then the untouched test partition is reported once.",
            "",
        ]
    )
    return "\n".join(lines)


def _importance_markdown(model: lgb.LGBMClassifier) -> str:
    gains = model.booster_.feature_importance(importance_type="gain")
    total = float(np.sum(gains))
    pairs = sorted(zip(FEATURE_COLUMNS, gains), key=lambda item: item[1], reverse=True)
    lines = [
        "# LightGBM feature importance",
        "",
        "Gain importance from the training-only LightGBM fit.",
        "",
        "| Rank | Feature | Gain | Share |",
        "|---:|---|---:|---:|",
    ]
    for rank, (feature, gain) in enumerate(pairs, start=1):
        share = gain / total if total else 0.0
        lines.append(f"| {rank} | `{feature}` | {gain:.2f} | {share:.2%} |")
    lines.append("")
    return "\n".join(lines)


def _save_calibration_plot(
    y_test: pd.Series,
    raw_probabilities: np.ndarray,
    final_probabilities: np.ndarray,
    calibrated: bool,
    run_stamp: str,
) -> None:
    _preserve_existing(CALIBRATION_PATH, run_stamp)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect calibration")
    raw_actual, raw_predicted = calibration_curve(
        y_test, raw_probabilities, n_bins=10, strategy="uniform"
    )
    ax.plot(raw_predicted, raw_actual, marker="o", label="LightGBM")
    if calibrated:
        final_actual, final_predicted = calibration_curve(
            y_test, final_probabilities, n_bins=10, strategy="uniform"
        )
        ax.plot(final_predicted, final_actual, marker="s", label="Isotonic LightGBM")
    ax.set(
        xlabel="Mean predicted fighter-A win probability",
        ylabel="Observed fighter-A win rate",
        title="Test-set calibration (10 bins)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CALIBRATION_PATH, dpi=160)
    plt.close(fig)


def train() -> dict:
    """Train, evaluate, report, and persist the Phase 5 model."""
    frame = load_features()
    train_frame, validation_frame, test_frame = chronological_split(frame)
    split_frames = {"validation": validation_frame, "test": test_frame}

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["label"].astype(int)
    x_validation = validation_frame[FEATURE_COLUMNS]
    y_validation = validation_frame["label"].astype(int)
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["label"].astype(int)

    # Training-only medians are persisted even though LightGBM consumes NaNs
    # directly; the Phase 6 artifact contract requires them.
    medians = x_train.median(axis=0).fillna(0.0)
    x_train_imputed = x_train.fillna(medians)
    x_validation_imputed = x_validation.fillna(medians)
    x_test_imputed = x_test.fillna(medians)

    logistic = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        ]
    )
    logistic.fit(x_train_imputed, y_train)

    lightgbm = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=2000,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lightgbm.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    metrics: dict[str, dict[str, dict[str, float]]] = {
        "Coin flip": {},
        "Elo only": {},
        "Logistic regression": {},
        "LightGBM": {},
    }
    raw_lgb_probabilities: dict[str, np.ndarray] = {}
    for split_name, split_frame in split_frames.items():
        labels = split_frame["label"].astype(int)
        features = split_frame[FEATURE_COLUMNS]
        imputed = features.fillna(medians)
        metrics["Coin flip"][split_name] = evaluate(labels, np.full(len(labels), 0.5))
        metrics["Elo only"][split_name] = evaluate(
            labels, elo_probabilities(split_frame["elo_diff"])
        )
        metrics["Logistic regression"][split_name] = evaluate(
            labels, logistic.predict_proba(imputed)[:, 1]
        )
        raw_lgb_probabilities[split_name] = lightgbm.predict_proba(features)[:, 1]
        metrics["LightGBM"][split_name] = evaluate(
            labels, raw_lgb_probabilities[split_name]
        )

    validation_ece = expected_calibration_error(
        y_validation, raw_lgb_probabilities["validation"]
    )
    calibrated = validation_ece > CALIBRATION_ECE_THRESHOLD
    final_model = lightgbm
    final_model_name = "lightgbm"
    if calibrated:
        final_model = CalibratedClassifierCV(
            FrozenEstimator(lightgbm), method="isotonic", n_jobs=-1
        )
        final_model.fit(x_validation, y_validation)
        final_model_name = "lightgbm_isotonic"
        metrics["Calibrated LightGBM"] = {}
        for split_name, split_frame in split_frames.items():
            probabilities = final_model.predict_proba(split_frame[FEATURE_COLUMNS])[:, 1]
            metrics["Calibrated LightGBM"][split_name] = evaluate(
                split_frame["label"].astype(int), probabilities
            )

    final_test_probabilities = final_model.predict_proba(x_test)[:, 1]
    now = datetime.now(timezone.utc)
    run_stamp = now.strftime("%Y%m%dT%H%M%SZ")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _preserve_existing(MODEL_PATH, run_stamp)
    artifact = {
        "model": final_model,
        "model_type": final_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "imputation_medians": {key: float(value) for key, value in medians.items()},
        "trained_at_utc": now.isoformat(),
        "training_cutoffs": {
            "train_before": TRAIN_END,
            "validation_before": VALIDATION_END,
        },
        "best_iteration": int(lightgbm.best_iteration_),
        "validation_ece": validation_ece,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)

    split_sizes = {
        "train": len(train_frame),
        "validation": len(validation_frame),
        "test": len(test_frame),
    }
    _write_text(
        METRICS_PATH,
        _metrics_markdown(
            metrics, split_sizes, final_model_name, validation_ece, calibrated
        ),
        run_stamp,
    )
    _write_text(IMPORTANCE_PATH, _importance_markdown(lightgbm), run_stamp)
    _save_calibration_plot(
        y_test,
        raw_lgb_probabilities["test"],
        final_test_probabilities,
        calibrated,
        run_stamp,
    )

    return {
        "model_path": MODEL_PATH,
        "metrics_path": METRICS_PATH,
        "calibration_path": CALIBRATION_PATH,
        "importance_path": IMPORTANCE_PATH,
        "model_type": final_model_name,
        "best_iteration": int(lightgbm.best_iteration_),
        "validation_ece": validation_ece,
        "split_sizes": split_sizes,
        "metrics": metrics,
    }


def main() -> None:
    result = train()
    print("=== ufcpred.train complete ===")
    print(
        "split rows    : "
        f"train={result['split_sizes']['train']}, "
        f"validation={result['split_sizes']['validation']}, "
        f"test={result['split_sizes']['test']}"
    )
    print(f"best iteration: {result['best_iteration']}")
    print(f"validation ECE: {result['validation_ece']:.4f}")
    print(f"final model   : {result['model_type']}")
    for model_name, split_results in result["metrics"].items():
        test = split_results["test"]
        print(
            f"{model_name:22s} test logloss={test['log_loss']:.4f} "
            f"brier={test['brier']:.4f} accuracy={test['accuracy']:.4f}"
        )
    print(f"model         : {result['model_path']}")
    print(f"metrics       : {result['metrics_path']}")
    print(f"calibration   : {result['calibration_path']}")
    print(f"importance    : {result['importance_path']}")


if __name__ == "__main__":
    main()
