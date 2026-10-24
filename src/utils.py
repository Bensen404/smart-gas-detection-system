"""Shared helpers for integrity checks, metrics, JSON output and plotting."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from . import config


# ---------------------------------------------------------------------------
# Files and integrity
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (
        config.DATA_DIR,
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.SAMPLE_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR,
        config.METRICS_DIR,
        config.PLOTS_DIR,
        config.FIRMWARE_DIR,
        config.APP_DIR,
        config.DOCS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_raw_dataset(path: Path | None = None) -> str:
    """Verify that the immutable raw dataset still matches its checksum."""
    path = path or config.DATASET_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}. "
            "Place the supplied CSV in data/raw/ before running the pipeline."
        )
    actual = sha256_file(path)
    if actual != config.EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Raw dataset integrity check failed. "
            f"Expected {config.EXPECTED_DATASET_SHA256}, found {actual}. "
            "The original CSV must not be modified."
        )
    return actual


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# JSON and text
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def environment_info() -> dict[str, Any]:
    import joblib
    import matplotlib
    import scipy
    import seaborn
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "joblib": joblib.__version__,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion(y_true: Iterable[str], y_pred: Iterable[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=config.CLASS_ORDER)


def multiclass_brier(y_true: Iterable[str], y_proba: np.ndarray) -> float:
    labels = np.asarray(list(y_true), dtype=object)
    one_hot = np.column_stack(
        [(labels == name).astype(float) for name in config.CLASS_ORDER]
    )
    return float(np.mean(np.sum((np.asarray(y_proba) - one_hot) ** 2, axis=1)))


def evaluate_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the evaluation metrics used throughout the project.

    ``y_proba`` must have columns in ``config.CLASS_ORDER``.  The helper
    reorders both probabilities and labels explicitly before computing log loss,
    because scikit-learn orders labels lexicographically internally.
    """
    classes = list(config.CLASS_ORDER)
    y_true = np.asarray(list(y_true), dtype=object)
    y_pred = np.asarray(list(y_pred), dtype=object)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(classes)
    }

    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_recall_fscore_support(
                y_true, y_pred, labels=classes, average="macro", zero_division=0
            )[0]
        ),
        "macro_recall": float(
            precision_recall_fscore_support(
                y_true, y_pred, labels=classes, average="macro", zero_division=0
            )[1]
        ),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro")),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=classes, average="weighted")
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=classes)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion(y_true, y_pred).tolist(),
        "labels": classes,
        "n_errors": int(np.sum(y_true != y_pred)),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim != 2 or y_proba.shape[1] != len(classes):
            raise ValueError(
                f"Expected probabilities with {len(classes)} columns, got {y_proba.shape}"
            )
        if not np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Probability rows must sum to 1")

        lexicographic = sorted(classes)
        reordered = y_proba[:, [classes.index(name) for name in lexicographic]]
        result["log_loss"] = float(
            log_loss(y_true, reordered, labels=lexicographic)
        )
        result["multiclass_brier"] = multiclass_brier(y_true, y_proba)

        top2 = np.argsort(y_proba, axis=1)[:, -2:]
        result["top2_accuracy"] = float(
            np.mean(
                [
                    y_true[index] in {classes[j] for j in top2[index]}
                    for index in range(len(y_true))
                ]
            )
        )

    return result


def classification_text(
    y_true: Iterable[str], y_pred: Iterable[str], title: str
) -> str:
    report = classification_report(
        y_true,
        y_pred,
        labels=config.CLASS_ORDER,
        target_names=config.CLASS_ORDER,
        digits=4,
        zero_division=0,
    )
    matrix = confusion(y_true, y_pred)
    lines = [
        "=" * 72,
        title,
        "=" * 72,
        "",
        report,
        "Confusion matrix (rows = true, columns = predicted)",
        f"{'':>12}" + "".join(f"{name:>12}" for name in config.CLASS_ORDER),
    ]
    for name, row in zip(config.CLASS_ORDER, matrix):
        lines.append(f"{name:>12}" + "".join(f"{int(value):>12}" for value in row))
    lines.append("")
    return "\n".join(lines)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def grouped_bootstrap_metrics(
    df: pd.DataFrame,
    prediction_column: str,
    n_boot: int = config.BOOTSTRAP_ITERATIONS,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Bootstrap whole sessions to respect within-session correlation."""
    rng = np.random.default_rng(seed)
    sessions = np.array(sorted(df[config.SESSION_COLUMN].unique()))
    if len(sessions) == 0:
        return {}
    rows = {
        session: df[df[config.SESSION_COLUMN] == session] for session in sessions
    }

    accuracies: list[float] = []
    macro_f1_scores: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(sessions, size=len(sessions), replace=True)
        sample = pd.concat([rows[session] for session in picked], ignore_index=True)
        accuracies.append(
            accuracy_score(
                sample[config.TARGET_COLUMN], sample[prediction_column]
            )
        )
        macro_f1_scores.append(
            f1_score(
                sample[config.TARGET_COLUMN],
                sample[prediction_column],
                labels=config.CLASS_ORDER,
                average="macro",
            )
        )

    return {
        "method": "session-group bootstrap (resampling whole sessions)",
        "n_boot": n_boot,
        "n_sessions": int(len(sessions)),
        "accuracy": {
            "ci95_low": float(np.percentile(accuracies, 2.5)),
            "ci95_high": float(np.percentile(accuracies, 97.5)),
        },
        "macro_f1": {
            "ci95_low": float(np.percentile(macro_f1_scores, 2.5)),
            "ci95_high": float(np.percentile(macro_f1_scores, 97.5)),
        },
    }
