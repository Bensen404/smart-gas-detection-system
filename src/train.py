"""Train and select the gas-state classifier.

Protocol
--------
1. Compare candidate classifiers with leave-one-session-out cross-validation on
   the 12 training sessions only.
2. Select by mean CV macro-F1; if candidates are within 0.001, prefer the
   smallest serialized model.
3. Refit the selected pipeline on the training sessions for validation metrics.
4. Refit once on training + validation sessions and freeze the artifact for the
   held-out test evaluation.  The test sessions are not read here.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, utils
else:
    from . import config, utils


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

FAMILY_MAP = {
    "DummyClassifier": DummyClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GaussianNB": GaussianNB,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
}


def build_estimator(definition: dict[str, Any]) -> Pipeline:
    family = definition["family"]
    if family not in FAMILY_MAP:
        raise ValueError(f"Unknown model family: {family}")
    estimator = FAMILY_MAP[family](**definition["params"])
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def ordered_proba(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    model_classes = list(model.named_steps["model"].classes_)
    indices = [model_classes.index(name) for name in config.CLASS_ORDER]
    return np.asarray(probabilities)[:, indices]


def fold_summary(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["accuracy", "macro_f1", "balanced_accuracy", "log_loss"]
    summary: dict[str, Any] = {"n_folds": len(metric_rows)}
    for key in keys:
        values = [row[key] for row in metric_rows if row.get(key) is not None]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return summary


def main() -> None:
    utils.ensure_directories()
    utils.set_global_seed()
    utils.verify_raw_dataset()

    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(config.PROCESSED_DIR / "validation.csv")

    print(
        f"[train] train rows={len(train)} sessions="
        f"{train[config.SESSION_COLUMN].nunique()}"
    )
    print(
        f"[train] validation rows={len(validation)} sessions="
        f"{validation[config.SESSION_COLUMN].nunique()}"
    )

    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    out_of_fold: dict[str, dict[str, np.ndarray]] = {}
    fitted_on_train: dict[str, Pipeline] = {}

    for definition in config.CANDIDATE_DEFINITIONS:
        name = definition["name"]
        features = definition["features"]
        X = train[features]
        y = train[config.TARGET_COLUMN]
        groups = train[config.SESSION_COLUMN]
        print(
            f"[train] CV: {name} ({definition['family']}, "
            f"features={features})"
        )

        logo = LeaveOneGroupOut()
        predictions = np.empty(len(train), dtype=object)
        probabilities = np.empty(
            (len(train), len(config.CLASS_ORDER)), dtype=float
        )
        fold_metrics: list[dict[str, Any]] = []
        start = time.perf_counter()

        for fold_number, (fit_index, test_index) in enumerate(
            logo.split(X, y, groups), start=1
        ):
            model = build_estimator(definition)
            model.fit(X.iloc[fit_index], y.iloc[fit_index])
            fold_predictions = model.predict(X.iloc[test_index])
            fold_probabilities = ordered_proba(model, X.iloc[test_index])
            predictions[test_index] = fold_predictions
            probabilities[test_index, :] = fold_probabilities

            metrics = utils.evaluate_predictions(
                y.iloc[test_index], fold_predictions, fold_probabilities
            )
            fold_metrics.append(metrics)
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": fold_number,
                    "held_out_session": sorted(
                        groups.iloc[test_index].unique()
                    )[0],
                    "n_test": len(test_index),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "log_loss": metrics.get("log_loss"),
                }
            )

        cv_seconds = time.perf_counter() - start
        oof_metrics = utils.evaluate_predictions(y, predictions, probabilities)
        summary = fold_summary(fold_metrics)

        # Confirmation fit on all training sessions, scored on validation.
        model_train = build_estimator(definition)
        fit_start = time.perf_counter()
        model_train.fit(X, y)
        fit_seconds = time.perf_counter() - fit_start
        validation_probabilities = ordered_proba(
            model_train, validation[features]
        )
        validation_predictions = model_train.predict(validation[features])
        validation_metrics = utils.evaluate_predictions(
            validation[config.TARGET_COLUMN],
            validation_predictions,
            validation_probabilities,
        )

        fitted_on_train[name] = model_train
        out_of_fold[name] = {
            "pred": predictions,
            "proba": probabilities,
        }

        # Measure the compressed artifact actually written to disk.  This is
        # the size used by the pre-registered tie-break rule.
        temporary_path = config.MODELS_DIR / f"_tmp_{name}.joblib"
        joblib.dump(model_train, temporary_path, compress=3)
        serialized_bytes = int(temporary_path.stat().st_size)
        temporary_path.unlink(missing_ok=True)

        candidate_rows.append(
            {
                "candidate": name,
                "family": definition["family"],
                "features": "+".join(features),
                "n_features": len(features),
                "params": definition["params"],
                "embedded_suitability": definition["embedded_suitability"],
                "cv_accuracy_mean": summary["accuracy"]["mean"],
                "cv_accuracy_std": summary["accuracy"]["std"],
                "cv_macro_f1_mean": summary["macro_f1"]["mean"],
                "cv_macro_f1_std": summary["macro_f1"]["std"],
                "cv_balanced_accuracy_mean": summary["balanced_accuracy"]["mean"],
                "cv_log_loss_mean": summary["log_loss"]["mean"],
                "oof_accuracy": oof_metrics["accuracy"],
                "oof_macro_f1": oof_metrics["macro_f1"],
                "oof_log_loss": oof_metrics.get("log_loss"),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_log_loss": validation_metrics.get("log_loss"),
                "validation_danger_recall": validation_metrics["per_class"][
                    "Danger"
                ]["recall"],
                "serialized_bytes": serialized_bytes,
                "cv_seconds": cv_seconds,
                "full_train_fit_seconds": fit_seconds,
            }
        )
        print(
            "         CV macro-F1 "
            f"{summary['macro_f1']['mean']:.4f} +/- "
            f"{summary['macro_f1']['std']:.4f} | "
            f"validation macro-F1 {validation_metrics['macro_f1']:.4f} | "
            f"{serialized_bytes:,} bytes"
        )

    comparison = pd.DataFrame(candidate_rows).sort_values(
        ["cv_macro_f1_mean", "serialized_bytes"], ascending=[False, True]
    )
    comparison = comparison.reset_index(drop=True)
    comparison.to_csv(
        config.METRICS_DIR / "model_comparison.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        config.METRICS_DIR / "cv_fold_metrics.csv", index=False
    )

    best_score = float(comparison["cv_macro_f1_mean"].max())
    tied = comparison[
        comparison["cv_macro_f1_mean"] >= best_score - config.CV_TIE_TOLERANCE
    ].copy()
    tied = tied.sort_values(
        ["serialized_bytes", "cv_macro_f1_mean"], ascending=[True, False]
    )
    selected_name = str(tied.iloc[0]["candidate"])
    selected_row = comparison[comparison["candidate"] == selected_name].iloc[0]
    selected_definition = next(
        definition
        for definition in config.CANDIDATE_DEFINITIONS
        if definition["name"] == selected_name
    )
    print(
        f"[train] selected: {selected_name} "
        f"(CV macro-F1={selected_row['cv_macro_f1_mean']:.4f})"
    )

    selected_train_model = fitted_on_train[selected_name]
    joblib.dump(
        selected_train_model,
        config.MODELS_DIR / "selected_model_train.joblib",
        compress=3,
    )

    train_validation = pd.concat([train, validation], ignore_index=True)
    final_features = selected_definition["features"]
    final_model = build_estimator(selected_definition)
    fit_start = time.perf_counter()
    final_model.fit(
        train_validation[final_features],
        train_validation[config.TARGET_COLUMN],
    )
    final_fit_seconds = time.perf_counter() - fit_start
    joblib.dump(
        final_model, config.MODELS_DIR / "final_model.joblib", compress=3
    )

    oof = train.copy()
    oof["predicted_label"] = out_of_fold[selected_name]["pred"]
    for index, name in enumerate(config.CLASS_ORDER):
        oof[f"prob_{name}"] = out_of_fold[selected_name]["proba"][:, index]
    oof["correct"] = (
        oof["predicted_label"] == oof[config.TARGET_COLUMN]
    )
    oof.to_csv(
        config.PROCESSED_DIR / "train_oof_predictions.csv", index=False
    )

    selected_metrics = {
        "selection_rule": {
            "primary": (
                "highest mean leave-one-session-out CV macro-F1 on the "
                "training sessions"
            ),
            "tie_break": (
                f"if within {config.CV_TIE_TOLERANCE}, choose the smallest "
                "serialized model"
            ),
            "test_set_used_for_selection": False,
        },
        "selected_candidate": selected_name,
        "family": selected_definition["family"],
        "features": final_features,
        "feature_order": final_features,
        "params": selected_definition["params"],
        "preprocessing": {
            "type": "StandardScaler",
            "fit_on": "training partition (inside every CV fold and for the final fit)",
        },
        "cv": {
            "mean_macro_f1": float(selected_row["cv_macro_f1_mean"]),
            "std_macro_f1": float(selected_row["cv_macro_f1_std"]),
            "mean_accuracy": float(selected_row["cv_accuracy_mean"]),
            "std_accuracy": float(selected_row["cv_accuracy_std"]),
            "mean_log_loss": float(selected_row["cv_log_loss_mean"]),
            "oof_macro_f1": float(selected_row["oof_macro_f1"]),
            "oof_accuracy": float(selected_row["oof_accuracy"]),
        },
        "validation": {
            "accuracy": float(selected_row["validation_accuracy"]),
            "macro_f1": float(selected_row["validation_macro_f1"]),
            "balanced_accuracy": float(
                selected_row["validation_balanced_accuracy"]
            ),
            "log_loss": float(selected_row["validation_log_loss"]),
            "danger_recall": float(selected_row["validation_danger_recall"]),
            "fit_on": "12 training sessions only",
        },
        "final_fit": {
            "fit_on": (
                "12 training sessions + 3 validation sessions "
                "(15 sessions total)"
            ),
            "n_rows": int(len(train_validation)),
            "fit_seconds": final_fit_seconds,
            "artifact": "models/final_model.joblib",
        },
        "serialized_bytes_train_model": int(
            selected_row["serialized_bytes"]
        ),
        "n_train_rows": int(len(train)),
        "n_validation_rows": int(len(validation)),
        "class_order": config.CLASS_ORDER,
        "random_seed": config.RANDOM_SEED,
    }
    utils.write_json(config.MODELS_DIR / "selected_model.json", selected_metrics)
    utils.write_json(
        config.MODELS_DIR / "candidate_configs.json",
        {
            "random_seed": config.RANDOM_SEED,
            "cv": "LeaveOneGroupOut on session_id (12 folds)",
            "candidates": config.CANDIDATE_DEFINITIONS,
        },
    )

    table_rows = []
    for row in comparison.itertuples():
        marker = " **selected**" if row.candidate == selected_name else ""
        table_rows.append(
            [
                row.candidate + marker,
                row.features,
                f"{row.cv_macro_f1_mean:.4f} +/- {row.cv_macro_f1_std:.4f}",
                f"{row.validation_macro_f1:.4f}",
                f"{row.validation_accuracy:.4f}",
                f"{row.serialized_bytes:,}",
                row.embedded_suitability,
            ]
        )

    report = f"""# Model Selection

## Protocol

- Training sessions: {len(config.EXPECTED_SPLIT['train'])} (`{', '.join(config.EXPECTED_SPLIT['train'])}`)
- Validation sessions: {len(config.EXPECTED_SPLIT['validation'])} (`{', '.join(config.EXPECTED_SPLIT['validation'])}`)
- Test sessions: {len(config.EXPECTED_SPLIT['test'])} (`{', '.join(config.EXPECTED_SPLIT['test'])}`) — **not read by this script**
- Cross-validation: leave-one-session-out, 12 folds on the training sessions
- Primary criterion: mean fold macro-F1
- Tie-break (within {config.CV_TIE_TOLERANCE}): smaller serialized model
- Random seed: {config.RANDOM_SEED}
- Preprocessing: `StandardScaler`, fitted inside each CV fold and on each fit partition

## Candidate comparison

{utils.markdown_table(
    [
        "Candidate",
        "Features",
        "CV macro-F1 (mean +/- std)",
        "Val macro-F1",
        "Val accuracy",
        "Joblib bytes",
        "Embedded suitability",
    ],
    table_rows,
)}

## Selected model

**{selected_name}** was selected with a mean CV macro-F1 of
{selected_row['cv_macro_f1_mean']:.4f} (standard deviation
{selected_row['cv_macro_f1_std']:.4f}) and a validation macro-F1 of
{selected_row['validation_macro_f1']:.4f}.

- Family: `{selected_definition['family']}`
- Input features: `{', '.join(final_features)}`
- Parameters: `{selected_definition['params']}`
- Final fit: {len(train_validation):,} rows from 15 train + validation sessions
- Artifact: `models/final_model.joblib`

The all-feature logistic regression was evaluated under the same grouped CV.
Temperature and humidity did not improve CV macro-F1 in this dataset, so the
ADC-only variant was retained. This removes the DHT sensor from the inference
dependency chain and reduces the embedded export to one input.

## Test isolation

The held-out test sessions were not used to choose the model, tune
hyper-parameters, select features or learn thresholds. They are evaluated once
in `src/evaluate.py`.
"""
    utils.write_text(config.METRICS_DIR / "model_selection.md", report)

    print(
        "[train] wrote models/final_model.joblib, "
        "models/selected_model.json and results/metrics/*"
    )


if __name__ == "__main__":
    main()
