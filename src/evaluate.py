"""Evaluate the frozen classifier on validation and held-out test sessions.

The test sessions were not used for model selection, feature selection,
hyper-parameter tuning or threshold selection.  This script evaluates them once
and writes the metrics, tables and plots used in the README.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, utils
else:
    from . import config, utils


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def ordered_proba(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    model_classes = list(model.named_steps["model"].classes_)
    indices = [model_classes.index(name) for name in config.CLASS_ORDER]
    return np.asarray(probabilities)[:, indices]


def add_predictions(
    df: pd.DataFrame,
    model,
    features: list[str],
    prefix: str = "",
) -> pd.DataFrame:
    output = df.copy()
    output[f"{prefix}predicted_label"] = model.predict(output[features])
    probabilities = ordered_proba(model, output[features])
    for index, name in enumerate(config.CLASS_ORDER):
        output[f"{prefix}prob_{name}"] = probabilities[:, index]
    output[f"{prefix}confidence"] = probabilities.max(axis=1)
    output[f"{prefix}correct"] = (
        output[f"{prefix}predicted_label"] == output[config.TARGET_COLUMN]
    )
    return output


def per_session_metrics(
    df: pd.DataFrame, prediction_column: str = "predicted_label"
) -> pd.DataFrame:
    rows = []
    for session, group in df.groupby(config.SESSION_COLUMN):
        metrics = utils.evaluate_predictions(
            group[config.TARGET_COLUMN], group[prediction_column], None
        )
        rows.append(
            {
                "session_id": session,
                "n": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "danger_recall": metrics["per_class"]["Danger"]["recall"],
                "warning_recall": metrics["per_class"]["Warning"]["recall"],
                "normal_recall": metrics["per_class"]["Normal"]["recall"],
                "errors": metrics["n_errors"],
                "danger_to_normal": int(
                    (
                        (group[config.TARGET_COLUMN] == "Danger")
                        & (group[prediction_column] == "Normal")
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)


def critical_errors(
    df: pd.DataFrame, prediction_column: str = "predicted_label"
) -> pd.DataFrame:
    errors = df[df[config.TARGET_COLUMN] != df[prediction_column]].copy()
    if errors.empty:
        return pd.DataFrame(
            columns=[
                config.SESSION_COLUMN,
                "mq5_adc",
                "true_label",
                "predicted_label",
                "confidence",
                "error_type",
                "severity",
            ]
        )
    errors["error_type"] = (
        errors[config.TARGET_COLUMN] + " -> " + errors[prediction_column]
    )
    severity = {
        "Danger -> Normal": "critical miss",
        "Danger -> Warning": "under-estimated danger",
        "Normal -> Danger": "false alarm",
        "Warning -> Normal": "missed warning",
        "Warning -> Danger": "over-estimated warning",
        "Normal -> Warning": "nuisance alert",
    }
    errors["severity"] = errors["error_type"].map(severity).fillna("other")
    columns = [
        config.SESSION_COLUMN,
        "mq5_adc",
        config.TARGET_COLUMN,
        prediction_column,
        "confidence",
        "error_type",
        "severity",
    ]
    return errors[columns].sort_values(["severity", "mq5_adc"])


def threshold_predict(values: np.ndarray, normal_max: float, warning_max: float):
    return np.where(
        values <= normal_max,
        "Normal",
        np.where(values <= warning_max, "Warning", "Danger"),
    )


def optimise_thresholds(train: pd.DataFrame) -> tuple[float, float, float]:
    """Learn two ordered ADC cut-points on the training split only."""
    values = train["mq5_adc"].to_numpy()
    labels = train[config.TARGET_COLUMN].to_numpy()
    best = (-1.0, 0.0, 0.0)
    step = config.BASELINE_THRESHOLD_STEP
    for normal_max in np.arange(600, 2001, step):
        for warning_max in np.arange(max(normal_max + step, 1400), 3201, step):
            prediction = threshold_predict(values, normal_max, warning_max)
            score = float(
                f1_score(
                    labels,
                    prediction,
                    labels=config.CLASS_ORDER,
                    average="macro",
                )
            )
            if score > best[0]:
                best = (score, float(normal_max), float(warning_max))
    return best[1], best[2], best[0]


def wilson(successes: int, n: int) -> dict[str, float]:
    low, high = utils.wilson_interval(successes, n)
    return {
        "estimate": successes / n if n else float("nan"),
        "ci95_low": low,
        "ci95_high": high,
    }


def mcnemar_exact(
    model_correct: np.ndarray, baseline_correct: np.ndarray
) -> dict[str, Any]:
    model_only = int(np.sum(model_correct & ~baseline_correct))
    baseline_only = int(np.sum(~model_correct & baseline_correct))
    discordant = model_only + baseline_only
    p_value = (
        1.0
        if discordant == 0
        else float(
            stats.binomtest(
                min(model_only, baseline_only), discordant, 0.5
            ).pvalue
            * 2
        )
    )
    return {
        "model_only_correct": model_only,
        "baseline_only_correct": baseline_only,
        "discordant_pairs": discordant,
        "exact_binomial_p": min(1.0, p_value),
    }


def calibration_scores(test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    confidence = test["confidence"].to_numpy()
    correct = test["correct"].to_numpy().astype(float)
    bins = np.linspace(0, 1, 11)
    bin_ids = np.clip(np.digitize(confidence, bins) - 1, 0, 9)
    rows = []
    for index in range(10):
        mask = bin_ids == index
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_lower": float(bins[index]),
                "bin_upper": float(bins[index + 1]),
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "gap": float(
                    abs(correct[mask].mean() - confidence[mask].mean())
                ),
            }
        )
    table = pd.DataFrame(rows)
    ece = float(np.sum(table["n"] / len(test) * table["gap"]))
    mce = float(table["gap"].max()) if len(table) else float("nan")
    return table, {"ece": ece, "mce": mce}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": config.FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    path = config.PLOTS_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame) -> Path:
    counts = [
        int((df[config.TARGET_COLUMN] == name).sum())
        for name in config.CLASS_ORDER
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(
        config.CLASS_ORDER,
        counts,
        color=[config.CLASS_COLORS[name] for name in config.CLASS_ORDER],
        width=0.62,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 25,
            f"{count:,}\n({count / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Rows")
    ax.set_title("Class distribution (full dataset)")
    ax.set_ylim(0, max(counts) * 1.22)
    fig.tight_layout()
    return _save(fig, "class_distribution.png")


def plot_adc_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bins = np.arange(0, 4100, 100)
    for name in config.CLASS_ORDER:
        subset = df[df[config.TARGET_COLUMN] == name]
        ax.hist(
            subset["mq5_adc"],
            bins=bins,
            alpha=0.45,
            label=name,
            color=config.CLASS_COLORS[name],
            edgecolor="none",
        )
    ax.set_xlabel("MQ-5 ADC value (12-bit)")
    ax.set_ylabel("Rows")
    ax.set_title("ADC distribution by class")
    ax.set_xlim(0, 4095)
    ax.legend(title="Label")
    fig.tight_layout()
    return _save(fig, "adc_distribution_by_class.png")


def plot_model_comparison(
    comparison: pd.DataFrame, selected_name: str
) -> Path:
    data = comparison.sort_values("cv_macro_f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    y = np.arange(len(data))
    colors = [
        "#C0392B" if name == selected_name else "#7F8C8D"
        for name in data["candidate"]
    ]
    ax.barh(
        y,
        data["cv_macro_f1_mean"],
        height=0.5,
        color=colors,
        label="Grouped CV macro-F1 (mean ± std)",
    )
    ax.errorbar(
        data["cv_macro_f1_mean"],
        y,
        xerr=data["cv_macro_f1_std"],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=2.5,
    )
    ax.scatter(
        data["validation_macro_f1"],
        y,
        marker="D",
        s=34,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Validation macro-F1",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(data["candidate"])
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlabel("Macro-F1")
    ax.set_title(
        "Candidate comparison (test sessions not used)\n"
        f"Selected: {selected_name} (smallest model within CV tie tolerance)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    return _save(fig, "model_comparison.png")


def plot_confusion_matrices(
    cv_df: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> Path:
    panels = [
        (
            "Train CV (out-of-fold)",
            cv_df[config.TARGET_COLUMN],
            cv_df["predicted_label"],
        ),
        (
            "Validation sessions",
            validation[config.TARGET_COLUMN],
            validation["predicted_label"],
        ),
        (
            "Held-out test sessions",
            test[config.TARGET_COLUMN],
            test["predicted_label"],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (title, y_true, y_pred) in zip(axes, panels):
        matrix = confusion_matrix(y_true, y_pred, labels=config.CLASS_ORDER)
        row_sums = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        ) * 100
        annotations = np.empty_like(matrix, dtype=object)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                annotations[row, column] = (
                    f"{matrix[row, column]}\n({percentages[row, column]:.1f}%)"
                )
        sns.heatmap(
            matrix,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=0.7,
            linecolor="white",
            xticklabels=config.CLASS_ORDER,
            yticklabels=config.CLASS_ORDER,
            ax=ax,
        )
        accuracy = np.trace(matrix) / matrix.sum()
        ax.set_title(f"{title}\naccuracy={accuracy:.4f}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    fig.suptitle(
        "Confusion matrices: grouped CV, validation and held-out test",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "confusion_matrices.png")


def plot_probability_curve(model, features: list[str], test: pd.DataFrame) -> Path:
    grid = pd.DataFrame({"mq5_adc": np.arange(0, 4096, 2)})
    probabilities = ordered_proba(model, grid[features])
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7.2),
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for index, name in enumerate(config.CLASS_ORDER):
        axes[0].plot(
            grid["mq5_adc"],
            probabilities[:, index],
            color=config.CLASS_COLORS[name],
            lw=2.2,
            label=name,
        )
    for index, name in enumerate(config.CLASS_ORDER):
        subset = test[test[config.TARGET_COLUMN] == name]
        axes[0].scatter(
            subset["mq5_adc"],
            np.full(len(subset), -0.055 + 0.018 * index),
            s=26,
            color=config.CLASS_COLORS[name],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
            label=f"test {name}",
        )
    axes[0].set_xlim(0, 4095)
    axes[0].set_ylim(-0.08, 1.05)
    axes[0].set_ylabel("Predicted probability")
    axes[0].set_xlabel("MQ-5 ADC value")
    axes[0].set_title(
        "Frozen model response over the ADC range, with held-out test readings"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(
        unique.values(), unique.keys(), ncol=3, fontsize=8, loc="center right"
    )

    sns.histplot(
        data=test,
        x="mq5_adc",
        hue=config.TARGET_COLUMN,
        hue_order=config.CLASS_ORDER,
        palette={name: config.CLASS_COLORS[name] for name in config.CLASS_ORDER},
        bins=np.arange(0, 4100, 100),
        element="step",
        fill=True,
        alpha=0.25,
        ax=axes[1],
    )
    axes[1].set_xlim(0, 4095)
    axes[1].set_xlabel("MQ-5 ADC value")
    axes[1].set_ylabel("Test rows")
    axes[1].set_title("Held-out ADC coverage")
    fig.tight_layout()
    return _save(fig, "probability_vs_adc.png")


def plot_per_session(session_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(session_df))
    width = 0.26
    ax.bar(
        x - width,
        session_df["accuracy"],
        width,
        label="Accuracy",
        color="#4C78A8",
    )
    ax.bar(
        x,
        session_df["macro_f1"],
        width,
        label="Macro-F1",
        color="#59A14F",
    )
    ax.bar(
        x + width,
        session_df["danger_recall"],
        width,
        label="Danger recall",
        color="#C0392B",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(session_df["session_id"])
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Held-out performance by session")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    for index, row in session_df.reset_index(drop=True).iterrows():
        ax.text(
            index - width,
            row["accuracy"] + 0.004,
            f"{row['accuracy']:.3f}",
            ha="center",
            fontsize=7,
            rotation=90,
        )
    fig.tight_layout()
    return _save(fig, "per_session_test.png")


def plot_calibration(
    test: pd.DataFrame, calibration: pd.DataFrame, scores: dict[str, float]
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    ax.plot([0, 1], [0, 1], ls=":", color="grey", label="Perfect calibration")
    ax.plot(
        calibration["mean_confidence"],
        calibration["accuracy"],
        "o-",
        color="#4C78A8",
        lw=2,
        label="Model",
    )
    ax.bar(
        calibration["mean_confidence"],
        calibration["n"] / len(test),
        width=0.075,
        alpha=0.18,
        color="#4C78A8",
        label="Share of rows",
    )
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(
        "Top-label reliability (test)\n"
        f"ECE={scores['ece']:.4f}, MCE={scores['mce']:.4f}"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "calibration_test.png")


def main() -> None:
    utils.ensure_directories()
    utils.set_global_seed()
    _style()
    utils.verify_raw_dataset()

    selected = utils.read_json(config.MODELS_DIR / "selected_model.json")
    comparison = pd.read_csv(config.METRICS_DIR / "model_comparison.csv")
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(config.PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    cv_df = pd.read_csv(config.PROCESSED_DIR / "train_oof_predictions.csv")
    full_data = pd.read_csv(config.DATASET_PATH)

    features = selected["features"]
    final_model = joblib.load(config.MODELS_DIR / "final_model.joblib")
    train_model = joblib.load(
        config.MODELS_DIR / "selected_model_train.joblib"
    )

    print(f"[evaluate] selected model: {selected['selected_candidate']}")
    print(f"[evaluate] features: {features}")
    print(
        f"[evaluate] test rows: {len(test)} from sessions "
        f"{sorted(test['session_id'].unique())}"
    )

    validation = add_predictions(validation, train_model, features)
    test = add_predictions(test, final_model, features)

    cv_metrics = utils.evaluate_predictions(
        cv_df[config.TARGET_COLUMN],
        cv_df["predicted_label"],
        cv_df[[f"prob_{name}" for name in config.CLASS_ORDER]].to_numpy(),
    )
    validation_metrics = utils.evaluate_predictions(
        validation[config.TARGET_COLUMN],
        validation["predicted_label"],
        validation[[f"prob_{name}" for name in config.CLASS_ORDER]].to_numpy(),
    )
    test_metrics = utils.evaluate_predictions(
        test[config.TARGET_COLUMN],
        test["predicted_label"],
        test[[f"prob_{name}" for name in config.CLASS_ORDER]].to_numpy(),
    )

    n_test = len(test)
    n_correct = int(test["correct"].sum())
    test_metrics["accuracy_wilson_ci95"] = wilson(n_correct, n_test)
    validation_metrics["accuracy_wilson_ci95"] = wilson(
        int(validation["correct"].sum()), len(validation)
    )
    test_metrics["grouped_bootstrap"] = utils.grouped_bootstrap_metrics(
        test, "predicted_label"
    )

    session_test = per_session_metrics(test)
    session_test.to_csv(
        config.METRICS_DIR / "per_session_test_metrics.csv", index=False
    )
    errors = critical_errors(test)
    errors.to_csv(config.METRICS_DIR / "critical_errors_test.csv", index=False)

    prediction_columns = [
        config.SESSION_COLUMN,
        "mq5_adc",
        "temperature_c",
        "humidity_rh_pct",
        config.TARGET_COLUMN,
        "predicted_label",
        *[f"prob_{name}" for name in config.CLASS_ORDER],
        "confidence",
        "correct",
    ]
    test[prediction_columns].to_csv(
        config.PROCESSED_DIR / "test_predictions.csv", index=False
    )
    validation[prediction_columns].to_csv(
        config.PROCESSED_DIR / "validation_predictions.csv", index=False
    )

    # Engineering baseline: two ADC thresholds learned on training data only.
    normal_max, warning_max, train_rule_f1 = optimise_thresholds(train)
    test["rule_prediction"] = threshold_predict(
        test["mq5_adc"].to_numpy(), normal_max, warning_max
    )
    validation_rule = threshold_predict(
        validation["mq5_adc"].to_numpy(), normal_max, warning_max
    )
    test_rule_metrics = utils.evaluate_predictions(
        test[config.TARGET_COLUMN], test["rule_prediction"]
    )
    validation_rule_metrics = utils.evaluate_predictions(
        validation[config.TARGET_COLUMN], validation_rule
    )
    baseline = {
        "method": "two ADC thresholds learned on the training split",
        "thresholds": {
            "normal_max": normal_max,
            "warning_max": warning_max,
        },
        "train_macro_f1": train_rule_f1,
        "validation": validation_rule_metrics,
        "test": test_rule_metrics,
        "mcnemar_test_vs_rule": mcnemar_exact(
            test["correct"].to_numpy(),
            (
                test["rule_prediction"] == test[config.TARGET_COLUMN]
            ).to_numpy(),
        ),
    }
    utils.write_json(
        config.METRICS_DIR / "baseline_comparison.json", baseline
    )

    # Plots.
    plot_class_distribution(full_data)
    plot_adc_distribution(full_data)
    plot_model_comparison(comparison, selected["selected_candidate"])
    plot_confusion_matrices(cv_df, validation, test)
    plot_probability_curve(final_model, features, test)
    plot_per_session(session_test)
    calibration, calibration_summary = calibration_scores(test)
    calibration.to_csv(
        config.METRICS_DIR / "calibration_bins_test.csv", index=False
    )
    plot_calibration(test, calibration, calibration_summary)
    test_metrics["calibration"] = calibration_summary

    # Confusion matrices as CSV.
    for name, frame in [
        ("train_cv", cv_df),
        ("validation", validation),
        ("test", test),
    ]:
        matrix = confusion_matrix(
            frame[config.TARGET_COLUMN],
            frame["predicted_label"],
            labels=config.CLASS_ORDER,
        )
        pd.DataFrame(
            matrix, index=config.CLASS_ORDER, columns=config.CLASS_ORDER
        ).to_csv(config.METRICS_DIR / f"confusion_matrix_{name}.csv")

    utils.write_text(
        config.METRICS_DIR / "classification_report_validation.txt",
        utils.classification_text(
            validation[config.TARGET_COLUMN],
            validation["predicted_label"],
            "SMART GAS AI - VALIDATION SESSIONS "
            "(model fitted on 12 training sessions)",
        ),
    )
    utils.write_text(
        config.METRICS_DIR / "classification_report_test.txt",
        utils.classification_text(
            test[config.TARGET_COLUMN],
            test["predicted_label"],
            "SMART GAS AI - HELD-OUT TEST SESSIONS "
            "(frozen model fitted on 15 train+validation sessions)",
        ),
    )

    metrics_payload = {
        "selected_model": selected["selected_candidate"],
        "family": selected["family"],
        "features": features,
        "preprocessing": selected["preprocessing"],
        "random_seed": config.RANDOM_SEED,
        "dataset": {
            "path": utils.relative_to_root(config.DATASET_PATH),
            "sha256": selected.get("source_sha256")
            or utils.read_json(config.PROCESSED_DIR / "metadata.json")[
                "source_sha256"
            ],
            "rows": int(len(full_data)),
        },
        "split_sessions": {
            name: sorted(part[config.SESSION_COLUMN].unique().tolist())
            for name, part in {
                "train": train,
                "validation": validation,
                "test": test,
            }.items()
        },
        "cv": {
            "overall_oof": cv_metrics,
            "reported_selection_score": selected["cv"],
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "baseline": baseline,
        "runtime": utils.environment_info(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    utils.write_json(config.METRICS_DIR / "metrics.json", metrics_payload)

    per_class_rows = [
        [
            name,
            f"{test_metrics['per_class'][name]['precision']:.4f}",
            f"{test_metrics['per_class'][name]['recall']:.4f}",
            f"{test_metrics['per_class'][name]['f1']:.4f}",
            test_metrics["per_class"][name]["support"],
        ]
        for name in config.CLASS_ORDER
    ]
    session_rows = [
        [
            row.session_id,
            row.n,
            f"{row.accuracy:.4f}",
            f"{row.macro_f1:.4f}",
            f"{row.danger_recall:.4f}",
            row.errors,
            row.danger_to_normal,
        ]
        for row in session_test.itertuples()
    ]
    matrix = test_metrics["confusion_matrix"]
    accuracy_ci = test_metrics["accuracy_wilson_ci95"]
    bootstrap = test_metrics["grouped_bootstrap"]

    report = f"""# Evaluation Summary

## Frozen model

| Item | Value |
|---|---|
| Selected candidate | `{selected['selected_candidate']}` |
| Family | `{selected['family']}` |
| Input features | `{', '.join(features)}` |
| Preprocessing | StandardScaler fitted on the training partition |
| Training rows (validation fit) | {len(train):,} |
| Final-fit rows (test fit) | {len(train) + len(validation):,} from 15 sessions |
| Test rows | {len(test):,} from {test['session_id'].nunique()} sessions |
| Test sessions | {', '.join(sorted(test['session_id'].unique()))} |
| Random seed | {config.RANDOM_SEED} |
| Test used for tuning | No |

## Held-out test metrics

| Metric | Value |
|---|---:|
| Accuracy | {test_metrics['accuracy']:.4f} |
| Accuracy 95% Wilson CI | [{accuracy_ci['ci95_low']:.4f}, {accuracy_ci['ci95_high']:.4f}] |
| Balanced accuracy | {test_metrics['balanced_accuracy']:.4f} |
| Macro precision | {test_metrics['macro_precision']:.4f} |
| Macro recall | {test_metrics['macro_recall']:.4f} |
| Macro F1 | {test_metrics['macro_f1']:.4f} |
| Weighted F1 | {test_metrics['weighted_f1']:.4f} |
| Cohen's kappa | {test_metrics['cohen_kappa']:.4f} |
| Matthews correlation coefficient | {test_metrics['matthews_corrcoef']:.4f} |
| Log loss | {test_metrics['log_loss']:.4f} |
| Multiclass Brier score | {test_metrics['multiclass_brier']:.4f} |
| Top-2 accuracy | {test_metrics['top2_accuracy']:.4f} |
| Errors | {test_metrics['n_errors']} / {n_test} |

Session-group bootstrap 95% intervals (only {bootstrap['n_sessions']} test
sessions, so these intervals are coarse):

- Accuracy: [{bootstrap['accuracy']['ci95_low']:.4f}, {bootstrap['accuracy']['ci95_high']:.4f}]
- Macro F1: [{bootstrap['macro_f1']['ci95_low']:.4f}, {bootstrap['macro_f1']['ci95_high']:.4f}]

Per-class test results:

{utils.markdown_table(["Class", "Precision", "Recall", "F1", "Support"], per_class_rows)}

Confusion matrix (rows = true, columns = predicted):

| True \\ Predicted | Normal | Warning | Danger |
|---|---:|---:|---:|
| Normal | {matrix[0][0]} | {matrix[0][1]} | {matrix[0][2]} |
| Warning | {matrix[1][0]} | {matrix[1][1]} | {matrix[1][2]} |
| Danger | {matrix[2][0]} | {matrix[2][1]} | {matrix[2][2]} |

## Per-session test results

{utils.markdown_table(["Session", "Rows", "Accuracy", "Macro F1", "Danger recall", "Errors", "Danger->Normal"], session_rows)}

## Rule-based baseline

Thresholds learned on the training split only: Normal <= {normal_max:.0f} ADC,
Warning <= {warning_max:.0f} ADC, otherwise Danger.

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Validation | {validation_rule_metrics['accuracy']:.4f} | {validation_rule_metrics['macro_f1']:.4f} |
| Test | {test_rule_metrics['accuracy']:.4f} | {test_rule_metrics['macro_f1']:.4f} |

McNemar exact test on the test split: model-only correct
{baseline['mcnemar_test_vs_rule']['model_only_correct']}, baseline-only correct
{baseline['mcnemar_test_vs_rule']['baseline_only_correct']}, p =
{baseline['mcnemar_test_vs_rule']['exact_binomial_p']:.4f}. The rule is not
significantly different from the model at this sample size, so no accuracy
advantage over a simple threshold rule is claimed.

## Calibration

Top-label ECE = {calibration_summary['ece']:.4f}, MCE =
{calibration_summary['mce']:.4f}. Probabilities were measured but not
recalibrated. Any alarm threshold that uses probabilities must be validated on
real sensor data.

## Artifacts

- `results/metrics/metrics.json`
- `results/metrics/classification_report_test.txt`
- `results/metrics/classification_report_validation.txt`
- `results/metrics/model_comparison.csv`
- `results/metrics/confusion_matrix_test.csv`
- `results/metrics/per_session_test_metrics.csv`
- `results/plots/*.png`

## Interpretation limits

The dataset is synthetic. These metrics describe the supplied data-generating
process, not a physical MQ-5 installation. Three held-out sessions provide only
a coarse estimate of session-to-session variation. See `docs/limitations.md`.
"""
    utils.write_text(config.METRICS_DIR / "evaluation_summary.md", report)

    print(
        f"[evaluate] test accuracy={test_metrics['accuracy']:.4f} "
        f"macro-F1={test_metrics['macro_f1']:.4f}"
    )
    print(
        "[evaluate] test Danger recall="
        f"{test_metrics['per_class']['Danger']['recall']:.4f}, "
        f"errors={test_metrics['n_errors']}"
    )
    print("[evaluate] wrote results/metrics/* and results/plots/*")


if __name__ == "__main__":
    main()
