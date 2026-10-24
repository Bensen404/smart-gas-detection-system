"""Validate the supplied dataset and build session-grouped data splits.

The raw CSV is never modified.  The script checks its SHA-256 before and after
preparation and writes derived files to ``data/processed/``.

Policy on bad data
------------------
The supplied dataset is complete, balanced and free of duplicates.  Rather than
silently dropping or imputing rows, this module raises a descriptive error when
it encounters missing values, duplicate rows, conflicting labels or values
outside the documented ranges.  That keeps the reported metrics traceable to an
explicit data state.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, utils
else:
    from . import config, utils


class DatasetValidationError(ValueError):
    """Raised when the raw dataset violates the documented contract."""


def _strip_string_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    changed: list[str] = []
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object:
            before = df[column].copy()
            df[column] = df[column].astype("string").str.strip()
            if not before.astype("string").equals(df[column]):
                changed.append(column)
    return df, changed


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate columns, dtypes, ranges, labels and duplicates.

    Returns the cleaned frame (columns reordered) and an audit dictionary.
    """
    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    missing_columns = [
        column for column in config.REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise DatasetValidationError(
            f"Missing required columns: {missing_columns}. "
            f"Found: {list(frame.columns)}"
        )
    unexpected = [
        column for column in frame.columns if column not in config.REQUIRED_COLUMNS
    ]
    if unexpected:
        raise DatasetValidationError(
            f"Unexpected columns in dataset: {unexpected}. "
            "Update the dataset contract before proceeding."
        )

    frame, stripped_columns = _strip_string_columns(frame)
    frame = frame[config.REQUIRED_COLUMNS].copy()

    missing_cells = int(frame.isna().sum().sum())
    if missing_cells:
        details = frame.isna().sum().to_string()
        raise DatasetValidationError(
            "Missing values found; refusing to impute silently.\n" + details
        )

    observed_labels = set(frame[config.TARGET_COLUMN].unique().tolist())
    expected_labels = set(config.CLASS_ORDER)
    if observed_labels != expected_labels:
        raise DatasetValidationError(
            f"Unexpected labels: {sorted(observed_labels)}; "
            f"expected {sorted(expected_labels)}"
        )

    for column in config.FEATURE_COLUMNS_ALL:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"Column {column!r} contains non-numeric values"
            ) from error
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise DatasetValidationError(
                f"Column {column!r} contains non-finite values"
            )

    if (frame["mq5_adc"] < config.ADC_MIN).any() or (
        frame["mq5_adc"] > config.ADC_MAX
    ).any():
        raise DatasetValidationError(
            f"mq5_adc must be within {config.ADC_MIN}..{config.ADC_MAX}"
        )
    if (frame["humidity_rh_pct"] < config.HUMIDITY_MIN).any() or (
        frame["humidity_rh_pct"] > config.HUMIDITY_MAX
    ).any():
        raise DatasetValidationError(
            "humidity_rh_pct must be within 0..100"
        )

    exact_duplicates = int(frame.duplicated().sum())
    duplicate_features = int(
        frame.duplicated(subset=config.FEATURE_COLUMNS_ALL).sum()
    )
    if exact_duplicates or duplicate_features:
        raise DatasetValidationError(
            "Duplicates detected "
            f"(exact rows={exact_duplicates}, feature rows={duplicate_features}). "
            "Investigate and document the handling before training."
        )

    conflicting = int(
        frame.groupby(config.FEATURE_COLUMNS_ALL, dropna=False)[
            config.TARGET_COLUMN
        ]
        .nunique()
        .gt(1)
        .sum()
    )
    if conflicting:
        raise DatasetValidationError(
            f"{conflicting} feature groups have conflicting labels"
        )

    audit = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_cells": missing_cells,
        "exact_duplicates": exact_duplicates,
        "duplicate_feature_rows": duplicate_features,
        "conflicting_feature_groups": conflicting,
        "stripped_string_columns": stripped_columns,
        "class_counts": {
            name: int((frame[config.TARGET_COLUMN] == name).sum())
            for name in config.CLASS_ORDER
        },
        "sessions": sorted(frame[config.SESSION_COLUMN].unique().tolist()),
    }
    return frame, audit


def load_raw_dataset(
    path: Path | None = None, verify_checksum: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the immutable raw dataset."""
    source = path or config.DATASET_PATH
    if verify_checksum:
        utils.verify_raw_dataset(source)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    frame = pd.read_csv(source)
    return validate_dataset(frame)


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create session-grouped train/validation/test partitions.

    Whole sessions are assigned to a partition.  This prevents the model from
    seeing the same environmental operating point in training and evaluation.
    """
    groups = df[config.SESSION_COLUMN]

    first = GroupShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SESSION_FRACTION,
        random_state=config.RANDOM_SEED,
    )
    train_validation_index, test_index = next(
        first.split(df, df[config.TARGET_COLUMN], groups)
    )

    train_validation_groups = groups.iloc[train_validation_index]
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=config.VAL_SESSION_FRACTION,
        random_state=config.RANDOM_SEED,
    )
    train_relative, validation_relative = next(
        second.split(
            df.iloc[train_validation_index],
            df[config.TARGET_COLUMN].iloc[train_validation_index],
            train_validation_groups,
        )
    )

    train_index = train_validation_index[train_relative]
    validation_index = train_validation_index[validation_relative]

    splits = {
        "train": df.iloc[np.sort(train_index)].reset_index(drop=True),
        "validation": df.iloc[np.sort(validation_index)].reset_index(drop=True),
        "test": df.iloc[np.sort(test_index)].reset_index(drop=True),
    }

    split_sessions = {
        name: sorted(part[config.SESSION_COLUMN].unique().tolist())
        for name, part in splits.items()
    }

    for name, expected in config.EXPECTED_SPLIT.items():
        if split_sessions[name] != sorted(expected):
            raise RuntimeError(
                f"Split mismatch for {name}: got {split_sessions[name]}, "
                f"expected {sorted(expected)}. Check the scikit-learn version "
                "and config.RANDOM_SEED."
            )

    partitions = list(split_sessions.values())
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            overlap = set(partitions[left]) & set(partitions[right])
            if overlap:
                raise RuntimeError(f"Session leakage detected: {sorted(overlap)}")

    return splits


def _split_profile(part: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(part)),
        "n_sessions": int(part[config.SESSION_COLUMN].nunique()),
        "sessions": sorted(part[config.SESSION_COLUMN].unique().tolist()),
        "class_counts": {
            name: int((part[config.TARGET_COLUMN] == name).sum())
            for name in config.CLASS_ORDER
        },
        "class_pct": {
            name: round(
                float((part[config.TARGET_COLUMN] == name).mean() * 100), 4
            )
            for name in config.CLASS_ORDER
        },
        "mq5_adc": {
            "min": float(part["mq5_adc"].min()),
            "max": float(part["mq5_adc"].max()),
            "mean": float(part["mq5_adc"].mean()),
            "std": float(part["mq5_adc"].std()),
        },
    }


def dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            column: int(value) for column, value in df.isna().sum().items()
        },
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(
            df.duplicated(subset=config.FEATURE_COLUMNS_ALL).sum()
        ),
        "n_sessions": int(df[config.SESSION_COLUMN].nunique()),
        "class_counts": {
            name: int((df[config.TARGET_COLUMN] == name).sum())
            for name in config.CLASS_ORDER
        },
        "descriptive_statistics": df[config.FEATURE_COLUMNS_ALL]
        .describe()
        .to_dict(),
    }


def write_processed_data(
    df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    audit: dict[str, Any],
    source_hash: str,
) -> dict[str, Any]:
    """Persist splits, metadata and a human-readable preparation summary."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in splits.items():
        part.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)

    sessions = {
        name: sorted(part[config.SESSION_COLUMN].unique().tolist())
        for name, part in splits.items()
    }

    split_manifest = {
        "random_seed": config.RANDOM_SEED,
        "source_sha256": source_hash,
        "split_sessions": sessions,
        "rows": {name: int(len(part)) for name, part in splits.items()},
        "no_session_overlap": True,
    }
    utils.write_json(config.PROCESSED_DIR / "split_manifest.json", split_manifest)

    metadata = {
        "project": "Smart Gas AI",
        "source_dataset": utils.relative_to_root(config.DATASET_PATH),
        "source_sha256": source_hash,
        "source_size_bytes": int(config.DATASET_PATH.stat().st_size),
        "original_preserved": True,
        "rows_total": int(len(df)),
        "columns": list(df.columns),
        "feature_columns_all": config.FEATURE_COLUMNS_ALL,
        "model_features": config.MODEL_FEATURES,
        "target_column": config.TARGET_COLUMN,
        "session_column": config.SESSION_COLUMN,
        "class_order": config.CLASS_ORDER,
        "random_seed": config.RANDOM_SEED,
        "split_protocol": {
            "method": "GroupShuffleSplit on session_id, applied twice",
            "test_session_fraction": config.TEST_SESSION_FRACTION,
            "validation_session_fraction_of_remaining": config.VAL_SESSION_FRACTION,
            "description": (
                "3 of 18 sessions are held out as test; 3 of the remaining 15 "
                "sessions are held out as validation; the other 12 sessions "
                "form the training split. No session appears in two splits."
            ),
        },
        "split_sessions": sessions,
        "split_profile": {
            name: _split_profile(part) for name, part in splits.items()
        },
        "dataset_profile": dataset_profile(df),
        "cleaning": {
            "rows_removed": 0,
            "rows_imputed": 0,
            "labels_changed": 0,
            "duplicates_removed": 0,
            "string_columns_stripped": audit["stripped_string_columns"],
            "note": (
                "The supplied CSV was already complete, balanced and internally "
                "consistent. It was validated rather than cleaned."
            ),
        },
        "environment": utils.environment_info(),
    }
    utils.write_json(config.PROCESSED_DIR / "metadata.json", metadata)

    lines = [
        "=" * 78,
        "SMART GAS AI - DATASET PREPARATION AND INTEGRITY REPORT",
        "=" * 78,
        "",
        f"Source dataset : {utils.relative_to_root(config.DATASET_PATH)}",
        f"SHA-256        : {source_hash}",
        f"Size           : {config.DATASET_PATH.stat().st_size:,} bytes",
        f"Shape          : {len(df):,} rows x {df.shape[1]} columns",
        f"Sessions       : {df[config.SESSION_COLUMN].nunique()}",
        "",
        "[1] Column contract",
        f"    Required columns present: {', '.join(config.REQUIRED_COLUMNS)}",
        f"    String columns stripped : {audit['stripped_string_columns'] or 'none needed'}",
        "",
        "[2] Data quality",
        f"    Missing cells            : {audit['missing_cells']}",
        f"    Exact duplicate rows     : {audit['exact_duplicates']}",
        f"    Duplicate feature rows   : {audit['duplicate_feature_rows']}",
        f"    Conflicting-label groups : {audit['conflicting_feature_groups']}",
        "    Rows removed             : 0",
        "    Rows imputed             : 0",
        "    Labels changed           : 0",
        "",
        "[3] Class distribution (full dataset)",
    ]
    for name in config.CLASS_ORDER:
        count = int((df[config.TARGET_COLUMN] == name).sum())
        lines.append(
            f"    {name:<10} {count:>5} rows ({count / len(df) * 100:5.2f}%)"
        )
    lines += [
        "",
        "[4] Session-grouped split",
        f"    Random seed: {config.RANDOM_SEED}",
        "    Test sessions       : " + ", ".join(sessions["test"]),
        "    Validation sessions : " + ", ".join(sessions["validation"]),
        "    Training sessions   : " + ", ".join(sessions["train"]),
        "",
        f"    {'Split':<12}{'Rows':>6}{'Sessions':>10}  Class counts",
    ]
    for name in ["train", "validation", "test"]:
        part = splits[name]
        counts = ", ".join(
            f"{label}={int((part[config.TARGET_COLUMN] == label).sum())}"
            for label in config.CLASS_ORDER
        )
        lines.append(
            f"    {name:<12}{len(part):>6}"
            f"{part[config.SESSION_COLUMN].nunique():>10}  {counts}"
        )
    lines += [
        "",
        "[5] Feature ranges",
        f"    {'Feature':<20}{'Min':>10}{'Max':>10}{'Mean':>12}{'Std':>12}",
    ]
    for column in config.FEATURE_COLUMNS_ALL:
        lines.append(
            f"    {column:<20}{df[column].min():>10.2f}{df[column].max():>10.2f}"
            f"{df[column].mean():>12.2f}{df[column].std():>12.2f}"
        )
    lines += [
        "",
        "[6] Model input contract",
        f"    Model features: {', '.join(config.MODEL_FEATURES)}",
        f"    Note: {config.MODEL_INPUT_NOTE}",
        "",
        "[7] Outputs",
        "    data/processed/train.csv",
        "    data/processed/validation.csv",
        "    data/processed/test.csv",
        "    data/processed/split_manifest.json",
        "    data/processed/metadata.json",
        "",
        "[8] Integrity",
        "    The raw CSV was opened read-only and was not modified.",
    ]
    hash_after = utils.sha256_file(config.DATASET_PATH)
    lines.append(f"    SHA-256 after preparation: {hash_after}")
    if hash_after != source_hash:
        raise RuntimeError("Raw dataset changed during preparation")
    lines.append("    Integrity re-check: PASS")
    utils.write_text(
        config.PROCESSED_DIR / "dataset_summary.txt", "\n".join(lines) + "\n"
    )
    return metadata


def write_sample_table(df: pd.DataFrame, per_class: int = 8) -> Path:
    """Write a small, deterministic sample of real rows for demos and tests."""
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample = (
        df.groupby(config.TARGET_COLUMN, group_keys=False)
        .head(per_class)
        .sort_values([config.TARGET_COLUMN, "mq5_adc"])
        .reset_index(drop=True)
    )
    path = config.SAMPLE_DIR / "sample_readings.csv"
    sample.to_csv(path, index=False)
    return path


def load_processed_splits() -> dict[str, pd.DataFrame]:
    paths = {
        name: config.PROCESSED_DIR / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Processed splits are missing. Run `python -m src.data_preprocessing` "
            f"first. Missing: {missing}"
        )
    return {name: pd.read_csv(path) for name, path in paths.items()}


def validate_adc(value: object) -> float:
    """Validate one raw ADC reading and return it as a float."""
    if isinstance(value, bool):
        raise ValueError("mq5_adc must be numeric, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"mq5_adc must be numeric, got {value!r}") from error
    if not math.isfinite(numeric):
        raise ValueError("mq5_adc must be finite")
    if numeric < config.ADC_MIN or numeric > config.ADC_MAX:
        raise ValueError(
            f"mq5_adc must be within {config.ADC_MIN}..{config.ADC_MAX}, "
            f"got {numeric}"
        )
    return numeric


def main() -> None:
    utils.ensure_directories()
    utils.set_global_seed()
    source_hash = utils.verify_raw_dataset()
    print(f"[prepare] source: {utils.relative_to_root(config.DATASET_PATH)}")
    print(f"[prepare] sha256: {source_hash}")

    df, audit = load_raw_dataset()
    splits = make_splits(df)
    write_processed_data(df, splits, audit, source_hash)
    sample_path = write_sample_table(df)

    print(
        "[prepare] split sizes:",
        {name: len(part) for name, part in splits.items()},
    )
    print("[prepare] test sessions:", sorted(splits["test"]["session_id"].unique()))
    print("[prepare] sample:", utils.relative_to_root(sample_path))
    print("[prepare] raw SHA-256 re-check: PASS")
    print("[prepare] wrote data/processed/* and data/processed/dataset_summary.txt")


if __name__ == "__main__":
    main()
