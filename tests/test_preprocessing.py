"""Tests for dataset validation and the session-grouped split."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src import config, utils
from src.data_preprocessing import (
    DatasetValidationError,
    load_raw_dataset,
    make_splits,
    validate_adc,
    validate_dataset,
)


class TestDatasetContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df, cls.audit = load_raw_dataset()

    def test_raw_checksum(self) -> None:
        self.assertEqual(
            utils.verify_raw_dataset(), config.EXPECTED_DATASET_SHA256
        )

    def test_shape_and_columns(self) -> None:
        self.assertEqual(self.df.shape, (3564, 5))
        self.assertEqual(list(self.df.columns), config.REQUIRED_COLUMNS)

    def test_no_missing_values(self) -> None:
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_no_duplicates(self) -> None:
        self.assertEqual(int(self.df.duplicated().sum()), 0)
        self.assertEqual(
            int(self.df.duplicated(subset=config.FEATURE_COLUMNS_ALL).sum()),
            0,
        )

    def test_balanced_labels(self) -> None:
        counts = self.df[config.TARGET_COLUMN].value_counts().to_dict()
        self.assertEqual(counts, {"Normal": 1188, "Warning": 1188, "Danger": 1188})

    def test_sessions(self) -> None:
        sessions = sorted(self.df[config.SESSION_COLUMN].unique().tolist())
        self.assertEqual(len(sessions), 18)
        self.assertEqual(sessions[0], "ENC_01")
        self.assertEqual(sessions[-1], "ENC_18")

    def test_split_sizes_and_isolation(self) -> None:
        splits = make_splits(self.df)
        self.assertEqual(len(splits["train"]), 2376)
        self.assertEqual(len(splits["validation"]), 594)
        self.assertEqual(len(splits["test"]), 594)
        for name, expected in config.EXPECTED_SPLIT.items():
            observed = sorted(
                splits[name][config.SESSION_COLUMN].unique().tolist()
            )
            self.assertEqual(observed, sorted(expected))
        train_sessions = set(splits["train"][config.SESSION_COLUMN])
        validation_sessions = set(splits["validation"][config.SESSION_COLUMN])
        test_sessions = set(splits["test"][config.SESSION_COLUMN])
        self.assertFalse(train_sessions & validation_sessions)
        self.assertFalse(train_sessions & test_sessions)
        self.assertFalse(validation_sessions & test_sessions)

    def test_split_class_balance(self) -> None:
        splits = make_splits(self.df)
        expected = {"train": 792, "validation": 198, "test": 198}
        for name, part in splits.items():
            counts = part[config.TARGET_COLUMN].value_counts().to_dict()
            self.assertEqual(
                counts,
                {
                    "Normal": expected[name],
                    "Warning": expected[name],
                    "Danger": expected[name],
                },
            )


class TestValidationErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df, _ = load_raw_dataset()

    def test_rejects_out_of_range_adc(self) -> None:
        broken = self.df.head(10).copy()
        broken.loc[broken.index[0], "mq5_adc"] = 5000
        with self.assertRaises(DatasetValidationError):
            validate_dataset(broken)

    def test_rejects_missing_value(self) -> None:
        broken = self.df.head(10).copy()
        broken.loc[broken.index[0], "mq5_adc"] = np.nan
        with self.assertRaises(DatasetValidationError):
            validate_dataset(broken)

    def test_rejects_unexpected_label(self) -> None:
        broken = self.df.head(10).copy()
        broken.loc[broken.index[0], config.TARGET_COLUMN] = "Unknown"
        with self.assertRaises(DatasetValidationError):
            validate_dataset(broken)

    def test_rejects_unexpected_column(self) -> None:
        broken = self.df.head(10).copy()
        broken["unexpected"] = 1
        with self.assertRaises(DatasetValidationError):
            validate_dataset(broken)


class TestReadingValidation(unittest.TestCase):
    def test_accepts_valid_adc(self) -> None:
        for value in (0, 1, 900, 1794, 4095, 1500.5):
            self.assertEqual(validate_adc(value), float(value))

    def test_rejects_invalid_adc(self) -> None:
        for value in (-1, 4096, 1e9, math.nan, math.inf, True, "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_adc(value)


if __name__ == "__main__":
    unittest.main()
