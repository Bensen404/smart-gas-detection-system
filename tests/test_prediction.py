"""Tests for inference, the model contract and the exported C header."""

from __future__ import annotations

import math
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, utils
from src.predict import load_artifacts, predict_reading

HEADER_PATH = config.FIRMWARE_DIR / "gas_model.h"


def _parse_c_array(text: str, name: str, size: int) -> np.ndarray:
    pattern = rf"{name}\s*\[[^\]]+\]\s*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise AssertionError(f"Could not find array {name} in {HEADER_PATH}")
    values = [
        float(token.strip().rstrip("fF"))
        for token in match.group(1).split(",")
        if token.strip()
    ]
    if len(values) != size:
        raise AssertionError(
            f"{name}: expected {size} values, found {len(values)}"
        )
    return np.asarray(values, dtype=np.float32)


def _softmax_float32(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max()
    exponentials = np.exp(shifted.astype(np.float64)).astype(np.float32)
    return (exponentials / exponentials.sum(dtype=np.float32)).astype(np.float32)


class TestPredictionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.selected = load_artifacts()

    def test_feature_contract(self) -> None:
        self.assertEqual(config.MODEL_FEATURES, ["mq5_adc"])
        self.assertEqual(self.selected["features"], ["mq5_adc"])
        self.assertEqual(self.selected["class_order"], config.CLASS_ORDER)

    def test_prediction_shape(self) -> None:
        for adc in (0, 500, 900, 1300, 1700, 2200, 2700, 3600, 4095):
            with self.subTest(adc=adc):
                result = predict_reading(
                    adc, model=self.model, selected=self.selected
                )
                self.assertIn(result["label"], config.CLASS_ORDER)
                self.assertEqual(
                    set(result["probabilities"]), set(config.CLASS_ORDER)
                )
                self.assertAlmostEqual(
                    sum(result["probabilities"].values()), 1.0, places=6
                )
                self.assertAlmostEqual(
                    result["confidence"],
                    max(result["probabilities"].values()),
                    places=9,
                )
                self.assertEqual(
                    result["class_index"],
                    config.CLASS_TO_INDEX[result["label"]],
                )
                self.assertEqual(
                    result["label"],
                    max(result["probabilities"], key=result["probabilities"].get),
                )

    def test_context_is_not_used(self) -> None:
        plain = predict_reading(
            1500, model=self.model, selected=self.selected
        )
        contextual = predict_reading(
            1500,
            temperature_c=40.0,
            humidity_rh_pct=85.0,
            model=self.model,
            selected=self.selected,
        )
        self.assertEqual(plain["label"], contextual["label"])
        self.assertEqual(plain["probabilities"], contextual["probabilities"])
        self.assertFalse(contextual["context_used_by_model"])
        self.assertEqual(contextual["context"]["temperature_c"], 40.0)
        self.assertEqual(contextual["context"]["humidity_rh_pct"], 85.0)

    def test_invalid_adc_is_rejected(self) -> None:
        for value in (-1, 4096, math.nan, math.inf, "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    predict_reading(
                        value, model=self.model, selected=self.selected
                    )

    def test_invalid_humidity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            predict_reading(
                1500,
                humidity_rh_pct=101,
                model=self.model,
                selected=self.selected,
            )


class TestExportedHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.selected = load_artifacts()
        if not HEADER_PATH.is_file():
            raise unittest.SkipTest(
                "Firmware header missing; run `python -m src.export_embedded`."
            )
        cls.header = HEADER_PATH.read_text(encoding="utf-8")
        scaler = cls.model.named_steps["scaler"]
        estimator = cls.model.named_steps["model"]
        cls.mean = np.asarray(scaler.mean_, dtype=np.float32)
        cls.scale = np.asarray(scaler.scale_, dtype=np.float32)

        # sklearn orders classes alphabetically (Danger, Normal, Warning), while
        # the firmware header uses Normal, Warning, Danger. Reorder before
        # comparing class parameters.
        model_classes = list(estimator.classes_)
        reorder = [model_classes.index(name) for name in config.CLASS_ORDER]
        cls.theta = np.asarray(estimator.theta_, dtype=np.float32).reshape(
            -1
        )[reorder]
        cls.variance = np.asarray(estimator.var_, dtype=np.float32).reshape(
            -1
        )[reorder]
        cls.prior = np.asarray(
            estimator.class_prior_, dtype=np.float32
        ).reshape(-1)[reorder]

    def test_header_constants_match_model(self) -> None:
        header_mean = _parse_c_array(
            self.header, "GAS_MODEL_MEAN", len(self.mean)
        )
        header_scale = _parse_c_array(
            self.header, "GAS_MODEL_SCALE", len(self.scale)
        )
        header_theta = _parse_c_array(self.header, "GAS_MODEL_THETA", 3)
        header_inverse_variance = _parse_c_array(
            self.header, "GAS_MODEL_INV_VAR", 3
        )
        header_log_constant = _parse_c_array(
            self.header, "GAS_MODEL_LOG_CONST", 3
        )
        header_prior = _parse_c_array(
            self.header, "GAS_MODEL_CLASS_PRIOR", 3
        )

        expected_inverse_variance = (
            np.float32(1.0) / self.variance
        ).astype(np.float32)
        expected_log_constant = (
            np.log(self.prior.astype(np.float64))
            - 0.5
            * np.log(2.0 * np.pi * self.variance.astype(np.float64))
        ).astype(np.float32)

        np.testing.assert_allclose(header_mean, self.mean, rtol=0, atol=1e-6)
        np.testing.assert_allclose(header_scale, self.scale, rtol=0, atol=1e-5)
        np.testing.assert_allclose(header_theta, self.theta, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            header_inverse_variance,
            expected_inverse_variance,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            header_log_constant,
            expected_log_constant,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(header_prior, self.prior, rtol=0, atol=1e-7)

    def test_header_predictions_match_python(self) -> None:
        header_mean = _parse_c_array(self.header, "GAS_MODEL_MEAN", 1)
        header_scale = _parse_c_array(self.header, "GAS_MODEL_SCALE", 1)
        header_theta = _parse_c_array(self.header, "GAS_MODEL_THETA", 3)
        header_inverse_variance = _parse_c_array(
            self.header, "GAS_MODEL_INV_VAR", 3
        )
        header_log_constant = _parse_c_array(
            self.header, "GAS_MODEL_LOG_CONST", 3
        )

        adc_values = np.arange(0, 4096, 50, dtype=np.float32)
        frame = pd.DataFrame({"mq5_adc": adc_values})
        python_probabilities = self.model.predict_proba(frame)
        model_classes = list(self.model.named_steps["model"].classes_)
        reorder = [model_classes.index(name) for name in config.CLASS_ORDER]
        python_probabilities = python_probabilities[:, reorder]

        for index, adc in enumerate(adc_values):
            z = np.float32(
                (np.float32(adc) - header_mean[0]) / header_scale[0]
            )
            d = z - header_theta
            logits = (
                header_log_constant
                - np.float32(0.5) * d * d * header_inverse_variance
            )
            probabilities = _softmax_float32(logits)
            with self.subTest(adc=float(adc)):
                np.testing.assert_allclose(
                    probabilities,
                    python_probabilities[index].astype(np.float32),
                    rtol=0,
                    atol=1e-5,
                )

    def test_parity_log_is_recorded(self) -> None:
        log_path = config.METRICS_DIR / "embedded_parity.txt"
        if not log_path.is_file():
            self.skipTest("Parity log missing")
        text = log_path.read_text(encoding="utf-8")
        if "Status: PASS" not in text:
            self.skipTest("Host C parity was not run in this checkout")
        match = re.search(r"Cases: (\d+)", text)
        self.assertIsNotNone(match)
        self.assertGreater(int(match.group(1)), 0)


if __name__ == "__main__":
    unittest.main()
