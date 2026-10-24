"""Export the frozen model to a header-only C99 implementation.

The deployed firmware consumes ``firmware/esp32_gas_detection/gas_model.h``.
This script generates that header from the Python artifact, writes a model
card and test vectors, and (when GCC is available) compiles and runs a host-side
C parity test against scikit-learn.

A successful host test proves source-level portability and numerical parity.
It does **not** prove ESP32 deployment readiness.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, utils
else:
    from . import config, utils


def _c_float(value: float) -> str:
    text = format(float(np.float32(value)), ".9g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def _softmax_float32(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted.astype(np.float64)).astype(np.float32)
    return (exponentials / np.sum(exponentials, dtype=np.float32)).astype(
        np.float32
    )


def _header_text(
    mean: np.ndarray,
    scale: np.ndarray,
    constants_block: str,
    logits_block: str,
    model_label: str,
    family: str,
) -> str:
    mean_c = ", ".join(_c_float(value) for value in mean)
    scale_c = ", ".join(_c_float(value) for value in scale)
    class_names = ", ".join(f'"{name}"' for name in config.CLASS_ORDER)
    return f"""/*
 * Smart Gas AI - frozen classifier export
 * ---------------------------------------
 * Model: {model_label} ({family})
 * Input: MQ-5 ADC, 12-bit (0..4095), feature order: mq5_adc
 * Output classes: 0 = Normal, 1 = Warning, 2 = Danger
 *
 * Auto-generated from models/final_model.joblib. Do not edit by hand.
 * Dataset SHA-256: {config.EXPECTED_DATASET_SHA256}
 * Random seed: {config.RANDOM_SEED}
 *
 * Header-only C99. It performs one standardisation, three class scores and an
 * argmax. Probabilities additionally use expf(). There is no dynamic
 * allocation and no external ML runtime.
 *
 * This is an educational prototype, not a safety-certified gas detector.
 */

#ifndef SMART_GAS_AI_MODEL_H
#define SMART_GAS_AI_MODEL_H

#include <math.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define GAS_MODEL_VERSION "1.0.0"
#define GAS_MODEL_N_FEATURES 1
#define GAS_MODEL_N_CLASSES 3

typedef enum {{
    GAS_CLASS_NORMAL = 0,
    GAS_CLASS_WARNING = 1,
    GAS_CLASS_DANGER = 2
}} gas_class_t;

static const char *GAS_MODEL_CLASS_NAMES[GAS_MODEL_N_CLASSES] = {{
    {class_names}
}};

/* StandardScaler statistics (float32). */
static const float GAS_MODEL_MEAN[GAS_MODEL_N_FEATURES] = {{
    {mean_c}
}};
static const float GAS_MODEL_SCALE[GAS_MODEL_N_FEATURES] = {{
    {scale_c}
}};

{constants_block}

/*
 * Compute three class scores for one ADC sample.
 * adc: raw 12-bit reading (0..4095). Out-of-range values are accepted by this
 * function but are outside the training distribution and should be rejected
 * by the caller.
 */
static inline void gas_model_logits(float adc, float logits[GAS_MODEL_N_CLASSES])
{{
    const float z = (adc - GAS_MODEL_MEAN[0]) / GAS_MODEL_SCALE[0];
{logits_block}
}}

/* Return the most likely class index without calling expf(). */
static inline int gas_model_predict(float adc)
{{
    float logits[GAS_MODEL_N_CLASSES];
    gas_model_logits(adc, logits);
    int best = 0;
    for (int c = 1; c < GAS_MODEL_N_CLASSES; ++c) {{
        if (logits[c] > logits[best]) {{
            best = c;
        }}
    }}
    return best;
}}

/* Numerical-stability softmax; writes three probabilities that sum to 1. */
static inline void gas_model_predict_proba(
    float adc,
    float probs[GAS_MODEL_N_CLASSES]
)
{{
    float logits[GAS_MODEL_N_CLASSES];
    gas_model_logits(adc, logits);
    float maximum = logits[0];
    for (int c = 1; c < GAS_MODEL_N_CLASSES; ++c) {{
        if (logits[c] > maximum) {{
            maximum = logits[c];
        }}
    }}
    float total = 0.0f;
    for (int c = 0; c < GAS_MODEL_N_CLASSES; ++c) {{
        probs[c] = expf(logits[c] - maximum);
        total += probs[c];
    }}
    for (int c = 0; c < GAS_MODEL_N_CLASSES; ++c) {{
        probs[c] /= total;
    }}
}}

#ifdef __cplusplus
}}
#endif

#endif /* SMART_GAS_AI_MODEL_H */
"""


PARITY_SOURCE = r"""#include <stdio.h>
#include <math.h>
#include "gas_model.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s test_vectors.csv\n", argv[0]);
        return 2;
    }
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    char line[1024];
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "empty file\n");
        fclose(fp);
        return 2;
    }

    int cases = 0;
    int class_matches = 0;
    int failures = 0;
    double max_probability_difference = 0.0;
    printf("case_id,adc,expected_class,predicted_class,class_ok,max_prob_abs_diff,status\n");

    while (fgets(line, sizeof(line), fp)) {
        int case_id;
        int expected_class;
        float adc;
        float expected[3];
        if (sscanf(line, "%d,%f,%d,%f,%f,%f",
                   &case_id, &adc, &expected_class,
                   &expected[0], &expected[1], &expected[2]) != 6) {
            continue;
        }

        int predicted_class = gas_model_predict(adc);
        float probabilities[3];
        gas_model_predict_proba(adc, probabilities);

        double difference = 0.0;
        for (int i = 0; i < 3; ++i) {
            double candidate = fabs((double)probabilities[i] - (double)expected[i]);
            if (candidate > difference) {
                difference = candidate;
            }
        }

        int class_ok = (predicted_class == expected_class);
        int ok = class_ok && (difference <= 1e-4);
        cases += 1;
        class_matches += class_ok;
        if (!ok) {
            failures += 1;
        }
        if (difference > max_probability_difference) {
            max_probability_difference = difference;
        }
        printf("%d,%.1f,%d,%d,%s,%.9f,%s\n",
               case_id, adc, expected_class, predicted_class,
               class_ok ? "yes" : "no", difference,
               ok ? "PASS" : "FAIL");
    }

    fclose(fp);
    printf("SUMMARY cases=%d class_matches=%d max_prob_abs_diff=%.9f failures=%d %s\n",
           cases, class_matches, max_probability_difference, failures,
           failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
"""


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    print("[export] " + " ".join(command))
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )


def _build_parameters(
    model, selected: dict[str, Any]
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
    str,
    dict[str, Any],
    list[np.ndarray],
    str,
]:
    family = selected["family"]
    if selected["features"] != config.MODEL_FEATURES:
        raise RuntimeError(
            "The firmware contract expects the ADC-only model "
            f"{config.MODEL_FEATURES}, but the selected model uses "
            f"{selected['features']}."
        )

    scaler = model.named_steps["scaler"]
    estimator = model.named_steps["model"]
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    if mean.shape != (1,) or scale.shape != (1,):
        raise RuntimeError(f"Unexpected scaler shapes: {mean.shape}, {scale.shape}")

    model_classes = list(estimator.classes_)
    if set(model_classes) != set(config.CLASS_ORDER):
        raise RuntimeError(f"Unexpected classes in fitted model: {model_classes}")
    reorder = [model_classes.index(name) for name in config.CLASS_ORDER]

    if family == "GaussianNB":
        theta = np.asarray(estimator.theta_, dtype=np.float32).reshape(-1)[reorder]
        variance = np.asarray(estimator.var_, dtype=np.float32).reshape(-1)[reorder]
        prior = np.asarray(estimator.class_prior_, dtype=np.float32).reshape(-1)[
            reorder
        ]
        if theta.shape != (3,) or variance.shape != (3,):
            raise RuntimeError(
                f"Unexpected GaussianNB parameter shapes: {theta.shape}, {variance.shape}"
            )
        inverse_variance = (np.float32(1.0) / variance).astype(np.float32)
        log_constant = (
            np.log(prior.astype(np.float64))
            - 0.5 * np.log(2.0 * np.pi * variance.astype(np.float64))
        ).astype(np.float32)
        constants = f"""/* Gaussian naive-Bayes class parameters (float32), one value per class. */
static const float GAS_MODEL_THETA[GAS_MODEL_N_CLASSES] = {{
    {", ".join(_c_float(value) for value in theta)}
}};
static const float GAS_MODEL_INV_VAR[GAS_MODEL_N_CLASSES] = {{
    {", ".join(_c_float(value) for value in inverse_variance)}
}};
static const float GAS_MODEL_LOG_CONST[GAS_MODEL_N_CLASSES] = {{
    {", ".join(_c_float(value) for value in log_constant)}
}};
static const float GAS_MODEL_CLASS_PRIOR[GAS_MODEL_N_CLASSES] = {{
    {", ".join(_c_float(value) for value in prior)}
}};"""
        logits = """    for (int c = 0; c < GAS_MODEL_N_CLASSES; ++c) {
        const float d = z - GAS_MODEL_THETA[c];
        logits[c] = GAS_MODEL_LOG_CONST[c] - 0.5f * d * d * GAS_MODEL_INV_VAR[c];
    }"""
        parameters = {
            "theta_float32": theta.astype(float).tolist(),
            "variance_float32": variance.astype(float).tolist(),
            "inverse_variance_float32": inverse_variance.astype(float).tolist(),
            "log_constant_float32": log_constant.astype(float).tolist(),
            "class_prior_float32": prior.astype(float).tolist(),
        }
        arrays = [mean, scale, theta, inverse_variance, log_constant, prior]
        equation = (
            "logit_c = log(prior_c) - 0.5*log(2*pi*var_c) "
            "- 0.5*((z - theta_c)^2 / var_c)"
        )
        return mean, scale, constants, logits, parameters, arrays, equation

    if family == "LogisticRegression":
        coefficients = np.asarray(estimator.coef_, dtype=np.float32)[reorder, :]
        intercept = np.asarray(estimator.intercept_, dtype=np.float32)[reorder]
        if coefficients.shape != (3, 1):
            raise RuntimeError(
                f"Unexpected logistic-regression coefficient shape: {coefficients.shape}"
            )
        constants = f"""/* Multinomial logistic-regression weights (float32), shape [class][feature]. */
static const float GAS_MODEL_COEF[GAS_MODEL_N_CLASSES][GAS_MODEL_N_FEATURES] = {{
    {",\n    ".join(", ".join(_c_float(value) for value in row) for row in coefficients)}
}};
static const float GAS_MODEL_INTERCEPT[GAS_MODEL_N_CLASSES] = {{
    {", ".join(_c_float(value) for value in intercept)}
}};"""
        logits = """    for (int c = 0; c < GAS_MODEL_N_CLASSES; ++c) {
        logits[c] = GAS_MODEL_COEF[c][0] * z + GAS_MODEL_INTERCEPT[c];
    }"""
        parameters = {
            "coefficients_float32": coefficients.astype(float).tolist(),
            "intercepts_float32": intercept.astype(float).tolist(),
        }
        arrays = [mean, scale, coefficients, intercept]
        equation = "logit_c = coef_c * z + intercept_c"
        return mean, scale, constants, logits, parameters, arrays, equation

    raise RuntimeError(
        "The embedded exporter supports GaussianNB and LogisticRegression; "
        f"selected family is {family}."
    )


def _test_vectors(model, adc_values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"mq5_adc": adc_values})
    probabilities = model.predict_proba(frame)
    model_classes = list(model.named_steps["model"].classes_)
    reorder = [model_classes.index(name) for name in config.CLASS_ORDER]
    probabilities = probabilities[:, reorder]
    predicted = model.predict(frame)

    rows = []
    for index, adc in enumerate(adc_values):
        predicted_index = config.CLASS_TO_INDEX[str(predicted[index])]
        rows.append(
            {
                "case_id": index + 1,
                "mq5_adc": float(adc),
                "expected_class": predicted_index,
                "expected_p_normal": float(probabilities[index, 0]),
                "expected_p_warning": float(probabilities[index, 1]),
                "expected_p_danger": float(probabilities[index, 2]),
                "expected_class_name": str(predicted[index]),
            }
        )
    return pd.DataFrame(rows)


def _parity_test(
    vectors: pd.DataFrame, header_path: Path
) -> tuple[dict[str, Any], str]:
    gcc = shutil.which("gcc")
    if not gcc:
        return (
            {
                "status": "NOT RUN",
                "reason": "gcc not found on PATH",
                "cases": 0,
                "class_matches": 0,
                "max_probability_abs_diff": None,
                "failures": None,
            },
            "C parity test not run: gcc was not found on PATH.\n",
        )

    with tempfile.TemporaryDirectory(prefix="smart_gas_ai_parity_") as directory:
        build = Path(directory)
        source = build / "parity_test.c"
        executable = build / "parity_test.exe"
        vectors_path = build / "test_vectors.csv"
        source.write_text(PARITY_SOURCE, encoding="utf-8")
        vectors.to_csv(vectors_path, index=False)

        version = _run([gcc, "--version"], build)
        compile_result = _run(
            [
                gcc,
                "-O2",
                "-std=c11",
                "-Wall",
                "-Wextra",
                f"-I{header_path.parent}",
                str(source),
                "-o",
                str(executable),
                "-lm",
            ],
            build,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                "C parity build failed:\n" + compile_result.stderr
            )

        run_result = _run([str(executable), str(vectors_path)], build)
        output = run_result.stdout
        if run_result.stderr:
            output += "\n" + run_result.stderr
        summary_line = next(
            (
                line
                for line in reversed(output.splitlines())
                if line.startswith("SUMMARY")
            ),
            "",
        )
        match = re.search(
            r"cases=(\d+) class_matches=(\d+) max_prob_abs_diff=([0-9.eE+-]+) "
            r"failures=(\d+) (PASS|FAIL)",
            summary_line,
        )
        if not match:
            raise RuntimeError(f"Could not parse parity summary: {summary_line!r}")
        summary = {
            "status": match.group(5),
            "cases": int(match.group(1)),
            "class_matches": int(match.group(2)),
            "max_probability_abs_diff": float(match.group(3)),
            "failures": int(match.group(4)),
            "tolerance": 1e-4,
            "compiler": version.stdout.splitlines()[0] if version.stdout else "gcc",
            "compile_command": (
                "gcc -O2 -std=c11 -Wall -Wextra "
                f"-I{utils.relative_to_root(header_path.parent)} "
                "parity_test.c -o parity_test.exe -lm"
            ),
        }
        if summary["failures"] != 0 or summary["status"] != "PASS":
            raise RuntimeError(
                "C parity test failed. See results/metrics/embedded_parity.txt."
            )
        if summary["cases"] == 0:
            raise RuntimeError("C parity test ran zero cases")

        log = (
            "SMART GAS AI - HOST C/PYTHON PARITY TEST\n"
            "=========================================\n"
            f"Compiler: {summary['compiler']}\n"
            f"Cases: {summary['cases']}\n"
            f"Class matches: {summary['class_matches']}\n"
            f"Max probability absolute difference: "
            f"{summary['max_probability_abs_diff']:.3e}\n"
            f"Failures: {summary['failures']}\n"
            f"Status: {summary['status']} (tolerance {summary['tolerance']})\n\n"
            + output
        )
        return summary, log


def main() -> None:
    utils.ensure_directories()
    utils.verify_raw_dataset()

    selected = utils.read_json(config.MODELS_DIR / "selected_model.json")
    metrics = utils.read_json(config.METRICS_DIR / "metrics.json")
    metadata = utils.read_json(config.PROCESSED_DIR / "metadata.json")
    model = joblib.load(config.MODELS_DIR / "final_model.joblib")

    mean, scale, constants, logits, parameters, arrays, equation = (
        _build_parameters(model, selected)
    )
    family = selected["family"]
    header = _header_text(
        mean,
        scale,
        constants,
        logits,
        selected["selected_candidate"],
        family,
    )
    header_path = config.FIRMWARE_DIR / "gas_model.h"
    header_path.write_text(header, encoding="utf-8")
    print(f"[export] wrote {utils.relative_to_root(header_path)}")

    adc_values = np.unique(
        np.concatenate(
            [
                np.arange(0, 4096, 25, dtype=float),
                np.array([0.0, 4095.0, 1300.0, 2200.0]),
            ]
        )
    )
    vectors = _test_vectors(model, adc_values)
    vectors.to_csv(config.MODELS_DIR / "test_vectors.csv", index=False)

    parity, parity_log = _parity_test(vectors, header_path)
    utils.write_text(
        config.METRICS_DIR / "embedded_parity.txt", parity_log
    )

    parameter_bytes = int(sum(array.nbytes for array in arrays))
    model_json = {
        "model_name": "Smart Gas AI MQ-5 classifier",
        "version": "1.0.0",
        "type": "Gaussian naive Bayes" if family == "GaussianNB" else "multinomial logistic regression",
        "selected_candidate": selected["selected_candidate"],
        "feature_order": config.MODEL_FEATURES,
        "input_units": "raw 12-bit ADC counts (0..4095)",
        "output_classes": config.CLASS_ORDER,
        "class_indices": config.CLASS_TO_INDEX,
        "preprocessing": {
            "type": "StandardScaler",
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
            "equation": "z = (mq5_adc - mean) / scale",
        },
        "decision_rule": equation,
        "softmax": "p_c = exp(logit_c - max(logits)) / sum_j exp(logit_j - max(logits))",
        "parameters": parameters,
        "constant_parameter_bytes": parameter_bytes,
        "training": {
            "dataset": utils.relative_to_root(config.DATASET_PATH),
            "dataset_sha256": metadata["source_sha256"],
            "train_sessions": metadata["split_sessions"]["train"],
            "validation_sessions": metadata["split_sessions"]["validation"],
            "test_sessions": metadata["split_sessions"]["test"],
            "fit_sessions_final": (
                metadata["split_sessions"]["train"]
                + metadata["split_sessions"]["validation"]
            ),
            "n_fit_rows": selected["final_fit"]["n_rows"],
            "random_seed": config.RANDOM_SEED,
        },
        "evaluation": {
            "test_accuracy": metrics["test"]["accuracy"],
            "test_macro_f1": metrics["test"]["macro_f1"],
            "test_danger_recall": metrics["test"]["per_class"]["Danger"]["recall"],
            "test_log_loss": metrics["test"]["log_loss"],
            "test_rows": metrics["test"]["n"],
            "test_sessions": metadata["split_sessions"]["test"],
        },
        "verification": {
            "host_c_compile": parity["status"] == "PASS",
            "host_python_c_parity": parity,
            "esp32_cross_compile": False,
            "esp32_on_device_test": False,
            "real_sensor_validation": False,
        },
        "source_artifact": "models/final_model.joblib",
        "c_header": "firmware/esp32_gas_detection/gas_model.h",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    utils.write_json(config.MODELS_DIR / "gas_model.json", model_json)

    model_card = {
        "model_name": "Smart Gas AI MQ-5 classifier",
        "version": "1.0.0",
        "model_type": "Gaussian naive Bayes" if family == "GaussianNB" else "logistic regression",
        "selected_candidate": selected["selected_candidate"],
        "input_features": config.MODEL_FEATURES,
        "input_domain": "raw 12-bit ADC counts, 0..4095",
        "output_classes": config.CLASS_ORDER,
        "training_data": {
            "path": utils.relative_to_root(config.DATASET_PATH),
            "sha256": metadata["source_sha256"],
            "rows": metadata["rows_total"],
            "sessions": metadata["dataset_profile"]["n_sessions"],
            "nature": "supplied synthetic/enclosed dataset",
            "train_sessions": metadata["split_sessions"]["train"],
            "validation_sessions": metadata["split_sessions"]["validation"],
            "test_sessions": metadata["split_sessions"]["test"],
        },
        "evaluation": {
            "protocol": (
                "leave-one-session-out CV on training sessions; grouped "
                "validation; one-time grouped test"
            ),
            "cv_macro_f1": selected["cv"]["mean_macro_f1"],
            "validation_macro_f1": selected["validation"]["macro_f1"],
            "test_accuracy": metrics["test"]["accuracy"],
            "test_macro_f1": metrics["test"]["macro_f1"],
            "test_danger_recall": metrics["test"]["per_class"]["Danger"]["recall"],
            "test_log_loss": metrics["test"]["log_loss"],
            "test_rows": metrics["test"]["n"],
            "test_sessions": metadata["split_sessions"]["test"],
        },
        "intended_use": [
            "educational and research prototyping of an MQ-5 gas-level classifier",
            "offline benchmarking of a compact classifier on the supplied dataset",
            "host-side reference implementation for embedded integration",
        ],
        "not_intended_for": [
            "life-safety, fire, toxic-gas or explosion protection",
            "regulatory or certified gas detection",
            "direct deployment on real hardware without sensor calibration and field validation",
            "inferring temperature or humidity (the model does not use them)",
        ],
        "limitations": [
            "Synthetic data; no real MQ-5 noise, drift, ageing or gas mixtures.",
            "Adjacent classes overlap in ADC space; errors occur near boundaries.",
            "Only three held-out sessions; session-to-session uncertainty is coarse.",
            "Probabilities are not calibrated on real data.",
            "ESP32 compatibility has been assessed only by host compilation and parity testing.",
            "No on-device latency, flash, RAM or thermal measurement has been performed.",
        ],
        "deployment_status": {
            "ready_for_esp32_deployment": False,
            "reason": (
                "Host-verified C export only; no ESP32 cross-compilation, "
                "on-device test or real-sensor validation."
            ),
            "required_before_deployment": [
                "cross-compile for the target ESP32 toolchain",
                "measure on-device latency, flash and RAM",
                "calibrate the MQ-5 ADC path and warm-up behaviour",
                "collect labelled real-sensor data across sessions and environments",
                "re-evaluate with session-grouped or time-based splits",
                "define and test fail-safe alarm behaviour independent of the ML model",
            ],
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    utils.write_json(config.MODELS_DIR / "model_card.json", model_card)

    print(
        f"[export] family={family}, parity={parity['class_matches']}/"
        f"{parity['cases']}, status={parity['status']}, "
        f"max|dp|={parity['max_probability_abs_diff']}"
    )
    print("[export] wrote firmware/esp32_gas_detection/gas_model.h")
    print("[export] wrote models/gas_model.json, models/model_card.json")
    print("[export] wrote results/metrics/embedded_parity.txt")


if __name__ == "__main__":
    main()
