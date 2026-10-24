"""Reusable inference for the Smart Gas AI classifier.

The deployed model consumes exactly one feature, ``mq5_adc``.  Temperature and
humidity are accepted as optional context and are returned with the result, but
they are not passed to the model.

Examples
--------
    python -m src.predict --adc 1500
    python -m src.predict --adc 2500 --temperature 28.0 --humidity 70 --json
    python -m src.predict --interactive
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, utils
    from src.data_preprocessing import validate_adc
else:
    from . import config, utils
    from .data_preprocessing import validate_adc


def load_artifacts(
    model_path: Path | None = None,
    selected_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    model_path = model_path or config.MODELS_DIR / "final_model.joblib"
    selected_path = selected_path or config.MODELS_DIR / "selected_model.json"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run `python -m src.train` first."
        )
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {selected_path}. "
            "Run `python -m src.train` first."
        )

    model = joblib.load(model_path)
    selected = utils.read_json(selected_path)
    if selected.get("features") != config.MODEL_FEATURES:
        raise RuntimeError(
            "The prediction contract expects "
            f"{config.MODEL_FEATURES}, but the saved model uses "
            f"{selected.get('features')}."
        )
    return model, selected


def _validate_context(
    temperature_c: float | None, humidity_rh_pct: float | None
) -> dict[str, float | None]:
    context: dict[str, float | None] = {
        "temperature_c": None,
        "humidity_rh_pct": None,
    }
    if temperature_c is not None:
        try:
            temperature = float(temperature_c)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"temperature_c must be numeric, got {temperature_c!r}"
            ) from error
        if not math.isfinite(temperature):
            raise ValueError("temperature_c must be finite")
        context["temperature_c"] = temperature

    if humidity_rh_pct is not None:
        try:
            humidity = float(humidity_rh_pct)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"humidity_rh_pct must be numeric, got {humidity_rh_pct!r}"
            ) from error
        if not math.isfinite(humidity):
            raise ValueError("humidity_rh_pct must be finite")
        if humidity < config.HUMIDITY_MIN or humidity > config.HUMIDITY_MAX:
            raise ValueError("humidity_rh_pct must be within 0..100")
        context["humidity_rh_pct"] = humidity

    return context


def predict_reading(
    mq5_adc: float,
    temperature_c: float | None = None,
    humidity_rh_pct: float | None = None,
    model: Any | None = None,
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one sensor reading.

    Parameters
    ----------
    mq5_adc:
        Raw 12-bit ADC value in the range 0..4095.
    temperature_c, humidity_rh_pct:
        Optional context. They are validated but are not model features.
    model, selected:
        Optional pre-loaded artifacts, useful for a dashboard or batch loop.
    """
    if model is None or selected is None:
        model, selected = load_artifacts()

    adc = validate_adc(mq5_adc)
    context = _validate_context(temperature_c, humidity_rh_pct)

    feature_frame = pd.DataFrame(
        [{name: adc for name in config.MODEL_FEATURES}],
        columns=config.MODEL_FEATURES,
    )
    probabilities = model.predict_proba(feature_frame)[0]
    model_classes = list(model.named_steps["model"].classes_)
    reorder = [model_classes.index(name) for name in config.CLASS_ORDER]
    probabilities = np.asarray(probabilities, dtype=float)[reorder]
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise RuntimeError("Model probabilities do not sum to 1")

    label = str(model.predict(feature_frame)[0])
    if label not in config.CLASS_TO_INDEX:
        raise RuntimeError(f"Unexpected predicted label: {label!r}")

    probability_map = {
        name: float(probabilities[index])
        for index, name in enumerate(config.CLASS_ORDER)
    }
    return {
        "label": label,
        "class_index": config.CLASS_TO_INDEX[label],
        "confidence": float(probabilities.max()),
        "probabilities": probability_map,
        "mq5_adc": adc,
        "context": context,
        "context_used_by_model": False,
        "model": {
            "candidate": selected.get("selected_candidate"),
            "family": selected.get("family"),
            "features": selected.get("features"),
            "random_seed": selected.get("random_seed"),
        },
    }


def format_result(result: dict[str, Any]) -> str:
    probabilities = result["probabilities"]
    lines = [
        f"MQ-5 ADC       : {result['mq5_adc']:.0f}",
        f"Predicted state: {result['label']}",
        f"Confidence     : {result['confidence']:.4f}",
        "Probabilities  : "
        + ", ".join(
            f"{name}={probabilities[name]:.4f}"
            for name in config.CLASS_ORDER
        ),
    ]
    context = result["context"]
    if context["temperature_c"] is not None or context["humidity_rh_pct"] is not None:
        lines.append(
            "Context        : "
            f"temperature_c={context['temperature_c']}, "
            f"humidity_rh_pct={context['humidity_rh_pct']} "
            "(not used by the model)"
        )
    return "\n".join(lines)


def _interactive(model: Any, selected: dict[str, Any]) -> int:
    print("Smart Gas AI interactive predictor")
    print("Enter an ADC value from 0 to 4095. Press Enter on a blank line to quit.")
    while True:
        try:
            raw = input("mq5_adc> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            return 0
        try:
            result = predict_reading(raw, model=model, selected=selected)
        except ValueError as error:
            print(f"Invalid input: {error}")
            continue
        print(format_result(result))
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify an MQ-5 reading as Normal, Warning or Danger."
    )
    parser.add_argument(
        "--adc",
        type=float,
        help="raw 12-bit MQ-5 ADC value (0..4095)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="optional temperature in Celsius (context only)",
    )
    parser.add_argument(
        "--humidity",
        type=float,
        help="optional relative humidity in percent (context only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the result as JSON",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="prompt for ADC values until a blank line is entered",
    )
    args = parser.parse_args(argv)

    try:
        model, selected = load_artifacts()
        if args.interactive or args.adc is None:
            return _interactive(model, selected)
        result = predict_reading(
            args.adc,
            temperature_c=args.temperature,
            humidity_rh_pct=args.humidity,
            model=model,
            selected=selected,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
