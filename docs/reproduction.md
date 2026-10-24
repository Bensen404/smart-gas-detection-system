# Reproduction

## Environment

| Item | Checked-in run |
|---|---|
| Python | 3.14.4 |
| Platform | Windows 11 (`Windows-11-10.0.26300-SP0`) |
| Processor | Intel64 Family 6 Model 154 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| Matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| GPU | Not used |

`requirements.txt` gives compatible version ranges. The exact versions above are
also recorded in `results/metrics/metrics.json` under `runtime`.

## Setup

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The raw dataset must be present at:

```text
data/raw/smart_gas_ai_enclosed_synthetic_dataset.csv
```

Its SHA-256 must be:

```text
82BFC7557E247D06E54C7B81AD5F2A9A294C5A44FD006274D451C6D99F1F01C7
```

All pipeline stages verify this checksum and abort on a mismatch.

## Full pipeline

Run the four stages in order:

```bash
python -m src.data_preprocessing
python -m src.train
python -m src.evaluate
python -m src.export_embedded
```

### 1. Prepare

```bash
python -m src.data_preprocessing
```

Expected key output:

```text
[prepare] source: data/raw/smart_gas_ai_enclosed_synthetic_dataset.csv
[prepare] sha256: 82BFC7557E247D06E54C7B81AD5F2A9A294C5A44FD006274D451C6D99F1F01C7
[prepare] split sizes: {'train': 2376, 'validation': 594, 'test': 594}
[prepare] test sessions: ['ENC_01', 'ENC_02', 'ENC_09']
[prepare] raw SHA-256 re-check: PASS
```

Writes the processed splits, `metadata.json`, `split_manifest.json`,
`dataset_summary.txt` and the 24-row documentation sample.

### 2. Train

```bash
python -m src.train
```

The script runs 12-fold leave-one-session-out CV for every candidate in
`src/config.py`, selects a model using the documented tie rule, fits the final
model on the 12 training + 3 validation sessions, and writes the model artifacts
and comparison tables.

Expected final selection: `gaussian_nb_adc`.

The script does **not** read `data/processed/test.csv`.

### 3. Evaluate

```bash
python -m src.evaluate
```

Expected key output:

```text
[evaluate] selected model: gaussian_nb_adc
[evaluate] features: ['mq5_adc']
[evaluate] test rows: 594 from sessions ['ENC_01', 'ENC_02', 'ENC_09']
[evaluate] test accuracy=0.9394 macro-F1=0.9393
[evaluate] test Danger recall=0.9495, errors=36
```

This is the only stage that reads the held-out test split. It writes
`results/metrics/metrics.json`, reports, confusion matrices, per-session metrics,
calibration bins and all seven plots.

Small differences may appear if library versions differ, especially around
floating-point formatting and plot rendering. The split, seed and model
selection are deterministic.

### 4. Export

```bash
python -m src.export_embedded
```

Expected output with GCC available:

```text
[export] family=GaussianNB, parity=165/165, status=PASS, max|dp|=2.38e-07
[export] wrote firmware/esp32_gas_detection/gas_model.h
[export] wrote models/gas_model.json, models/model_card.json
[export] wrote results/metrics/embedded_parity.txt
```

`src.export_embedded` looks for `gcc` on `PATH`. If GCC is unavailable, it still
writes the header and model artifacts, records that the host parity test was not
run, and reports `reason: gcc not found on PATH`. The firmware header itself is
generated either way.

The generated header is written to:

```text
firmware/esp32_gas_detection/gas_model.h
```

Do not edit it by hand; rerun the export after retraining.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite has 29 tests:

- dataset shape, checksum, missing values, duplicates, classes and sessions;
- split sizes, class balance and session isolation;
- ADC validation and dataset validation errors;
- prediction shape, class order, probability sum and context handling;
- exported C constants against the fitted scikit-learn model;
- Python/C probability parity over the ADC grid;
- dashboard HTML, JSON API, invalid input and method handling.

All tests should pass on a clean checkout with the generated artifacts present.

## Prediction CLI

```bash
python -m src.predict --adc 1500
python -m src.predict --adc 2800 --json
python -m src.predict --interactive
```

Example text output:

```text
MQ-5 ADC       : 1500
Predicted state: Warning
Confidence     : 0.9797
Probabilities  : Normal=0.0190, Warning=0.9797, Danger=0.0013
```

The exact probabilities depend on the frozen model. Invalid inputs such as
`-1`, `4096`, `nan` or a non-numeric string exit with code 2.

## Dashboard

```bash
python app/app.py
```

Open <http://127.0.0.1:8000>. Use `--host` and `--port` to change the bind
address:

```bash
python app/app.py --host 127.0.0.1 --port 8080
```

The dashboard has no third-party web framework. It serves a local page and two
JSON endpoints; see `app/README.md`.

## Notebook

`notebooks/exploration.ipynb` is checked in with executed outputs. To rerun it
interactively, install Jupyter:

```bash
pip install jupyterlab
jupyter lab notebooks/exploration.ipynb
```

The notebook reads the raw data, processed splits and saved metrics. It does not
retrain the model.

## Determinism and artifacts

- Split seed: 42.
- Model seed: 42.
- Candidate order and selection rule are fixed in `src/config.py`.
- The raw dataset is immutable and checksum-verified.
- Timestamps in `metrics.json`, `gas_model.json` and `model_card.json` change on
  each run; metrics and model constants should not.
- The `.joblib` artifacts are small enough to commit. If you retrain, replace
  them together with the generated header, model card and metrics.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Raw dataset not found` | CSV missing or wrong path | Place the supplied CSV at `data/raw/smart_gas_ai_enclosed_synthetic_dataset.csv` |
| `Raw dataset integrity check failed` | CSV was modified | Restore the original file; do not edit the raw data |
| `Model artifact not found` | Train step not run | Run `python -m src.train` |
| `Firmware header missing` in tests | Export step not run | Run `python -m src.export_embedded` |
| `gcc not found on PATH` | No host C compiler | Install GCC or accept the skipped parity test; the header is still generated |
| Dashboard cannot bind | Port already in use | Use `--port 8080` or another free port |
| `ModuleNotFoundError: src` | Running from the wrong directory | Run commands from the repository root |
