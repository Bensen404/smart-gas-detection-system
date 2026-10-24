# Source

| Module | Purpose |
|---|---|
| `config.py` | Paths, dataset contract, checksum, split assignment, candidates, seed and plot colours |
| `utils.py` | Integrity checks, JSON helpers, metrics, bootstrap and environment reporting |
| `data_preprocessing.py` | Dataset validation, session-grouped split, processed outputs and sample table |
| `train.py` | Candidate comparison with leave-one-session-out CV, selection and final fit |
| `evaluate.py` | One-time test evaluation, baseline comparison, calibration and plots |
| `export_embedded.py` | Header-only C export, model card, test vectors and host parity test |
| `predict.py` | Reusable inference API and command-line predictor |
| `__init__.py` | Package marker |

## Commands

Run from the repository root:

```bash
python -m src.data_preprocessing
python -m src.train
python -m src.evaluate
python -m src.export_embedded
python -m src.predict --adc 1500
```

## Design notes

- The raw CSV is immutable and checksum-verified by every pipeline stage.
- The test split is read only by `evaluate.py`, after model selection and the
  final fit.
- The model contract is one feature, `mq5_adc`, with class order
  `Normal`, `Warning`, `Danger`.
- `StandardScaler` is fitted inside every CV fold, not on the full dataset.
- `export_embedded.py` generates `gas_model.h`; never edit that header by hand.
- Candidate definitions live in `config.py`; adding a candidate automatically
  includes it in the comparison and selection report.

See [docs/architecture.md](../docs/architecture.md) for the full pipeline.
