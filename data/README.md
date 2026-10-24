# Data

## Layout

| Path | Purpose |
|---|---|
| `raw/smart_gas_ai_enclosed_synthetic_dataset.csv` | Immutable supplied dataset, protected by SHA-256 |
| `processed/train.csv` | 12 training sessions, 2,376 rows |
| `processed/validation.csv` | 3 validation sessions, 594 rows |
| `processed/test.csv` | 3 held-out test sessions, 594 rows |
| `processed/split_manifest.json` | Seed, checksum, session assignment and row counts |
| `processed/metadata.json` | Dataset audit, feature ranges and class counts |
| `processed/dataset_summary.txt` | Human-readable preparation report |
| `processed/train_oof_predictions.csv` | Out-of-fold CV predictions |
| `processed/validation_predictions.csv` | Validation predictions |
| `processed/test_predictions.csv` | Test predictions |
| `sample/sample_readings.csv` | 24-row documentation sample from `ENC_01` |

## Rules

- Never edit the raw CSV. Every pipeline stage verifies its checksum and aborts on
  a mismatch.
- Regenerate processed files with `python -m src.data_preprocessing`; do not edit
  them by hand.
- The model uses `mq5_adc` only. Temperature and humidity are context features.

See [docs/dataset.md](../docs/dataset.md) for the full schema, statistics and
split rationale.
