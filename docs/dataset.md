# Dataset

## Provenance

The project uses the supplied file:

```text
data/raw/smart_gas_ai_enclosed_synthetic_dataset.csv
```

It is described in the repository as a **synthetic/enclosed** dataset. The
original collection method, generator, license and redistribution terms could
not be verified from the files available. It is included so the reported results
can be reproduced; confirm permission before redistributing it.

The separate legacy `Smart-Gas-AI` project uses a different 4-class dataset and
a Random Forest model. It is not the source of this dataset or model.

## Integrity

| Item | Value |
|---|---|
| Rows | 3,564 |
| Columns | 5 |
| Sessions | 18 |
| File size | 105,633 bytes |
| SHA-256 | `82BFC7557E247D06E54C7B81AD5F2A9A294C5A44FD006274D451C6D99F1F01C7` |

`src/config.py` pins the checksum. Every pipeline stage calls
`utils.verify_raw_dataset()` and aborts if the file changes.

## Schema

| Column | Example | Description |
|---|---|---|
| `session_id` | `ENC_01` | Enclosure/session identifier, `ENC_01`..`ENC_18` |
| `mq5_adc` | `1794` | Raw 12-bit ADC reading, 0..4095 |
| `temperature_c` | `27.5` | Temperature in °C, context only |
| `humidity_rh_pct` | `54.5` | Relative humidity in %, context only |
| `label` | `Warning` | `Normal`, `Warning` or `Danger` |

Only `mq5_adc` is passed to the classifier. Temperature and humidity are kept
for analysis and are not used by the deployed firmware.

## Quality checks

| Check | Result |
|---|---|
| Missing cells | 0 |
| Exact duplicate rows | 0 |
| Duplicate feature rows | 0 |
| Conflicting-label groups | 0 |
| Rows removed | 0 |
| Rows imputed | 0 |
| Labels changed | 0 |

Class distribution:

| Class | Rows | Share |
|---|---:|---:|
| Normal | 1,188 | 33.33% |
| Warning | 1,188 | 33.33% |
| Danger | 1,188 | 33.33% |

## Feature statistics

| Feature | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| `mq5_adc` | 197.00 | 3,602.00 | 1,793.84 | 778.02 |
| `temperature_c` | 15.00 | 44.70 | 27.50 | 6.28 |
| `humidity_rh_pct` | 15.00 | 95.00 | 54.48 | 18.48 |

Mean ADC by class:

| Class | Mean ADC |
|---|---:|
| Normal | 916.14 |
| Warning | 1,757.86 |
| Danger | 2,707.52 |

The class distributions overlap at the boundaries. Temperature and humidity are
nearly uncorrelated with ADC in this synthetic dataset (Pearson correlation
0.008 and 0.001 respectively), which is consistent with the decision to deploy
an ADC-only model.

## Session-grouped split

The split is generated with `GroupShuffleSplit` and seed 42, then pinned in
`src/config.py`.

| Partition | Sessions | Rows | Rows per class |
|---|---:|---:|---:|
| Training | 12 | 2,376 | 792 each |
| Validation | 3 | 594 | 198 each |
| Test | 3 | 594 | 198 each |

| Partition | Session IDs |
|---|---|
| Training | `ENC_04`, `ENC_05`, `ENC_06`, `ENC_07`, `ENC_08`, `ENC_10`, `ENC_11`, `ENC_12`, `ENC_14`, `ENC_16`, `ENC_17`, `ENC_18` |
| Validation | `ENC_03`, `ENC_13`, `ENC_15` |
| Test | `ENC_01`, `ENC_02`, `ENC_09` |

No session overlaps between partitions. Grouping by session prevents rows from
the same enclosure run from leaking into both training and evaluation.

## Sample rows

`data/sample/sample_readings.csv` contains 24 rows from `ENC_01`, spanning all
three classes. It is a convenience sample for documentation and smoke tests, not
an additional training set.

```text
session_id,mq5_adc,temperature_c,humidity_rh_pct,label
ENC_01,2394,28.5,88.9,Danger
ENC_01,2572,26.1,89.9,Danger
ENC_01,859,30.8,86.0,Normal
ENC_01,1431,26.3,95.0,Warning
```

## Generated data files

| File | Purpose |
|---|---|
| `data/processed/train.csv` | 12 training sessions |
| `data/processed/validation.csv` | 3 validation sessions |
| `data/processed/test.csv` | 3 held-out test sessions |
| `data/processed/split_manifest.json` | Seed, checksum, sessions and row counts |
| `data/processed/metadata.json` | Feature ranges, class counts and dataset audit |
| `data/processed/dataset_summary.txt` | Human-readable preparation report |
| `data/processed/train_oof_predictions.csv` | Out-of-fold CV predictions |
| `data/processed/validation_predictions.csv` | Validation predictions |
| `data/processed/test_predictions.csv` | Test predictions |

Regenerate them with:

```bash
python -m src.data_preprocessing
```

## Caveats

- The data is synthetic. It does not contain real MQ-5 warm-up transients,
  sensor drift, ageing, unit-to-unit variation, gas mixtures or environmental
  interference.
- The balanced class design is convenient for evaluation but does not reflect the
  prior probability of a leak in a real installation.
- Only three sessions are held out for test, so session-to-session uncertainty
  is estimated coarsely.
