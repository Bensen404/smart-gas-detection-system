# Model

## Deployed model

| Item | Value |
|---|---|
| Candidate | `gaussian_nb_adc` |
| Family | `GaussianNB` |
| Feature | `mq5_adc` only |
| Preprocessing | `StandardScaler` |
| Class order | `Normal`, `Warning`, `Danger` |
| Random seed | 42 |
| Final fit | 2,970 rows from 15 sessions (12 train + 3 validation) |
| Test fit | Not refit on test data |
| Artifact | `models/final_model.joblib` |
| C header | `firmware/esp32_gas_detection/gas_model.h` |

The model consumes a raw 12-bit ADC value and returns three probabilities.
Temperature and humidity are not inputs.

## Candidate comparison

Candidates were compared with leave-one-session-out cross-validation on the 12
training sessions. The test sessions were not used for selection.

| Candidate | Features | CV macro F1 (mean) | Serialized bytes | Embedded suitability |
|---|---|---:|---:|---|
| `logreg_adc` | ADC | 0.9506 | 952 | high |
| `gaussian_nb_adc` | ADC | 0.9498 | 885 | high |
| `qda_adc` | ADC | 0.9498 | 950 | medium |
| `extra_trees_200_adc` | ADC | 0.9489 | 517,316 | low |
| `knn15_adc` | ADC | 0.9485 | 26,048 | low |
| `svc_rbf_adc` | ADC | 0.9485 | 5,360 | low |
| `logreg_all_features` | ADC + temperature + humidity | 0.9481 | 1,088 | high |
| `decision_tree_depth3_adc` | ADC | 0.9468 | 1,726 | high |
| `hist_gradient_boosting_adc` | ADC | 0.9461 | 331,728 | low |
| `random_forest_200_adc` | ADC | 0.9398 | 570,882 | low |
| `dummy_most_frequent` | ADC | 0.1667 | 769 | n/a |

The selection rule was fixed before evaluation:

1. Highest mean grouped CV macro F1.
2. If candidates are within `0.001`, choose the smallest serialized model.

Logistic Regression had the highest CV score (0.9506), but GaussianNB was within
the tie tolerance (difference 0.000822) and serialized to 885 bytes versus 952
bytes. GaussianNB therefore won the tie-break. QDA matched GaussianNB's CV score
but was larger and rated less suitable for the embedded target.

**Random Forest and Logistic Regression were evaluated but are not deployed.**
Only the GaussianNB pipeline is exported to the ESP32. The separate legacy
4-class `Smart-Gas-AI` project uses a different dataset and model and is not part
of this repository.

## Evaluation protocol

| Partition | Sessions | Rows | Use |
|---|---:|---:|---|
| Training CV | 12 | 2,376 | Leave-one-session-out model comparison |
| Validation | 3 | 594 | Check the selected model before the final fit |
| Test | 3 | 594 | One-time held-out evaluation after fitting on train + validation |

The validation split is used to check the selected candidate. The final model is
then fitted on all 15 non-test sessions. The test split is touched once, by
`src.evaluate`, after that fit.

## Held-out test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9394 |
| Accuracy 95% Wilson CI | [0.9172, 0.9559] |
| Balanced accuracy | 0.9394 |
| Macro precision | 0.9394 |
| Macro recall | 0.9394 |
| Macro F1 | 0.9393 |
| Weighted F1 | 0.9393 |
| Cohen's kappa | 0.9091 |
| Matthews correlation coefficient | 0.9092 |
| Log loss | 0.1484 |
| Multiclass Brier score | 0.0895 |
| Top-2 accuracy | 1.0000 |
| Errors | 36 / 594 |

Session-group bootstrap 95% intervals (only 3 test sessions, so these are
coarse):

- Accuracy: [0.9192, 0.9646]
- Macro F1: [0.9192, 0.9645]

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9366 | 0.9697 | 0.9529 | 198 |
| Warning | 0.9175 | 0.8990 | 0.9082 | 198 |
| Danger | 0.9641 | 0.9495 | 0.9567 | 198 |

Confusion matrix (rows = true, columns = predicted):

| True \ Predicted | Normal | Warning | Danger |
|---|---:|---:|---:|
| Normal | 192 | 6 | 0 |
| Warning | 13 | 178 | 7 |
| Danger | 0 | 10 | 188 |

Warning is the hardest class: its recall is 0.8990, and most errors are
Normal↔Warning or Warning↔Danger confusions near the class boundaries. There are
no Danger→Normal errors in the held-out test split.

Per-session test results:

| Session | Rows | Accuracy | Macro F1 | Danger recall | Errors | Danger→Normal |
|---|---:|---:|---:|---:|---:|---:|
| `ENC_01` | 198 | 0.9343 | 0.9342 | 0.9394 | 13 | 0 |
| `ENC_02` | 198 | 0.9192 | 0.9192 | 0.9091 | 16 | 0 |
| `ENC_09` | 198 | 0.9646 | 0.9645 | 1.0000 | 7 | 0 |

## Cross-validation and validation

| Evaluation | Accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| Grouped CV (out-of-fold) | 0.9499 | 0.9498 | 0.1408 |
| Validation | 0.9512 | 0.9512 | 0.1367 |

The CV macro-F1 standard deviation across the 12 folds is 0.0181, which is larger
than the difference between the top candidates. This is why the selection rule
uses a tie tolerance rather than treating small ranking differences as meaningful.

## Rule-based baseline

A two-threshold rule was fitted on the training split only:

```text
Normal  if mq5_adc <= 1300
Warning if mq5_adc <= 2200
Danger  otherwise
```

| Split / system | Accuracy | Macro F1 |
|---|---:|---:|
| Rule baseline, validation | 0.9512 | 0.9512 |
| Rule baseline, test | 0.9411 | 0.9409 |
| GaussianNB, test | 0.9394 | 0.9393 |

McNemar exact test on the test split:

| Model only correct | Baseline only correct | Discordant pairs | Exact p |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 1.0000 |

The ML model does **not** show a statistically significant accuracy advantage
over a simple threshold rule on this dataset. The value of the ML model here is
that it produces smooth probabilities and is a compact, complete embedded
example, not that it beats a threshold by a meaningful margin.

## Calibration

| Metric | Test value |
|---|---:|
| Top-label ECE | 0.0117 |
| Top-label MCE | 0.2574 |

Probabilities were measured but not recalibrated. Any alarm threshold that uses
the probabilities must be validated on real sensor data; a low average ECE does
not guarantee safety at the tail.

## Embedded export

`src/export_embedded.py` converts the fitted pipeline into a header-only C99
implementation. The exported constants are:

| Constant | Meaning |
|---|---|
| `GAS_MODEL_MEAN` | StandardScaler mean, float32 |
| `GAS_MODEL_SCALE` | StandardScaler scale, float32 |
| `GAS_MODEL_THETA` | GaussianNB class means in scaled space |
| `GAS_MODEL_INV_VAR` | Inverse class variances |
| `GAS_MODEL_LOG_CONST` | Log prior minus the Gaussian normalisation term |
| `GAS_MODEL_CLASS_PRIOR` | Class priors |

Parameter footprint: **56 bytes** of float32 constants. The C functions perform
one standardisation, three quadratic scores and a numerically stable softmax.

Host verification:

| Check | Result |
|---|---|
| Compiler | MinGW GCC 6.3.0 |
| Cases | 165 ADC values |
| Class matches | 165 / 165 |
| Max probability difference | 2.38e-07 |
| Tolerance | 1e-4 |
| Status | PASS |

This is a **host** parity test. The header has not been cross-compiled for the
ESP32 toolchain, flashed to a device or exercised with a real sensor.

## Deployment status

`models/model_card.json` records `ready_for_esp32_deployment: false`. Before any
real deployment:

- cross-compile with the target ESP32 toolchain;
- measure flash, RAM and per-inference latency on the device;
- calibrate the MQ-5 warm-up, supply and ADC path;
- collect labelled real-sensor data across sessions and environments;
- re-evaluate with session-grouped or time-based splits;
- add an independent fail-safe alarm path that does not depend on the model.
