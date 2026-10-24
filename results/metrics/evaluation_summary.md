# Evaluation Summary

## Frozen model

| Item | Value |
|---|---|
| Selected candidate | `gaussian_nb_adc` |
| Family | `GaussianNB` |
| Input features | `mq5_adc` |
| Preprocessing | StandardScaler fitted on the training partition |
| Training rows (validation fit) | 2,376 |
| Final-fit rows (test fit) | 2,970 from 15 sessions |
| Test rows | 594 from 3 sessions |
| Test sessions | ENC_01, ENC_02, ENC_09 |
| Random seed | 42 |
| Test used for tuning | No |

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

Session-group bootstrap 95% intervals (only 3 test
sessions, so these intervals are coarse):

- Accuracy: [0.9192, 0.9646]
- Macro F1: [0.9192, 0.9645]

Per-class test results:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.9366 | 0.9697 | 0.9529 | 198 |
| Warning | 0.9175 | 0.8990 | 0.9082 | 198 |
| Danger | 0.9641 | 0.9495 | 0.9567 | 198 |

Confusion matrix (rows = true, columns = predicted):

| True \ Predicted | Normal | Warning | Danger |
|---|---:|---:|---:|
| Normal | 192 | 6 | 0 |
| Warning | 13 | 178 | 7 |
| Danger | 0 | 10 | 188 |

## Per-session test results

| Session | Rows | Accuracy | Macro F1 | Danger recall | Errors | Danger->Normal |
|---|---|---|---|---|---|---|
| ENC_01 | 198 | 0.9343 | 0.9342 | 0.9394 | 13 | 0 |
| ENC_02 | 198 | 0.9192 | 0.9192 | 0.9091 | 16 | 0 |
| ENC_09 | 198 | 0.9646 | 0.9645 | 1.0000 | 7 | 0 |

## Rule-based baseline

Thresholds learned on the training split only: Normal <= 1300 ADC,
Warning <= 2200 ADC, otherwise Danger.

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Validation | 0.9512 | 0.9512 |
| Test | 0.9411 | 0.9409 |

McNemar exact test on the test split: model-only correct
0, baseline-only correct
1, p =
1.0000. The rule is not
significantly different from the model at this sample size, so no accuracy
advantage over a simple threshold rule is claimed.

## Calibration

Top-label ECE = 0.0117, MCE =
0.2574. Probabilities were measured but not
recalibrated. Any alarm threshold that uses probabilities must be validated on
real sensor data.

## Artifacts

- `results/metrics/metrics.json`
- `results/metrics/classification_report_test.txt`
- `results/metrics/classification_report_validation.txt`
- `results/metrics/model_comparison.csv`
- `results/metrics/confusion_matrix_test.csv`
- `results/metrics/per_session_test_metrics.csv`
- `results/plots/*.png`

## Interpretation limits

The dataset is synthetic. These metrics describe the supplied data-generating
process, not a physical MQ-5 installation. Three held-out sessions provide only
a coarse estimate of session-to-session variation. See `docs/limitations.md`.
