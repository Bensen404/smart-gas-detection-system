# Model Selection

## Protocol

- Training sessions: 12 (`ENC_04, ENC_05, ENC_06, ENC_07, ENC_08, ENC_10, ENC_11, ENC_12, ENC_14, ENC_16, ENC_17, ENC_18`)
- Validation sessions: 3 (`ENC_03, ENC_13, ENC_15`)
- Test sessions: 3 (`ENC_01, ENC_02, ENC_09`) — **not read by this script**
- Cross-validation: leave-one-session-out, 12 folds on the training sessions
- Primary criterion: mean fold macro-F1
- Tie-break (within 0.001): smaller serialized model
- Random seed: 42
- Preprocessing: `StandardScaler`, fitted inside each CV fold and on each fit partition

## Candidate comparison

| Candidate | Features | CV macro-F1 (mean +/- std) | Val macro-F1 | Val accuracy | Joblib bytes | Embedded suitability |
|---|---|---|---|---|---|---|
| logreg_adc | mq5_adc | 0.9506 +/- 0.0178 | 0.9512 | 0.9512 | 952 | high |
| gaussian_nb_adc **selected** | mq5_adc | 0.9498 +/- 0.0181 | 0.9512 | 0.9512 | 885 | high |
| qda_adc | mq5_adc | 0.9498 +/- 0.0181 | 0.9512 | 0.9512 | 950 | medium |
| extra_trees_200_adc | mq5_adc | 0.9489 +/- 0.0176 | 0.9512 | 0.9512 | 517,316 | low |
| knn15_adc | mq5_adc | 0.9485 +/- 0.0193 | 0.9529 | 0.9529 | 26,048 | low |
| svc_rbf_adc | mq5_adc | 0.9485 +/- 0.0190 | 0.9528 | 0.9529 | 5,360 | low |
| logreg_all_features | mq5_adc+temperature_c+humidity_rh_pct | 0.9481 +/- 0.0178 | 0.9512 | 0.9512 | 1,088 | high |
| decision_tree_depth3_adc | mq5_adc | 0.9468 +/- 0.0163 | 0.9495 | 0.9495 | 1,726 | high |
| hist_gradient_boosting_adc | mq5_adc | 0.9461 +/- 0.0157 | 0.9529 | 0.9529 | 331,728 | low |
| random_forest_200_adc | mq5_adc | 0.9398 +/- 0.0158 | 0.9461 | 0.9461 | 570,882 | low |
| dummy_most_frequent | mq5_adc | 0.1667 +/- 0.0000 | 0.1667 | 0.3333 | 769 | n/a (baseline) |

## Selected model

**gaussian_nb_adc** was selected with a mean CV macro-F1 of
0.9498 (standard deviation
0.0181) and a validation macro-F1 of
0.9512.

- Family: `GaussianNB`
- Input features: `mq5_adc`
- Parameters: `{}`
- Final fit: 2,970 rows from 15 train + validation sessions
- Artifact: `models/final_model.joblib`

The all-feature logistic regression was evaluated under the same grouped CV.
Temperature and humidity did not improve CV macro-F1 in this dataset, so the
ADC-only variant was retained. This removes the DHT sensor from the inference
dependency chain and reduces the embedded export to one input.

## Test isolation

The held-out test sessions were not used to choose the model, tune
hyper-parameters, select features or learn thresholds. They are evaluated once
in `src/evaluate.py`.
