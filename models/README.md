# Models

| Artifact | Used by | Purpose |
|---|---|---|
| `final_model.joblib` | `src.predict`, `app.app`, `src.evaluate`, `src.export_embedded` | Final `StandardScaler` + `GaussianNB` fitted on 15 non-test sessions |
| `selected_model_train.joblib` | `src.evaluate` | Model fitted on the 12 training sessions only, used for validation predictions |
| `selected_model.json` | Prediction and export | Selected candidate, features, CV/validation summary and final-fit metadata |
| `candidate_configs.json` | Training | Candidate definitions and CV protocol |
| `gas_model.json` | Firmware/reference | Human-readable model constants and equations |
| `model_card.json` | Documentation | Intended use, limitations and deployment status |
| `test_vectors.csv` | Parity testing | ADC values with Python class and probability references |

The deployed model is **`gaussian_nb_adc`**, not Random Forest or Logistic
Regression. See [docs/model.md](../docs/model.md).

Regenerate with:

```bash
python -m src.train
python -m src.evaluate
python -m src.export_embedded
```

Do not hand-edit `gas_model.h`; it is generated from `final_model.joblib` by
`src.export_embedded`.
