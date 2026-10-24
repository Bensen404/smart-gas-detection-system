# Tests

The suite uses the Python standard library `unittest` runner. It needs the
processed data, model artifacts and generated header to be present.

```bash
python -m unittest discover -s tests -v
```

## Coverage

| File | What it checks |
|---|---|
| `test_preprocessing.py` | Raw checksum, schema, shape, missing values, duplicates, class balance, sessions, split sizes/isolation/balance, ADC validation and dataset validation errors |
| `test_prediction.py` | Feature contract, class order, probability shape/sum, confidence, context handling, invalid ADC/humidity rejection, exported C constants, Python/C probability parity and parity log |
| `test_dashboard.py` | Dashboard HTML, `/api/info`, `/api/predict`, invalid/missing ADC, unknown route and POST rejection |

Expected result on a clean checkout:

```text
Ran 29 tests
OK
```

If the firmware header is missing, the export-related tests are skipped with a
message pointing to `python -m src.export_embedded`. If the parity log is absent
or was not run, the parity-log test is skipped.

The tests do not retrain the model and do not modify the raw dataset.
