# Notebook

`exploration.ipynb` is an executed exploratory notebook covering:

1. dataset checksum and schema checks;
2. missing/duplicate/class/session summaries;
3. ADC distributions and class overlap;
4. temperature/humidity context and session effects;
5. session-grouped split integrity;
6. frozen-model metrics and confusion matrix;
7. GaussianNB parameters exported to firmware;
8. rule-based baseline and McNemar result;
9. probability curves over the ADC range;
10. takeaways and limitations.

The checked-in notebook includes the actual outputs and six generated figures.
It reads the raw CSV, processed splits and saved metrics; it does not retrain the
model.

## Run interactively

```bash
pip install jupyterlab
jupyter lab notebooks/exploration.ipynb
```

Run it from the repository root or from this directory. The setup cell locates
the project root automatically.

## Refresh outputs

If the data or metrics change, rerun all cells in Jupyter and save the notebook.
Do not hand-edit the JSON.
