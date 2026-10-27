# Third Model Run

This folder documents and evaluates the model trained from the fresh cleaned
dataset in `dataset_analysis/3rd_Model_Dataset`.

Run the evaluator from the project root with:

```powershell
.\.venv\Scripts\python.exe third_run\evaluate_third_model.py
```

The evaluator uses the untouched test split in `training/third_model_data`
and writes the results to `THIRD_MODEL_METRICS.txt`.