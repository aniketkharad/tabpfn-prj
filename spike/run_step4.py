import os
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

# Set macOS thread limit to avoid libomp conflict
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load TABPFN_TOKEN from .env without printing secrets
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("TABPFN_TOKEN="):
            os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()

import tabpfn_client
from tabpfn_client import init, get_api_usage
from tabpfn_rel import PredictiveContext, PredictiveQuery, PredictiveQuerySpec

print("=== STEP 4: Official Olist Seller Churn Spike ===")

init()
initial_usage_str = str(get_api_usage())
print(f"Initial API Usage: {initial_usage_str}")

task_path = Path("spike/task_churn.yaml")
data_dir = Path("data/olist")

spec = PredictiveQuerySpec.from_yaml(str(task_path), data_dir=str(data_dir))
context = PredictiveContext(spec)

print(f"Task type: {spec.task.task_type}")
print(f"Entity table: {spec.task.entity_table} (ID: {spec.task.entity_col})")
print(f"Target col: {spec.task.target_col}")

start_time = time.perf_counter()
print("\nFitting model: tabpfn-rel-client-latest (n_trials=0)...")
fitted = context.fit(model="tabpfn-rel-client-latest", n_trials=0, seed=0)
fit_time = time.perf_counter() - start_time
print(f"Fit completed in {fit_time:.2f} s")

predict_start = time.perf_counter()
query = PredictiveQuery(entities="all", at_timestamp="test_timestamp")
preds = fitted.predict(query)
predict_time = time.perf_counter() - predict_start
total_time = time.perf_counter() - start_time
print(f"Predict completed in {predict_time:.2f} s")
print(f"Total end-to-end time: {total_time:.2f} s")

final_usage_str = str(get_api_usage())
print(f"\nFinal API Usage: {final_usage_str}")

print("\n--- Output Inspection ---")
print(f"Output shape: {preds.shape}")
print(f"Columns: {list(preds.columns)}")
print(f"Prediction head:\n{preds.head()}")
print(f"\nProbability summary for '{spec.task.target_col}_pred':")
pred_series = preds[f"{spec.task.target_col}_pred"]
print(pred_series.describe())

# Compute held-out test labels and ROC-AUC
try:
    test_labels = context.compute_test_labels()
    print(f"\nHeld-out test labels count: {len(test_labels):,}")
    scored = test_labels.merge(
        preds,
        on=[spec.task.time_col, spec.task.entity_col],
        how="inner",
    )
    if len(scored) > 0 and scored[spec.task.target_col].nunique() > 1:
        auc = roc_auc_score(scored[spec.task.target_col], scored[f"{spec.task.target_col}_pred"])
        print(f"Held-out test ROC-AUC: {auc:.4f} (scored on {len(scored)} active test entities)")
    else:
        print(f"Could not compute ROC-AUC: test labels unique count is {scored[spec.task.target_col].nunique()}")
except Exception as e:
    print(f"Notice on computing test labels: {e}")

print("\nStep 4 run finished successfully.")
