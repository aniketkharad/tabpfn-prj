import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

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

def compute_free_baselines(context, test_labels, target_col, entity_col):
    """Compute constant-global and constant-per-entity baselines from training data."""
    train_df = context.task.get_table("train", mask_input_cols=False).df
    global_rate = float(train_df[target_col].mean())
    
    # Per-entity historical rate
    entity_rates = train_df.groupby(entity_col)[target_col].mean().to_dict()
    
    # Score on test entities
    y_true = test_labels[target_col].to_numpy()
    
    # 1. Global constant
    global_preds = np.full(len(test_labels), global_rate)
    
    # 2. Per-entity constant
    id_map = getattr(context._source._dataset, "pkey_maps", {}).get(context.task.entity_table)
    # test_labels might have original entity IDs or reindexed IDs depending on call
    per_entity_preds = []
    for entity in test_labels[entity_col]:
        per_entity_preds.append(entity_rates.get(entity, global_rate))
    per_entity_preds = np.array(per_entity_preds)
    
    auc_global = roc_auc_score(y_true, global_preds) if len(np.unique(y_true)) > 1 else 0.5
    auc_per_entity = roc_auc_score(y_true, per_entity_preds) if len(np.unique(y_true)) > 1 else 0.5
    
    return {
        "global_rate": global_rate,
        "auc_global": auc_global,
        "auc_per_entity": auc_per_entity,
    }

def run_task(task_yaml_path: str, task_name: str):
    print(f"\n=======================================================")
    print(f"RUNNING SPIKE 2: {task_name}")
    print(f"Task file: {task_yaml_path}")
    print(f"=======================================================")
    
    usage_before = str(get_api_usage())
    print(f"API Usage Before: {usage_before}")
    
    data_dir = Path("data/olist")
    spec = PredictiveQuerySpec.from_yaml(task_yaml_path, data_dir=str(data_dir))
    context = PredictiveContext(spec)
    
    target_col = spec.task.target_col
    entity_col = spec.task.entity_col
    time_col = spec.task.time_col
    
    print(f"Entity: {spec.task.entity_table} ({entity_col}), Target: {target_col}")
    
    # Check splits
    for split in ("train", "val", "test"):
        df = context.task.get_table(split, mask_input_cols=False).df
        pos_rate = df[target_col].mean() if len(df) > 0 else 0
        print(f"  Split '{split}': {len(df):,} rows, {df['timestamp'].nunique():>} anchors, positive rate {pos_rate:.1%}")
    
    # Fit TabPFN-Rel
    start_time = time.perf_counter()
    print("\nFitting TabPFN-Rel (tabpfn-rel-client-latest, n_trials=0)...")
    fitted = context.fit(model="tabpfn-rel-client-latest", n_trials=0, seed=0)
    fit_duration = time.perf_counter() - start_time
    print(f"Fit completed in {fit_duration:.2f} s")
    
    # Predict TabPFN-Rel
    pred_start = time.perf_counter()
    query = PredictiveQuery(entities="all", at_timestamp="test_timestamp")
    preds = fitted.predict(query)
    pred_duration = time.perf_counter() - pred_start
    total_duration = time.perf_counter() - start_time
    print(f"Predict completed in {pred_duration:.2f} s")
    print(f"Total time: {total_duration:.2f} s")
    
    usage_after = str(get_api_usage())
    print(f"API Usage After: {usage_after}")
    
    pred_col = f"{target_col}_pred"
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Prediction head:\n{preds.head(3)}")
    print(f"\nProbabilities stats:\n{preds[pred_col].describe()}")
    
    # Evaluate held-out test labels
    test_labels = context.compute_test_labels()
    print(f"\nHeld-out test labels count: {len(test_labels):,}")
    
    scored = test_labels.merge(
        preds,
        on=[time_col, entity_col],
        how="inner",
    )
    
    tabpfn_auc = roc_auc_score(scored[target_col], scored[pred_col])
    print(f"\n--- MODEL EVALUATION (Test ROC-AUC) ---")
    print(f"TabPFN-Rel:       {tabpfn_auc:.4f}")
    
    # Baselines
    baseline_res = compute_free_baselines(context, scored, target_col, entity_col)
    print(f"Dummy Per-Entity: {baseline_res['auc_per_entity']:.4f}")
    print(f"Dummy Global:     {baseline_res['auc_global']:.4f} (base rate: {baseline_res['global_rate']:.3f})")
    
    return {
        "task": task_name,
        "total_time": total_duration,
        "tabpfn_auc": tabpfn_auc,
        "per_entity_auc": baseline_res["auc_per_entity"],
        "global_auc": baseline_res["auc_global"],
        "shape": preds.shape,
    }

if __name__ == "__main__":
    init()
    results = []
    
    # Run Option A: Seller Bad Review Risk
    res_a = run_task("spike/task_option_a_bad_reviews.yaml", "Option A: Seller Bad Review Risk")
    results.append(res_a)
    
    # Run Option B: High-Volume Seller Growth Surge
    res_b = run_task("spike/task_option_b_high_volume.yaml", "Option B: High-Volume Seller Growth Surge")
    results.append(res_b)
    
    print("\n\n=======================================================")
    print("SPIKE 2 FINAL SUMMARY")
    print("=======================================================")
    for r in results:
        print(f"\nTask: {r['task']}")
        print(f"  Shape: {r['shape']}, Wall time: {r['total_time']:.1f} s")
        print(f"  TabPFN-Rel ROC-AUC:   {r['tabpfn_auc']:.4f}")
        print(f"  Baseline Per-Entity:  {r['per_entity_auc']:.4f}")
        print(f"  Baseline Global:      {r['global_auc']:.4f}")
