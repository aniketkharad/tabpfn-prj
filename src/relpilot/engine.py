"""Engine module for RelPilot: temporal backtesting, metrics, baselines, and artifact generation."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from relpilot.gate import DEFAULT_MIN_PRECISION, DEFAULT_MIN_SUPPORT, find_act_threshold
from relpilot.workspace import Workspace

MAX_TRAIN_ENTITIES = 2000
MAX_PREDICT_ENTITIES = 500
DEFAULT_MODEL = "tabpfn-rel-client-latest"


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) and reliability bin table."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bins) - 1
    # Clip to valid index 0..n_bins-1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)
    reliability_bins: list[dict[str, Any]] = []

    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        if count == 0:
            reliability_bins.append(
                {
                    "bin": b,
                    "range": [round(float(bins[b]), 2), round(float(bins[b + 1]), 2)],
                    "count": 0,
                    "mean_pred": None,
                    "empirical_rate": None,
                }
            )
            continue

        mean_pred = float(np.mean(y_prob[mask]))
        empirical_rate = float(np.mean(y_true[mask]))
        gap = abs(empirical_rate - mean_pred)
        ece += (count / total_samples) * gap

        reliability_bins.append(
            {
                "bin": b,
                "range": [round(float(bins[b]), 2), round(float(bins[b + 1]), 2)],
                "count": count,
                "mean_pred": round(mean_pred, 4),
                "empirical_rate": round(empirical_rate, 4),
                "calibration_gap": round(gap, 4),
            }
        )

    return float(round(ece, 4)), reliability_bins


def compute_precision_and_lift_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ks: tuple[int, ...] = (10, 25, 50, 100),
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute precision@k and lift@k for ranked predictions."""
    order = np.argsort(-y_prob)
    sorted_y = y_true[order]
    base_rate = float(np.mean(y_true)) if len(y_true) > 0 else 0.0

    p_at_k: dict[str, float] = {}
    lift_at_k: dict[str, float] = {}

    for k in ks:
        if k > len(sorted_y):
            continue
        p = float(np.mean(sorted_y[:k]))
        p_at_k[f"p@{k}"] = round(p, 4)
        lift = round(p / base_rate, 2) if base_rate > 0 else 1.0
        lift_at_k[f"lift@{k}"] = lift

    return p_at_k, lift_at_k


def run_task(
    workspace: Workspace,
    task_name: str,
    model: str = DEFAULT_MODEL,
    cutoff: str | None = None,
    runs_dir: str | Path = "runs",
    cache_dir: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit TabPFN-Rel, run backtests, generate baselines, and save artifacts."""
    # Ensure macOS thread isolation
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Load TABPFN_TOKEN if present
    env_path = Path(".env")
    if env_path.exists() and "TABPFN_TOKEN" not in os.environ:
        for line in env_path.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()

    from tabpfn_client import get_api_usage, init
    from tabpfn_rel import PredictiveContext, PredictiveQuery, PredictiveQuerySpec

    init()

    # Query usage before fit
    try:
        usage_before = get_api_usage()
        print(f"[API Usage Before Fit] {usage_before}")
    except Exception as e:
        usage_before = f"Unavailable: {e}"

    task_file = workspace.get_task_file(task_name)
    spec = PredictiveQuerySpec.from_yaml(str(task_file), data_dir=str(workspace.data_dir))
    context = PredictiveContext(spec)

    target_col = spec.task.target_col
    entity_col = spec.task.entity_col
    time_col = spec.task.time_col

    # Train TabPFN-Rel on historical labels
    print(f"[RelPilot Engine] Fitting {model} on task '{task_name}' (n_trials=0, seed={seed})...")
    start_time = time.perf_counter()
    fitted = context.fit(model=model, n_trials=0, seed=seed, cache_dir=cache_dir)
    fit_duration = time.perf_counter() - start_time
    print(f"[RelPilot Engine] Fit finished in {fit_duration:.2f} s")

    # Query usage after fit
    try:
        usage_after = get_api_usage()
        print(f"[API Usage After Fit] {usage_after}")
    except Exception as e:
        usage_after = f"Unavailable: {e}"

    # Backtest on the latest fully-labelled period
    print("[RelPilot Engine] Computing backtest on latest fully-labelled period...")
    test_labels = context.compute_test_labels()

    # Predict test cohort for backtest
    backtest_query = PredictiveQuery(entities="all", at_timestamp="test_timestamp")
    test_preds = fitted.predict(backtest_query, cache_dir=cache_dir)

    pred_col = f"{target_col}_pred"
    scored_backtest = test_labels.merge(
        test_preds,
        on=[time_col, entity_col],
        how="inner",
    )

    y_true = scored_backtest[target_col].to_numpy().astype(int)
    y_prob = scored_backtest[pred_col].to_numpy().astype(float)

    # Check if probabilities are present
    has_probabilities = (
        np.all(y_prob >= 0.0)
        and np.all(y_prob <= 1.0)
        and (np.unique(y_prob).size > 2 or np.any((y_prob > 0.0) & (y_prob < 1.0)))
    )

    # Metric evaluation
    if len(np.unique(y_true)) > 1:
        tabpfn_auroc = round(float(roc_auc_score(y_true, y_prob)), 4)
    else:
        tabpfn_auroc = 0.5

    loss_val = None
    ece_val = None
    rel_bins: list[dict[str, Any]] = []
    p_at_k: dict[str, float] = {}
    lift_at_k: dict[str, float] = {}

    if has_probabilities:
        # Clip slightly to avoid inf in log-loss
        clipped_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
        loss_val = round(float(log_loss(y_true, clipped_prob)), 4)
        ece_val, rel_bins = compute_ece(y_true, y_prob, n_bins=10)
        p_at_k, lift_at_k = compute_precision_and_lift_at_k(y_true, y_prob)

    # Baselines (pure in-process, zero API cost)
    train_df = context.task.get_table("train", mask_input_cols=False).df
    global_rate = float(train_df[target_col].mean()) if len(train_df) > 0 else 0.5
    entity_rates = train_df.groupby(entity_col)[target_col].mean().to_dict()

    per_entity_preds = np.array([entity_rates.get(ent, global_rate) for ent in scored_backtest[entity_col]])
    if len(np.unique(y_true)) > 1:
        per_entity_auroc = round(float(roc_auc_score(y_true, per_entity_preds)), 4)
    else:
        per_entity_auroc = 0.5
    global_baseline_auroc = 0.5000

    # Derive act-threshold from backtest
    threshold, support, achieved_prec = find_act_threshold(
        y_true,
        y_prob,
        min_precision=DEFAULT_MIN_PRECISION,
        min_support=DEFAULT_MIN_SUPPORT,
    )

    # Predict current entities
    target_anchor = cutoff or "test_timestamp"
    live_query = PredictiveQuery(entities="all", at_timestamp=target_anchor)
    live_preds = fitted.predict(live_query, cache_dir=cache_dir)

    # Cap live entities to MAX_PREDICT_ENTITIES if needed
    if len(live_preds) > MAX_PREDICT_ENTITIES:
        live_preds = live_preds.sort_values(by=pred_col, ascending=False).head(MAX_PREDICT_ENTITIES)

    # Generate Run ID and persist artifacts
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    scored_backtest.to_parquet(run_dir / "backtest.parquet", index=False)
    live_preds.to_parquet(run_dir / "predictions.parquet", index=False)

    compact_trust_report = {
        "run_id": run_id,
        "task": task_name,
        "model": model,
        "backtest_samples": len(y_true),
        "auroc": tabpfn_auroc,
        "baseline_global_auroc": global_baseline_auroc,
        "baseline_per_entity_auroc": per_entity_auroc,
        "log_loss": loss_val,
        "ece": ece_val,
        "probabilities_available": has_probabilities,
        "recommended_threshold": round(threshold, 4) if threshold is not None else None,
        "threshold_support": support,
        "threshold_precision": round(achieved_prec, 4) if achieved_prec is not None else None,
        "precision_at_k": p_at_k,
        "lift_at_k": lift_at_k,
    }

    full_metrics = {
        **compact_trust_report,
        "fit_duration_seconds": round(fit_duration, 2),
        "reliability_bins": rel_bins,
        "api_usage_before": str(usage_before),
        "api_usage_after": str(usage_after),
    }

    with open(run_dir / "trust_report.json", "w", encoding="utf-8") as f:
        json.dump(compact_trust_report, f, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # Prepare top-K live predictions
    top_k_df = live_preds.sort_values(by=pred_col, ascending=False).head(10)
    top_k_list = [
        {"entity_id": str(row[entity_col]), "score": round(float(row[pred_col]), 4)}
        for _, row in top_k_df.iterrows()
    ]

    return {
        "run_id": run_id,
        "task": task_name,
        "model": model,
        "trust_report": compact_trust_report,
        "top_k": top_k_list,
        "artifacts_dir": str(run_dir),
    }


def load_trust_report(run_id: str, runs_dir: str | Path = "runs") -> dict[str, Any]:
    """Retrieve the trust report JSON artifact for a run."""
    report_path = Path(runs_dir) / run_id / "trust_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Trust report for run '{run_id}' not found at {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)
