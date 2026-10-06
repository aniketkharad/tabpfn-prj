"""Unit tests for RelPilot core modules (offline, no external network)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from relpilot.gate import find_act_threshold, partition_actions
from relpilot.workspace import Workspace, WorkspaceValidationError
from relpilot.server import describe_workspace, propose_actions, read_skill, trust_report


# ==============================================================================
# (a) Task-to-spec Translation & Strict Temporal Split Tests
# ==============================================================================

def test_task_temporal_cutoffs_strictly_ordered():
    """Verify that task specifications enforce val_timestamp < test_timestamp."""
    task_files = list(Path("workspaces/demo/tasks").glob("*.yaml"))
    assert len(task_files) >= 1, "At least one task file should exist in workspaces/demo/tasks"

    for t_file in task_files:
        with open(t_file, "r", encoding="utf-8") as f:
            t_data = yaml.safe_load(f)

        val_ts = pd.Timestamp(t_data["val_timestamp"])
        test_ts = pd.Timestamp(t_data["test_timestamp"])
        assert val_ts < test_ts, f"val_timestamp must be strictly before test_timestamp in {t_file.name}"

        # Verify SQL forward window syntax
        query = t_data["query"]
        assert "purchase_ts > timestamp" in query
        assert "purchase_ts <= timestamp + INTERVAL '{timedelta}'" in query
        # Verify backward-looking existence condition
        assert "purchase_ts > timestamp - INTERVAL '{timedelta}'" in query
        assert "purchase_ts <= timestamp" in query


# ==============================================================================
# (b) Gate Tests: Monotone, Support Minimum, and No-Threshold Case
# ==============================================================================

def test_gate_monotone_behavior():
    """Higher precision targets should produce equal or higher thresholds."""
    np.random.seed(42)
    # Generate synthetic scores and labels with signal
    scores = np.random.uniform(0.0, 1.0, size=500)
    labels = (scores + np.random.normal(0, 0.2, size=500) > 0.6).astype(int)

    t_70, s_70, p_70 = find_act_threshold(labels, scores, min_precision=0.70, min_support=20)
    t_80, s_80, p_80 = find_act_threshold(labels, scores, min_precision=0.80, min_support=20)

    assert t_70 is not None
    if t_80 is not None:
        assert t_80 >= t_70, f"Threshold for 80% precision ({t_80}) should be >= threshold for 70% ({t_70})"


def test_gate_support_minimum():
    """High precision on fewer than MIN_SUPPORT samples must not be chosen."""
    # 10 perfect samples, but min_support is 20
    scores = np.array([0.95] * 10 + [0.10] * 100)
    labels = np.array([1] * 10 + [0] * 100)

    threshold, support, prec = find_act_threshold(labels, scores, min_precision=0.90, min_support=20)
    assert threshold is None, "Should reject when sample support is below minimum"
    assert support == 0


def test_gate_no_threshold_case():
    """When target precision cannot be achieved, returns None and empty proposed list."""
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5] * 20)
    labels = np.zeros(len(scores), dtype=int)  # All zeros, precision is 0.0 everywhere

    threshold, support, prec = find_act_threshold(labels, scores, min_precision=0.50, min_support=20)
    assert threshold is None
    assert support == 0
    assert prec is None

    preds_df = pd.DataFrame({"entity_id": [f"e_{i}" for i in range(10)], "score": [0.8] * 10})
    proposed, needs_review = partition_actions(
        preds_df,
        entity_col="entity_id",
        pred_col="score",
        action_name="test_action",
        run_id="run_123",
        threshold=None,
    )
    assert len(proposed) == 0
    assert len(needs_review) == 10
    assert "No backtest threshold" in needs_review[0]["reason"]


# ==============================================================================
# (c) Workspace Validation Errors
# ==============================================================================

def test_workspace_validation_missing_dir(tmp_path):
    """Missing workspace folder raises WorkspaceValidationError."""
    with pytest.raises(WorkspaceValidationError, match="does not exist"):
        Workspace(tmp_path / "non_existent_folder")


def test_workspace_validation_missing_data_dir(tmp_path):
    """Missing 'data' subfolder raises WorkspaceValidationError."""
    empty_ws = tmp_path / "empty_ws"
    empty_ws.mkdir()
    with pytest.raises(WorkspaceValidationError, match="Missing required 'data' directory"):
        Workspace(empty_ws)


def test_workspace_validation_invalid_schema_fkey(tmp_path):
    """Schema with dangling foreign key raises WorkspaceValidationError."""
    ws_dir = tmp_path / "ws_invalid"
    ws_dir.mkdir()
    data_dir = ws_dir / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1]}).to_csv(data_dir / "t1.csv", index=False)

    schema = {
        "t1": {
            "pkey": "id",
            "path": "t1.csv",
            "fkeys": {"parent_id": "non_existent_parent"},
        }
    }
    with open(ws_dir / "schema.yaml", "w", encoding="utf-8") as f:
        yaml.dump(schema, f)

    with pytest.raises(WorkspaceValidationError, match="pointing to non-existent table"):
        Workspace(ws_dir)


def test_workspace_validation_inverted_cutoffs(tmp_path):
    """Task with val_timestamp >= test_timestamp raises WorkspaceValidationError."""
    ws_dir = tmp_path / "ws_inverted"
    ws_dir.mkdir()
    data_dir = ws_dir / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1], "ts": ["2020-01-01"]}).to_csv(data_dir / "t1.csv", index=False)

    schema = {"t1": {"pkey": "id", "path": "t1.csv", "time_col": "ts"}}
    with open(ws_dir / "schema.yaml", "w", encoding="utf-8") as f:
        yaml.dump(schema, f)

    tasks_dir = ws_dir / "tasks"
    tasks_dir.mkdir()
    task = {
        "database": "schema.yaml",
        "entity_table": "t1",
        "entity_col": "id",
        "time_col": "ts",
        "target_col": "target",
        "task_type": "binary_classification",
        "timedelta": "30 days",
        "val_timestamp": "2020-06-01",
        "test_timestamp": "2020-01-01",  # INVERTED!
        "query": "SELECT ts, id, 1 AS target FROM t1",
    }
    with open(tasks_dir / "inverted_task.yaml", "w", encoding="utf-8") as f:
        yaml.dump(task, f)

    with pytest.raises(WorkspaceValidationError, match="strictly before"):
        Workspace(ws_dir)


# ==============================================================================
# (d) Compact Output Size Tests (<= 5 KB limit)
# ==============================================================================

def test_describe_workspace_output_size():
    """describe_workspace() output must be <= 5 KB."""
    out = describe_workspace()
    assert len(out.encode("utf-8")) <= 5120, f"describe_workspace payload size {len(out)} > 5 KB"
    data = json.loads(out)
    assert "tables" in data
    assert "tasks" in data
    assert "skills" in data


def test_read_skill_output_size():
    """read_skill output must be valid and concise."""
    for skill in ("task-design", "acting-on-predictions"):
        text = read_skill(skill)
        assert len(text) > 100
        assert len(text.encode("utf-8")) <= 5120, f"Skill {skill} size {len(text)} > 5 KB"


def test_trust_report_and_propose_actions_output_size(tmp_path, monkeypatch):
    """trust_report and propose_actions tool outputs must be <= 5 KB."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_id = "test_run_001"
    run_folder = runs_dir / run_id
    run_folder.mkdir()

    trust_data = {
        "run_id": run_id,
        "task": "bad_review_risk",
        "model": "tabpfn-rel-client-latest",
        "auroc": 0.7735,
        "baseline_global_auroc": 0.5000,
        "baseline_per_entity_auroc": 0.5000,
        "ece": 0.045,
        "recommended_threshold": 0.65,
        "threshold_support": 42,
        "threshold_precision": 0.81,
    }
    with open(run_folder / "trust_report.json", "w", encoding="utf-8") as f:
        json.dump(trust_data, f)

    # Synthetic backtest and predictions
    np.random.seed(0)
    backtest_df = pd.DataFrame({
        "seller_id": [f"s_{i}" for i in range(100)],
        "timestamp": [pd.Timestamp("2018-06-15")] * 100,
        "bad_review_risk": [1 if i < 30 else 0 for i in range(100)],
        "bad_review_risk_pred": np.linspace(0.9, 0.1, 100),
    })
    backtest_df.to_parquet(run_folder / "backtest.parquet", index=False)

    preds_df = pd.DataFrame({
        "seller_id": [f"live_{i}" for i in range(100)],
        "timestamp": [pd.Timestamp("2018-06-15")] * 100,
        "bad_review_risk_pred": np.linspace(0.95, 0.05, 100),
    })
    preds_df.to_parquet(run_folder / "predictions.parquet", index=False)

    monkeypatch.setenv("RELPILOT_RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("RELPILOT_OUTBOX", str(tmp_path / "test_outbox.jsonl"))

    # Test trust_report payload size
    report_out = trust_report(run_id)
    assert len(report_out.encode("utf-8")) <= 5120, f"trust_report payload size {len(report_out)} > 5 KB"

    # Test propose_actions payload size
    actions_out = propose_actions(run_id, action="qa_outreach", min_precision=0.75)
    assert len(actions_out.encode("utf-8")) <= 5120, f"propose_actions payload size {len(actions_out)} > 5 KB"

    resp = json.loads(actions_out)
    assert resp["proposed_count"] > 0
    assert resp["status"] == "actions_proposed"


def test_report_generation(tmp_path):
    """relpilot.report must create self-contained HTML with SVG chart, baselines, and <= 80 lines."""
    from relpilot.report import generate_report
    import relpilot.report

    # Check line count constraint (<= 80 lines)
    import inspect
    lines = inspect.getsourcelines(relpilot.report)[0]
    assert len(lines) <= 80, f"report.py must be <= 80 lines, got {len(lines)}"

    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    metrics = {
        "run_id": "run_test",
        "task": "bad_review_risk",
        "auroc": 0.7734,
        "baseline_global_auroc": 0.5,
        "baseline_per_entity_auroc": 0.5,
        "ece": 0.131,
        "recommended_threshold": 0.7842,
        "threshold_support": 61,
        "threshold_precision": 0.8033,
        "reliability_bins": [
            {"count": 10, "mean_pred": 0.1, "empirical_rate": 0.05},
            {"count": 20, "mean_pred": 0.8, "empirical_rate": 0.75},
        ]
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    df = pd.DataFrame({
        "seller_id": ["s1", "s2"],
        "timestamp": ["2018-06-15", "2018-06-15"],
        "bad_review_risk_pred": [0.95, 0.85]
    })
    df.to_parquet(run_dir / "predictions.parquet")

    out_file = generate_report(str(run_dir))
    assert out_file.exists()
    content = out_file.read_text()

    # Self-contained check
    assert "<svg" in content
    assert "Reliability Chart" in content
    assert "Recommended Threshold" in content
    assert "Top Risk Entities" in content
    assert "TabPFN-Rel" in content
    assert "http://" not in content and "https://" not in content  # no external assets

