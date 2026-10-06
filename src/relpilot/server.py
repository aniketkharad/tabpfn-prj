"""RelPilot MCP Server: exposes 5 tools over stdio for relational prediction & action gating."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
from mcp.server.mcpserver import MCPServer

from relpilot.engine import load_trust_report, run_task as engine_run_task
from relpilot.gate import DEFAULT_MIN_PRECISION, DEFAULT_MIN_SUPPORT, find_act_threshold, partition_actions, record_outbox
from relpilot.workspace import Workspace, WorkspaceValidationError

def _get_workspace_dir() -> str:
    return os.environ.get("RELPILOT_WORKSPACE", "workspaces/demo")


def _get_runs_dir() -> Path:
    return Path(os.environ.get("RELPILOT_RUNS_DIR", "runs"))


def _get_outbox_path() -> Path:
    return Path(os.environ.get("RELPILOT_OUTBOX", "outbox.jsonl"))


server = MCPServer("relpilot")


def _get_workspace() -> Workspace:
    return Workspace(_get_workspace_dir())


def _format_error(msg: str) -> str:
    return json.dumps({"error": True, "message": msg})


@server.tool()
def describe_workspace() -> str:
    """Describe tables, columns, foreign keys, prediction tasks, and available skills."""
    try:
        ws = _get_workspace()
        desc = ws.describe()
        return json.dumps(desc)
    except Exception as e:
        return _format_error(f"Failed to describe workspace: {e}")


@server.tool()
def read_skill(name: str) -> str:
    """Read a playbook skill markdown file by name (e.g. 'task-design' or 'acting-on-predictions')."""
    try:
        ws = _get_workspace()
        return ws.read_skill(name)
    except Exception as e:
        return _format_error(f"Failed to read skill '{name}': {e}")


@server.tool()
def run_task(
    task: str,
    model: str = "tabpfn-rel-client-latest",
    cutoff: str | None = None,
) -> str:
    """Fit TabPFN-Rel, run backtests on latest labelled period, predict entities, and save artifacts."""
    try:
        ws = _get_workspace()
        result = engine_run_task(
            workspace=ws,
            task_name=task,
            model=model,
            cutoff=cutoff,
            runs_dir=_get_runs_dir(),
        )
        return json.dumps(result)
    except Exception as e:
        return _format_error(f"Error executing task '{task}': {e}")


@server.tool()
def trust_report(run_id: str) -> str:
    """Retrieve the trust report for a run, including AUROC vs baselines, ECE, calibration bins, and threshold."""
    try:
        report = load_trust_report(run_id, runs_dir=_get_runs_dir())
        return json.dumps(report)
    except Exception as e:
        return _format_error(f"Failed to retrieve trust report for run '{run_id}': {e}")


@server.tool()
def propose_actions(
    run_id: str,
    action: str,
    min_precision: float = DEFAULT_MIN_PRECISION,
    use_wilson_bound: bool = False,
) -> str:
    """The Gate: propose operational actions for entities above the backtest precision threshold."""
    try:
        run_dir = _get_runs_dir() / run_id
        if not run_dir.exists():
            return _format_error(f"Run directory '{run_id}' not found.")

        preds_path = run_dir / "predictions.parquet"
        backtest_path = run_dir / "backtest.parquet"
        report_path = run_dir / "trust_report.json"

        if not preds_path.exists() or not backtest_path.exists() or not report_path.exists():
            return _format_error(f"Missing required artifacts in run directory '{run_dir}'.")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        task_name = report.get("task", "")
        ws = _get_workspace()
        task_file = ws.get_task_file(task_name)
        
        # Read task spec for column names
        import yaml
        with open(task_file, "r", encoding="utf-8") as f:
            task_data = yaml.safe_load(f)

        target_col = task_data["target_col"]
        entity_col = task_data["entity_col"]
        pred_col = f"{target_col}_pred"

        # Load backtest and find threshold matching requested min_precision
        backtest_df = pd.read_parquet(backtest_path)
        y_true = backtest_df[target_col].to_numpy().astype(int)
        y_prob = backtest_df[pred_col].to_numpy().astype(float)

        threshold, support, achieved_prec = find_act_threshold(
            y_true,
            y_prob,
            min_precision=min_precision,
            min_support=DEFAULT_MIN_SUPPORT,
            use_wilson_bound=use_wilson_bound,
        )

        preds_df = pd.read_parquet(preds_path)
        proposed, needs_review = partition_actions(
            preds_df=preds_df,
            entity_col=entity_col,
            pred_col=pred_col,
            action_name=action,
            run_id=run_id,
            threshold=threshold,
            achieved_precision=achieved_prec,
        )

        outbox_file = _get_outbox_path()
        written_count = record_outbox(proposed, outbox_file=outbox_file)

        response: dict[str, Any] = {
            "run_id": run_id,
            "action": action,
            "min_precision_target": min_precision,
            "threshold": threshold,
            "achieved_precision": achieved_prec,
            "threshold_support": support,
            "proposed_count": len(proposed),
            "newly_written_count": written_count,
            "needs_review_count": len(needs_review),
            "outbox_file": str(outbox_file),
            "status": "actions_proposed" if threshold is not None else "no_threshold_met",
            "message": (
                f"Proposed {len(proposed)} actions meeting {min_precision:.0%} precision target ({written_count} newly appended to outbox)."
                if threshold is not None
                else f"No backtest threshold achieved the required {min_precision:.0%} precision target with minimum support of {DEFAULT_MIN_SUPPORT}."
            ),
            "sample_proposed": proposed[:5],
            "sample_needs_review": needs_review[:5],
        }
        return json.dumps(response)
    except Exception as e:
        return _format_error(f"Failed to propose actions: {e}")


def main() -> None:
    """Run the MCP server over stdio."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
