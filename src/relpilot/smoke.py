"""Live smoke test runner for RelPilot MVP."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Thread limit for macOS OpenMP safety
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load environment variables if .env exists
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("TABPFN_TOKEN=") and "TABPFN_TOKEN" not in os.environ:
            os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()

import tabpfn_client
from tabpfn_client import get_api_usage, init
from relpilot.engine import run_task
from relpilot.server import propose_actions
from relpilot.workspace import Workspace


def run_smoke() -> None:
    print("==================================================================")
    print("RELPILOT LIVE SMOKE TEST")
    print("==================================================================")

    init()
    initial_usage = get_api_usage()
    print(f"Starting API Quota: {initial_usage}\n")

    workspace_dir = os.environ.get("RELPILOT_WORKSPACE", "workspaces/demo")
    print(f"Loading Workspace: {workspace_dir}")
    ws = Workspace(workspace_dir)
    print(f"Available tasks: {list(ws.tasks.keys())}")
    print(f"Available skills: {list(ws.skills.keys())}\n")

    task_to_run = "bad_review_risk"
    print(f"--- Running Task: {task_to_run} ---")
    result = run_task(
        workspace=ws,
        task_name=task_to_run,
        model="tabpfn-rel-client-latest",
        runs_dir="runs",
    )

    run_id = result["run_id"]
    trust_rep = result["trust_report"]
    print(f"\nTask completed! Run ID: {run_id}")
    print("\n--- TRUST REPORT ---")
    print(json.dumps(trust_rep, indent=2))

    print("\n--- EVALUATING GATE & PROPOSING ACTIONS ---")
    gate_output_json = propose_actions(
        run_id=run_id,
        action="qa_fulfillment_outreach",
        min_precision=0.75,
    )
    gate_result = json.loads(gate_output_json)
    print(json.dumps(gate_result, indent=2))

    final_usage = get_api_usage()
    print("\n==================================================================")
    print("SMOKE TEST COMPLETE")
    print(f"Final API Quota: {final_usage}")
    print("==================================================================")


if __name__ == "__main__":
    run_smoke()
