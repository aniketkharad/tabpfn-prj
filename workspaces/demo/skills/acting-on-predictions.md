# Playbook: Reading Trust Reports and Gating Operational Actions

This guide explains how an AI agent or operator reads a RelPilot **Trust Report** and applies the **Precision Gate** before triggering real-world actions.

---

## 1. The Core Principle: Earned Trust

> "The agent acts only where the model has proven it deserves trust."

Zero-shot predictions over multi-table relational data are powerful, but raw model probabilities should **never** directly trigger downstream actions (discounts, merchant penalties, inventory orders) without empirical temporal validation.

RelPilot evaluates TabPFN-Rel on a strictly backtested historical period before scoring live entities. The decision gate requires that any autonomous action operate at an empirically proven precision threshold with sufficient statistical support.

---

## 2. Structure of the Trust Report

The `trust_report(run_id)` MCP tool returns a compact JSON summary covering:

1. **Discrimination vs. Baselines**:
   - `auroc`: Area Under the ROC curve on the backtest cohort.
   - `baseline_global_auroc`: Global constant predictor (always 0.500).
   - `baseline_per_entity_auroc`: Entity historical rate baseline.
   - *Requirement*: TabPFN-Rel must demonstrate clear lift over the naive baselines.

2. **Calibration & Probability Reliability**:
   - `ece`: Expected Calibration Error across 10 probability bins. A low ECE (< 0.10) indicates that predicted probabilities reflect true empirical event frequencies.
   - `reliability_bins`: Binned predicted probabilities vs. actual observed positive rates.

3. **Ranking & Top-Cohort Utility**:
   - `precision_at_k`: Observed precision at top 10, 25, 50, and 100 scored entities.
   - `lift_at_k`: Precision relative to the cohort base rate.

4. **Backtest-Derived Act-Threshold**:
   - `recommended_threshold`: The lowest probability threshold $\tau$ where backtest precision meets or exceeds the target (default 80%) with at least `min_support` positive occurrences (default 20).
   - `achieved_precision`: The empirical precision observed at $\tau$.
   - `support`: Number of backtest entities at or above $\tau$.

---

## 3. The Gating Decision (`propose_actions`)

When `propose_actions(run_id, action, min_precision=0.8)` is invoked:

1. **Threshold Lookup**:
   - The gate verifies whether a valid threshold exists on the backtest split meeting `precision >= min_precision` and `support >= 20`.
   - **No Safe Threshold**: If no probability cut reaches the precision target, the gate **refuses to propose any actions**. All entities are routed to `needs_review` with the reason `"Insufficient backtest precision (target: X%)"`.

2. **Action Partitioning**:
   - Entities with predicted probability $p \ge \tau$ are written to `outbox.jsonl` with status `proposed`.
   - Entities with $p < \tau$ are placed into the `needs_review` list with explanations (e.g. `"Confidence 0.65 below safe threshold 0.78"`).

3. **Audit Trail**:
   - Every autonomous action proposed by RelPilot is appended to `outbox.jsonl` with full metadata (run ID, entity ID, prediction score, threshold applied, timestamp), creating an auditable paper trail.
