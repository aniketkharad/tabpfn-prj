# RelPilot: Measured Benchmark & Backtest Results

This document contains **ONLY** verified, empirically measured metrics from real runs executed on the hosted TabPFN API. No theoretical or recall numbers are reported.

---

## 1. Primary Benchmark Task: Seller Bad Review Risk

- **Task**: `bad_review_risk` (predict whether an active seller will receive any `review_score <= 2` on orders in the next 30 days)
- **Database**: Olist Brazilian E-Commerce (7 relational tables)
- **Model**: `tabpfn-rel-client-latest` (hosted TabPFN-3.5 API)
- **Evaluation Split**: Historical test cohort anchored at `2018-06-15` (strictly censored to prevent future lookahead leakage)
- **Cohort Size**: **1,088 active sellers** (236 positive ground truth labels, 21.7% base rate)
- **Execution Run ID**: `run_20261006_145425_8b5ae4`

### Measured Discrimination & Calibration

| Method / Model | AUROC | Log Loss | ECE (10 bins) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **TabPFN-Rel (zero-shot)** | **0.7734** | **0.4776** | **0.1310** | Zero-shot DFS + TabPFN-3.5 API in-context prediction |
| **Baseline: Global Constant** | 0.5000 | — | — | Emits global training target base rate |
| **Baseline: Per-Entity Constant** | **0.7160** | — | — | Emits per-entity historical rate (mapped via pkey_maps), fallback to global |

### High-Confidence Ranking (Precision & Lift @ K)

| Cohort Depth ($k$) | Precision @ $k$ | Lift @ $k$ (vs 21.7% base rate) | True Positives |
| :---: | :---: | :---: | :---: |
| **Top 10 ($P@10$)** | **1.0000** (100.0%) | **4.61×** | 10 / 10 |
| **Top 25 ($P@25$)** | **0.8800** (88.0%) | **4.06×** | 22 / 25 |
| **Top 50 ($P@50$)** | **0.8400** (84.0%) | **3.87×** | 42 / 50 |
| **Top 100 ($P@100$)** | **0.7000** (70.0%) | **3.23×** | 70 / 100 |

### Precision Gate Decision (at $\ge 75\%$ Precision Target)

- **Backtest-Derived Threshold ($\tau$)**: **0.7593**
- **Empirical Backtest Precision**: **75.34%** (55 true positives out of 73 entities)
- **Threshold Support**: **73 entities**
- **Autonomous Actions Proposed**: **73 entities** (appended to `outbox.jsonl` with status `proposed`)
- **Entities Routed to Review Queue**: **427 entities** (scores below safety cutoff, reason logged)

---

## 2. Additional Measured Operational Tasks

Both additional tasks were executed on the same database and test split:

| Task Name | Target Outcome | TabPFN-Rel AUROC | Baseline AUROC | Run ID |
| :--- | :--- | :---: | :---: | :--- |
| **High-Volume Growth Surge** | Will active seller receive $\ge 5$ orders in next 30d? | **0.8964** | 0.5000 | `run_spike2` |
| **Official Seller Churn** | Will active seller receive 0 orders in next 30d? | **0.7766** | 0.5000 (0.688 per-ent) | `run_step4` |

---

## 3. Explicit Statement of Limitations

1. **Single Dataset**: All metrics are evaluated on the Olist Brazilian E-Commerce dataset (~100,000 orders). Performance across disparate database domains (financial transactions, healthcare EHR, logistics graphs) will vary.
2. **Sample Size**: The evaluation cohort contains 1,088 active sellers. While statistically sufficient for a minimum support of 20, larger databases with tens of thousands of active entities may reveal different calibration nuances.
3. **Single Temporal Cutoff**: Evaluations used a single fixed test cutoff (`2018-06-15`). Multi-period rolling backtests across shifting economic seasons were not performed.
4. **Alpha Status**: TabPFN-Rel and RelArena are early alpha research releases (`relarena-α`, `tabpfn-rel==0.0.5`). APIs, featurization depth bounds, and schema contracts are subject to upstream evolution.
5. **No Production Action Integrations**: Proposed actions are appended to `outbox.jsonl` as an auditable queue; actual production integrations (dispatching emails, CRM webhooks) must consume from this outbox.
