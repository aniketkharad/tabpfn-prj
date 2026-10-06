# RelPilot Scripted Demo Transcript

**Executed**: 2026-10-06 09:21:15 UTC

## User Prompt

> I have an e-commerce database with sellers, orders, items, and reviews. First, discover the workspace tables and read the skill on acting on predictions. Next, run the bad_review_risk prediction task. Explain the backtest trust report in plain words, comparing against baselines. Finally, propose QA outreach actions for entities meeting at least 75% precision.

## Execution Log & Tool Calls

- `-> describe_workspace()`
- `-> read_skill(name='acting-on-predictions')`
- `-> run_task(task='bad_review_risk')`
- `-> trust_report(run_id='run_20261006_145425_8b5ae4')`
- `-> propose_actions(min_precision=0.75, run_id='run_20261006_145425_8b5ae4', action='QA outreach')`

## Agent Response & Trust Synthesis

I have successfully discovered the workspace, consulted the prediction skill playbook, executed the `bad_review_risk` predictive task, and applied the precision gate to propose actions. Here is the step-by-step summary:

---

### 1. Workspace Discovery & Skill Consultation
* **Tables Discovered:** 
  * `sellers` (Primary Entity table for the task)
  * `customers`
  * `product_category`
  * `products`
  * `orders`
  * `order_items`
  * `order_reviews` (Where review scores are recorded)
* **Prediction Task:** `bad_review_risk` (targets binary classification of sellers with a timedelta of 30 days and test timestamp of `2018-06-15 00:00:00`).
* **Skill Read:** Consulted `acting-on-predictions` which outlines how RelPilot uses rigorous backtesting and a strict **Precision Gate** so that we never action entities unless the model has empirically proven its accuracy in the backtest cohort.

---

### 2. Backtest Trust Report: `bad_review_risk`
The model ran a historical temporal backtest over **1,088 samples** in the latest labeled period.

| Metric | TabPFN-Rel Performance | Baselines (Global & Per-Entity) | Interpretation |
| :--- | :--- | :--- | :--- |
| **AUROC** | **0.7734** | **0.5000** | **Strong predictive power.** RelPilot shows significant lift over random guessing or entity historic rates. |
| **Log Loss** | **0.4776** | — | Measures the quality of the raw probabilities. |
| **Expected Calibration Error (ECE)** | **0.131** | — | Reflects reliable probabilities relative to true positive rates. |

#### High-Confidence Ranking Performance (Precision & Lift @ K)
* **Top 10 Sellers ($P@10$):** **100% precision** (1.0) with a **4.61x lift** over the base rate.
* **Top 25 Sellers ($P@25$):** **88% precision** (0.88) with a **4.06x lift**.
* **Top 50 Sellers ($P@50$):** **84% precision** (0.84) with a **3.87x lift**.
* **Top 100 Sellers ($P@100$):** **70% precision** (0.7) with a **3.23x lift**.

---

### 3. Gated Action Proposal: QA Outreach (Target: $\ge$ 75% Precision)
By setting a minimum precision constraint of **75%**, the **Precision Gate** calculated the following parameters:

* **Backtest-Derived Threshold ($\tau$):** **0.7593**
* **Achieved Empirical Precision:** **75.34%** (successfully meeting the $\ge$ 75% target)
* **Action Status:** `actions_proposed`

#### Outcomes:
1. **Proposed Actions (73 sellers):** 
   These 73 sellers scored above the safety threshold of `0.7593` and have been queued in `outbox.jsonl` for **QA outreach** actions.
   * *Top proposed entity:* `1025f0e2d44d7041d6cf58b6550e0bfa` (Score: **0.9880**)
   * *Other high-priority samples:* `955fee9216a65b617aa5c0531780ce60` (Score: **0.9874**), `4a3ca9315b744ce9f8e9374361493884` (Score: **0.9817**).

2. **Needs Human Review (427 sellers):** 
   Sellers scoring below the safety threshold of `0.7593` are held back to prevent unnecessary outreach.
   * *Sample border-line entity:* `70a12e78e608ac31179aea7f8422044b` (Score: **0.7583** — withheld because it is just below the `0.7593` threshold).

---
*Transcript generated automatically by `python -m relpilot.demo`.*