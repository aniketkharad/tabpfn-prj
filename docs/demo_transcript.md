# RelPilot Scripted Demo Transcript

**Executed**: 2026-10-06 11:01:58 UTC

## User Prompt

> I have an e-commerce database with sellers, orders, items, and reviews. First, discover the workspace tables and read the skill on acting on predictions. Next, run the bad_review_risk prediction task. Explain the backtest trust report in plain words, comparing against baselines. Finally, propose QA outreach actions for entities meeting at least 75% precision.

## Execution Log & Tool Calls

- `-> describe_workspace()`
- `-> read_skill(name='acting-on-predictions')`
- `-> run_task(task='bad_review_risk')`
- `-> trust_report(run_id='run_20261006_163338_08a817')`
- `-> propose_actions(min_precision=0.75, action='QA Outreach', run_id='run_20261006_163338_08a817')`

## Agent Response & Trust Synthesis

### 1. Workspace Discovery & Consultation of Playbook Skills

By discovering the workspace, we identified the following relational tables:
*   **Sellers**: Entity primary key `seller_id` (contains location data).
*   **Customers**: Entity primary key `customer_id`.
*   **Products & Product Categories**: Contains physical dimensions and category classifications.
*   **Orders**: Captures purchases and purchase timestamps.
*   **Order Items & Order Reviews**: Joins orders, products, and sellers, logging prices, freight values, and review scores over time.

We also verified two predefined prediction tasks on `sellers`: **`bad_review_risk`** and **`churn`**. 

#### Consultation of "Acting on Predictions" Skill:
The central tenet of the playbook is **Earned Trust**. Relational models are extremely powerful, but raw probabilities shouldn't directly trigger expensive or invasive operational actions without backtested empirical validation. We utilize the **Precision Gate** to set a strict statistical threshold based on a target precision. The agent will only propose automated actions for entities exceeding this empirical threshold; all others are safely routed to a review list.

---

### 2. Execution of prediction task: `bad_review_risk`
We trained TabPFN-Rel on historical data and backtested on the latest labeled period.
*   **Run ID:** `run_20261006_163338_08a817`
*   **Backtest Sample Size:** 1,088 historical cohorts

---

### 3. Trust Report Analysis (In Plain Words)

The backtest results demonstrate outstanding predictive power:

*   **Discrimination (AUROC):**
    *   **TabPFN-Rel AUROC:** `0.7734`
    *   **Global Constant Baseline AUROC:** `0.5` (represents random guessing)
    *   **Per-Entity Historical Rate Baseline AUROC:** `0.716` (predicting based solely on a seller's past average review rate)
    *   *Interpretation:* TabPFN-Rel achieves a **0.2734** improvement over random guessing and a **0.0574** improvement over a seller's past history. This shows the model is successfully extracting complex relational patterns (like delivery times, product combinations, and pricing dynamics) beyond simple historical rates.
*   **Calibration (ECE):** The Expected Calibration Error is `0.1312`, indicating that the predicted probabilities are reasonably close to the actual empirical rate of negative reviews.
*   **Top-Cohort Performance (Precision & Lift):**
    *   **p@10:** `1.0` (100% of the top 10 highest-risk sellers actually got bad reviews, which is a **4.61x** lift over the average base rate).
    *   **p@25:** `0.88` (88% of the top 25 got bad reviews, a **4.06x** lift).
    *   **p@50:** `0.84` (84% of the top 50 got bad reviews, a **3.87x** lift).
    *   **p@100:** `0.70` (70% of the top 100 got bad reviews, a **3.23x** lift).

---

### 4. Proposing "QA Outreach" Actions (at $\ge 75\%$ Precision)

We applied the Precision Gate to find the optimal risk probability cutoff where we can guarantee at least a **75% precision target** with historical statistical support.

*   **Calculated Safe Threshold:** `0.7595347762107849` (~`0.7595`)
*   **Achieved Backtest Precision:** `75.34%` (`0.7534246575342466`)
*   **Proposed Actions:** **73 sellers** met or exceeded this threshold and have had "QA Outreach" actions proposed.
*   **Needs Review:** **427 sellers** had high-risk scores but fell below the safe threshold of `0.7595`. They are held back for human review to prevent unnecessary outreach.

#### Sample Entities Approved for Automated QA Outreach:
1.  **Seller ID:** `1025f0e2d44d7041d6cf58b6550e0bfa` (Risk Score: `0.9872`)
2.  **Seller ID:** `955fee9216a65b617aa5c0531780ce60` (Risk Score: `0.9866`)
3.  **Seller ID:** `4a3ca9315b744ce9f8e9374361493884` (Risk Score: `0.9805`)
4.  **Seller ID:** `8b321bb669392f5163d04c59e235e066` (Risk Score: `0.9785`)
5.  **Seller ID:** `ea8482cd71df3c1969d7b9473ff13abc` (Risk Score: `0.9784`)

#### Sample Entities Held Back for Human Review:
*   **Seller ID:** `70a12e78e608ac31179aea7f8422044b` (Risk Score: `0.7574` — fell just short of the `0.7595` minimum precision threshold)
*   **Seller ID:** `5343d0649eca2a983820bfe93fc4d17e` (Risk Score: `0.7518` — below the safe threshold)

All 73 approved actions have been formally appended to `outbox.jsonl` with their risk scores and backtest metadata, establishing a robust and transparent audit trail.

---
*Transcript generated automatically by `python -m relpilot.demo`.*