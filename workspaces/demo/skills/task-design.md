# Playbook: Designing Leakage-Safe Relational Prediction Tasks

This guide describes how to author a declarative relational prediction task for RelPilot using the verified Relational Predictive Interface (RPI) specifications.

---

## 1. Core Principles of Temporal Relational Prediction

Every prediction in a relational database is evaluated **as of an anchor timestamp $t$**.
To prevent temporal target leakage, the data is partitioned across two distinct time horizons relative to $t$:

```
                           Anchor Time t
  ──────────────────────────────┼──────────────────────────────▶ Time
    INPUT FEATURES              │   TARGET LABEL
    All rows in all tables with │   Computed strictly over
    event timestamp <= t        │   forward window (t, t + timedelta]
```

### The Three Leakage Rules

1. **Strict Forward Label Window**:
   - The target label must be computed solely from events occurring in the window `(timestamp, timestamp + INTERVAL '{timedelta}']`.
   - Never inspect rows on or before `timestamp` to define the target value itself.

2. **Entity Existence at Anchor (No Lookahead Cohort Selection)**:
   - Churn or risk is only defined for entities that are currently active as of the anchor.
   - Use a backward-looking condition (`WHERE EXISTS ... AND purchase_ts > timestamp - INTERVAL '{timedelta}' AND purchase_ts <= timestamp`) to qualify entities.
   - If you select entities based on future activity, the task will suffer from selection bias.

3. **Exclude Backfilled and Future Columns**:
   - In `schema.yaml`, use the `columns` allow-list to explicitly prune columns that are only populated after the event occurs (such as delivery dates, carrier tracking status, or cancellation timestamps on an order table).
   - Any column without a timestamp is treated as timeless/static; if it contains post-purchase attributes, it will leak future information into the model's feature set.

---

## 2. Declarative Task Schema Structure

A task YAML file must declare the following fields:

- `database`: Path to the database schema YAML (relative to the task file, e.g. `../schema.yaml`).
- `entity_table`: Target table containing the entity primary key (e.g. `sellers`).
- `entity_col`: Primary key column name for the entity (e.g. `seller_id`).
- `time_col`: The timestamp column emitted by the query (must be aliased to match).
- `target_col`: Target column name emitted by the query (e.g. `bad_review_risk`).
- `task_type`: Either `binary_classification` (0/1) or `regression` (continuous numeric).
- `timedelta`: Forward prediction window as a pandas Timedelta string (e.g. `'30 days'`).
- `val_timestamp`: Timestamp cutoff separating training from validation.
- `test_timestamp`: Timestamp cutoff separating validation from final testing.
- `num_eval_timestamps`: Number of forward anchor steps for validation/test evaluation (typically `1`).
- `query`: A DuckDB SQL query that joins against `timestamp_df` and returns **EXACTLY THREE columns**:
  1. `timestamp` (aliased to `time_col`)
  2. Entity ID (aliased to `entity_col`)
  3. Target outcome (aliased to `target_col`)

---

## 3. Reference Query Pattern

```sql
SELECT timestamp, seller_id,
  CAST(EXISTS (
    SELECT 1 FROM order_items
    JOIN order_reviews ON order_items.order_id = order_reviews.order_id
    WHERE order_items.seller_id = sellers.seller_id
      AND order_items.purchase_ts > timestamp
      AND order_items.purchase_ts <= timestamp + INTERVAL '{timedelta}'
      AND order_reviews.review_score <= 2
  ) AS INTEGER) AS bad_review_risk
FROM timestamp_df, sellers
WHERE EXISTS (
    SELECT 1 FROM order_items
    WHERE order_items.seller_id = sellers.seller_id
      AND purchase_ts > timestamp - INTERVAL '{timedelta}'
      AND purchase_ts <= timestamp
)
```
