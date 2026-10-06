"""Gate module for RelPilot: backtest-derived action thresholds and outbox logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

DEFAULT_MIN_SUPPORT = 20
DEFAULT_MIN_PRECISION = 0.8


def find_act_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = DEFAULT_MIN_PRECISION,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> tuple[float | None, int, float | None]:
    """Find the lowest decision threshold that satisfies precision and support constraints.

    Args:
        y_true: 1D array of ground truth binary labels (0 or 1).
        y_prob: 1D array of predicted probabilities in [0, 1].
        min_precision: Minimum acceptable precision target (e.g. 0.8).
        min_support: Minimum number of positive cases required above threshold.

    Returns:
        (lowest_threshold, support_count, achieved_precision)
        If no threshold qualifies, returns (None, 0, None).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0 or len(y_prob) == 0 or len(y_true) != len(y_prob):
        return None, 0, None

    # Unique thresholds sorted ascending
    thresholds = np.unique(y_prob)
    valid_thresholds: list[tuple[float, int, float]] = []

    for t in thresholds:
        mask = y_prob >= t
        count = int(np.sum(mask))
        if count < min_support:
            # Further higher thresholds will only have fewer or equal samples
            continue

        true_positives = int(np.sum(y_true[mask] == 1))
        precision = float(true_positives / count) if count > 0 else 0.0

        if precision >= min_precision:
            valid_thresholds.append((float(t), count, precision))

    if not valid_thresholds:
        return None, 0, None

    # Choose the lowest threshold among qualifying ones (maximizes automated action volume)
    valid_thresholds.sort(key=lambda x: x[0])
    best = valid_thresholds[0]
    return best[0], best[1], best[2]


def partition_actions(
    preds_df: pd.DataFrame,
    entity_col: str,
    pred_col: str,
    action_name: str,
    run_id: str,
    threshold: float | None,
    achieved_precision: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition entity predictions into proposed actions and review list.

    Returns:
        (proposed_actions, needs_review_actions)
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    proposed: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []

    if threshold is None:
        for _, row in preds_df.iterrows():
            needs_review.append(
                {
                    "entity_id": str(row[entity_col]),
                    "score": round(float(row[pred_col]), 4),
                    "reason": "No backtest threshold met required precision with minimum support",
                }
            )
        return proposed, needs_review

    for _, row in preds_df.iterrows():
        score = float(row[pred_col])
        entity_id = str(row[entity_col])
        if score >= threshold:
            proposed.append(
                {
                    "run_id": run_id,
                    "entity_id": entity_id,
                    "action": action_name,
                    "score": round(score, 4),
                    "threshold": round(threshold, 4),
                    "backtest_precision": round(achieved_precision, 4) if achieved_precision else None,
                    "status": "proposed",
                    "created_at": now_iso,
                }
            )
        else:
            needs_review.append(
                {
                    "entity_id": entity_id,
                    "score": round(score, 4),
                    "reason": f"Confidence {score:.4f} below backtest threshold {threshold:.4f}",
                }
            )

    return proposed, needs_review


def record_outbox(
    proposed_actions: list[dict[str, Any]],
    outbox_file: str | Path = "outbox.jsonl",
) -> int:
    """Append proposed actions to the audit outbox."""
    if not proposed_actions:
        return 0

    path = Path(outbox_file)
    with open(path, "a", encoding="utf-8") as f:
        for action in proposed_actions:
            f.write(json.dumps(action) + "\n")

    return len(proposed_actions)
