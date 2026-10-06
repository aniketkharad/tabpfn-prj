"""Data preparation script for RelPilot demo workspace.

Verifies raw Olist files and derives the leak-free `order_items.csv` with event `purchase_ts`.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def prepare_demo_data(data_dir: str | Path = "workspaces/demo/data") -> bool:
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[Error] Data directory not found: {data_path.resolve()}", file=sys.stderr)
        print("Please ensure Olist CSV files are present or download them via kagglehub.", file=sys.stderr)
        return False

    required_raw = [
        "olist_sellers_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_products_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "product_category_name_translation.csv",
    ]

    missing = [f for f in required_raw if not (data_path / f).exists()]
    if missing:
        print(f"[Warning] Missing raw Olist tables in {data_path}: {missing}", file=sys.stderr)
        print("Download them from Kaggle (olistbr/brazilian-ecommerce) or copy to workspaces/demo/data/", file=sys.stderr)
        return False

    target_items = data_path / "order_items.csv"
    if not target_items.exists():
        print(f"Generating derived table: {target_items}...", file=sys.stderr)
        items_df = pd.read_csv(data_path / "olist_order_items_dataset.csv")
        orders_df = pd.read_csv(data_path / "olist_orders_dataset.csv")[["order_id", "order_purchase_timestamp"]]

        # Join order purchase timestamp to order items to create temporal event column
        merged = items_df.merge(orders_df, on="order_id", how="inner")
        merged = merged.rename(columns={"order_purchase_timestamp": "purchase_ts"})
        merged.to_csv(target_items, index=False)
        print(f"Created {target_items} ({len(merged)} rows).", file=sys.stderr)
    else:
        print(f"Derived table already exists: {target_items}", file=sys.stderr)

    print("Demo dataset ready.", file=sys.stderr)
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "workspaces/demo/data"
    success = prepare_demo_data(target)
    sys.exit(0 if success else 1)
