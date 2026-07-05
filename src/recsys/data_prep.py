"""Loading and preparing the Olist tables into the single order_items view
that every recommender in this project is built on."""

from pathlib import Path

import pandas as pd

REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
}


def load_raw_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {name: pd.read_csv(data_dir / filename) for name, filename in REQUIRED_FILES.items()}


def build_order_items(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (order, product), with purchase timestamp, price, and category attached.

    This is the base table every recommender (popularity, content, collaborative) is built from.
    """
    orders = tables["orders"][["order_id", "order_purchase_timestamp"]]
    products = tables["products"].merge(
        tables["category_translation"], on="product_category_name", how="left"
    )[["product_id", "product_category_name_english"]]

    order_items = tables["order_items"][["order_id", "product_id", "price"]]
    order_items = order_items.merge(orders, on="order_id", how="left")
    order_items = order_items.merge(products, on="product_id", how="left")
    order_items["order_purchase_timestamp"] = pd.to_datetime(
        order_items["order_purchase_timestamp"]
    )
    return order_items.dropna(subset=["order_purchase_timestamp", "product_id"]).reset_index(
        drop=True
    )


def temporal_train_test_split(
    order_items: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by order timestamp, not randomly.

    A random row-level split would leak the future into training: co-purchase
    stats or content features computed on a randomly-sampled training set could
    include orders that happened *after* a test order, which the model would
    never have known about in reality. Splitting by time, with every item of a
    given order kept on the same side of the cut, avoids that leakage.
    """
    order_times = order_items.groupby("order_id")["order_purchase_timestamp"].min().sort_values()
    cutoff_idx = int(len(order_times) * (1 - test_frac))
    cutoff_time = order_times.iloc[cutoff_idx]

    train = order_items[order_items["order_purchase_timestamp"] < cutoff_time]
    test = order_items[order_items["order_purchase_timestamp"] >= cutoff_time]
    return train.reset_index(drop=True), test.reset_index(drop=True)
