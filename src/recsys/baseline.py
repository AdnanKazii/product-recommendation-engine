"""Popularity baseline recommender.

This is the floor every other model in this project has to beat. It ignores
the input product entirely and just returns the globally best-selling
products. Any smarter recommender that can't outperform this isn't actually
using the "recommendation" part of its logic.
"""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    def __init__(self) -> None:
        self.ranked_products_: list[str] = []

    def fit(self, order_items_train: pd.DataFrame) -> "PopularityRecommender":
        self.ranked_products_ = order_items_train["product_id"].value_counts().index.tolist()
        return self

    def recommend(self, product_id: str | None = None, k: int = 10) -> list[str]:
        """Return the top-k best-selling products, excluding product_id if given."""
        candidates = (
            self.ranked_products_
            if product_id is None
            else [p for p in self.ranked_products_ if p != product_id]
        )
        return candidates[:k]
