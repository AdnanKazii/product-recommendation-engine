"""Item-item collaborative filtering from basket-level co-purchase data.

Built from *orders*, not customer purchase history - Day 2's EDA showed
96.9% of real customers only ever order once, so user-based collaborative
filtering has almost nothing to work with. Basket co-purchase ("this order
contained both A and B") is a signal that exists even for one-time buyers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


class CollaborativeRecommender:
    def fit(self, order_items_train: pd.DataFrame) -> "CollaborativeRecommender":
        pairs = order_items_train[["order_id", "product_id"]].drop_duplicates()

        self.product_ids_ = pairs["product_id"].unique()
        order_ids = pairs["order_id"].unique()
        self.id_to_idx_ = {pid: i for i, pid in enumerate(self.product_ids_)}
        order_to_idx = {oid: i for i, oid in enumerate(order_ids)}

        row = pairs["product_id"].map(self.id_to_idx_).to_numpy()
        col = pairs["order_id"].map(order_to_idx).to_numpy()
        data = np.ones(len(pairs))

        item_order_matrix = sparse.csr_matrix(
            (data, (row, col)), shape=(len(self.product_ids_), len(order_ids))
        )

        # L2-normalize each product's row so the dot product below is cosine similarity
        normalized = normalize(item_order_matrix, norm="l2", axis=1)
        self.similarity_ = (normalized @ normalized.T).tocsr()
        return self

    def recommend(self, product_id: str, k: int = 10) -> list[str]:
        idx = self.id_to_idx_.get(product_id)
        if idx is None:
            return []

        row = self.similarity_[idx].toarray().ravel()
        row[idx] = -1  # never recommend the product to itself

        top_idx = np.argsort(row)[::-1]
        results = []
        for i in top_idx[:k]:
            if row[i] <= 0:
                break  # zero similarity means "never co-purchased" - not a real signal
            results.append(self.product_ids_[i])
        return results
