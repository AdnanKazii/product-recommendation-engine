"""Content-based recommender: similarity by category, price, and review score.

This is the cold-start fallback. It doesn't need any co-purchase history at
all - just the attributes of the product itself - so it still works for
products the collaborative model has little or no signal for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class ContentBasedRecommender:
    def __init__(
        self,
        category_weight: float = 1.0,
        price_weight: float = 1.0,
        review_weight: float = 1.0,
    ) -> None:
        self.category_weight = category_weight
        self.price_weight = price_weight
        self.review_weight = review_weight

    def fit(self, product_features: pd.DataFrame) -> "ContentBasedRecommender":
        self.product_ids_ = product_features.index.to_numpy()
        self.id_to_idx_ = {pid: i for i, pid in enumerate(self.product_ids_)}

        category_dummies = pd.get_dummies(product_features["category"]).to_numpy(dtype=float)
        category_block = category_dummies * self.category_weight

        # log1p flattens the long price tail we saw in EDA before scaling
        price_scaled = StandardScaler().fit_transform(
            np.log1p(product_features[["price"]])
        )
        price_block = price_scaled * self.price_weight

        review_scaled = StandardScaler().fit_transform(product_features[["avg_review_score"]])
        review_block = review_scaled * self.review_weight

        self.feature_matrix_ = np.hstack([category_block, price_block, review_block])
        self.nn_ = NearestNeighbors(metric="cosine").fit(self.feature_matrix_)
        return self

    def recommend(self, product_id: str, k: int = 10) -> list[str]:
        idx = self.id_to_idx_.get(product_id)
        if idx is None:
            return []

        # ask for k+1 neighbors since the product is always its own closest match
        n_neighbors = min(k + 1, len(self.product_ids_))
        _, indices = self.nn_.kneighbors(
            self.feature_matrix_[idx : idx + 1], n_neighbors=n_neighbors
        )
        return [
            self.product_ids_[i] for i in indices[0] if self.product_ids_[i] != product_id
        ][:k]
