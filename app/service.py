"""Wraps the fitted hybrid recommender + product metadata behind a small
interface the API routes depend on. Kept separate from main.py so it can be
swapped out in tests via FastAPI's dependency override mechanism."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.recsys.collab_model import CollaborativeRecommender
from src.recsys.content_model import ContentBasedRecommender
from src.recsys.data_prep import build_order_items, build_product_features, load_raw_tables
from src.recsys.hybrid import HybridRecommender

from app.schemas import ProductOut

DATA_DIR = Path(__file__).parent.parent / "data"


class RecommenderService:
    def __init__(self, hybrid: HybridRecommender, product_lookup: pd.DataFrame) -> None:
        self.hybrid = hybrid
        # indexed by product_id, columns: category, price, avg_review_score, purchase_count
        self.product_lookup = product_lookup

    def get_product(self, product_id: str) -> ProductOut | None:
        if product_id not in self.product_lookup.index:
            return None
        row = self.product_lookup.loc[product_id]
        return ProductOut(
            product_id=product_id,
            category=row["category"],
            price=float(row["price"]),
            avg_review_score=(
                float(row["avg_review_score"]) if pd.notna(row["avg_review_score"]) else None
            ),
        )

    def top_products(self, limit: int = 20, category: str | None = None) -> list[ProductOut]:
        df = self.product_lookup
        if category is not None:
            df = df[df["category"] == category]
        top_ids = df.sort_values("purchase_count", ascending=False).index[:limit]
        return [self.get_product(pid) for pid in top_ids]

    def recommend(self, product_id: str, k: int = 10) -> list[ProductOut] | None:
        if product_id not in self.product_lookup.index:
            return None
        rec_ids = self.hybrid.recommend(product_id, k=k)
        return [self.get_product(pid) for pid in rec_ids]


def build_recommender_service(data_dir: Path = DATA_DIR) -> RecommenderService:
    tables = load_raw_tables(data_dir)
    order_items = build_order_items(tables)

    # served model trains on everything we have - the temporal split is only
    # for the offline evaluation numbers, not for the live recommender
    features = build_product_features(tables, order_items)
    features["purchase_count"] = order_items["product_id"].value_counts()

    content = ContentBasedRecommender().fit(features)
    collaborative = CollaborativeRecommender().fit(order_items)
    hybrid = HybridRecommender(collaborative=collaborative, content=content)

    return RecommenderService(hybrid=hybrid, product_lookup=features)
