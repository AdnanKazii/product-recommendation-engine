"""End-to-end regression test against the real dataset.

Unlike the rest of the suite, this runs the full pipeline (real data -> all
four models -> evaluation) and checks the one invariant that actually
matters: the hybrid model must beat the popularity baseline. It's slower
and depends on data/ being populated (gitignored, not present in CI), so it
skips itself rather than failing when the data isn't there.
"""

from pathlib import Path

import pytest

from src.recsys.baseline import PopularityRecommender
from src.recsys.collab_model import CollaborativeRecommender
from src.recsys.content_model import ContentBasedRecommender
from src.recsys.data_prep import (
    build_order_items,
    build_product_features,
    load_raw_tables,
    temporal_train_test_split,
)
from src.recsys.evaluate import build_eval_cases, evaluate_model
from src.recsys.hybrid import HybridRecommender

DATA_DIR = Path(__file__).parent.parent / "data"
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "olist_orders_dataset.csv").exists(),
    reason="real Olist dataset not present (gitignored) - skipping integration test",
)


def test_hybrid_beats_baseline_on_real_data():
    tables = load_raw_tables(DATA_DIR)
    order_items = build_order_items(tables)
    train, test = temporal_train_test_split(order_items, test_frac=0.2)

    popularity = PopularityRecommender().fit(train)
    content = ContentBasedRecommender().fit(build_product_features(tables, train))
    collaborative = CollaborativeRecommender().fit(train)
    hybrid = HybridRecommender(collaborative=collaborative, content=content)

    eval_cases = build_eval_cases(test)
    assert len(eval_cases) > 0  # sanity: the test set must contain multi-item orders

    baseline_result = evaluate_model(popularity, eval_cases, k=10)
    hybrid_result = evaluate_model(hybrid, eval_cases, k=10)

    assert hybrid_result["hit_rate_at_k"] > baseline_result["hit_rate_at_k"]
    assert hybrid_result["recall_at_k"] > baseline_result["recall_at_k"]
