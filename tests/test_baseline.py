import pandas as pd

from src.recsys.baseline import PopularityRecommender


def test_ranks_products_by_purchase_frequency():
    order_items = pd.DataFrame({"product_id": ["A", "A", "A", "B", "B", "C"]})
    rec = PopularityRecommender().fit(order_items)

    assert rec.recommend(k=3) == ["A", "B", "C"]


def test_excludes_the_query_product_from_its_own_recommendations():
    order_items = pd.DataFrame({"product_id": ["A", "A", "B", "C"]})
    rec = PopularityRecommender().fit(order_items)

    result = rec.recommend(product_id="A", k=2)

    assert "A" not in result
    assert result == ["B", "C"]


def test_recommend_with_no_product_id_returns_pure_popularity_ranking():
    order_items = pd.DataFrame({"product_id": ["A", "B", "B", "C", "C", "C"]})
    rec = PopularityRecommender().fit(order_items)

    assert rec.recommend(k=2) == ["C", "B"]
