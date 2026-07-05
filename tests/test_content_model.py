import pandas as pd

from src.recsys.content_model import ContentBasedRecommender


def _fixture_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["electronics", "electronics", "furniture"],
            "price": [100.0, 110.0, 900.0],
            "avg_review_score": [4.5, 4.6, 2.0],
        },
        index=["p1", "p2", "p3"],
    )


def test_recommends_the_more_similar_product_first():
    rec = ContentBasedRecommender().fit(_fixture_features())

    result = rec.recommend("p1", k=2)

    # p2 shares category/price/review with p1; p3 is a different category,
    # very different price, and much worse reviewed - p2 must rank first.
    assert result[0] == "p2"
    assert "p1" not in result


def test_unknown_product_returns_empty_list():
    rec = ContentBasedRecommender().fit(_fixture_features())

    assert rec.recommend("does-not-exist", k=5) == []


def test_k_limits_number_of_results():
    rec = ContentBasedRecommender().fit(_fixture_features())

    assert len(rec.recommend("p1", k=1)) == 1
