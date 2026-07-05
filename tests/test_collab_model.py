import pandas as pd

from src.recsys.collab_model import CollaborativeRecommender


def _fixture_order_items() -> pd.DataFrame:
    # o1: {A, B}   o2: {A, B}   o3: {A, C}   o4: {B, D}
    # A & B co-occur twice -> highest similarity
    # A & C co-occur once  -> some similarity
    # A & D never co-occur -> zero similarity, must be excluded
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o2", "o3", "o3", "o4", "o4"],
            "product_id": ["A", "B", "A", "B", "A", "C", "B", "D"],
        }
    )


def test_recommends_most_frequently_co_purchased_item_first():
    rec = CollaborativeRecommender().fit(_fixture_order_items())

    result = rec.recommend("A", k=3)

    assert result[0] == "B"  # co-occurs with A twice
    assert "C" in result  # co-occurs with A once
    assert "D" not in result  # never co-occurs with A


def test_unknown_product_returns_empty_list():
    rec = CollaborativeRecommender().fit(_fixture_order_items())

    assert rec.recommend("does-not-exist", k=5) == []


def test_product_with_no_co_purchases_returns_empty_list():
    order_items = pd.DataFrame(
        {"order_id": ["o1", "o2"], "product_id": ["A", "B"]}
    )  # A and B each only ever appear alone
    rec = CollaborativeRecommender().fit(order_items)

    assert rec.recommend("A", k=5) == []
