import pandas as pd

from src.recsys.data_prep import temporal_train_test_split


def test_temporal_split_keeps_orders_intact_and_time_ordered():
    order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3", "o4", "o5"],
            "product_id": ["p1", "p2", "p1", "p2", "p3", "p1"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-02",
                    "2018-01-03",
                    "2018-01-04",
                    "2018-01-05",
                ]
            ),
        }
    )

    train, test = temporal_train_test_split(order_items, test_frac=0.4)

    # no order should have its items split across both sides
    assert set(train["order_id"]) & set(test["order_id"]) == set()
    # every train timestamp must be strictly before every test timestamp
    assert train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()
    # no rows lost
    assert len(train) + len(test) == len(order_items)
