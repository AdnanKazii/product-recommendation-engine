import pandas as pd

from src.recsys.evaluate import (
    build_eval_cases,
    evaluate_model,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
)


def test_build_eval_cases_skips_single_item_orders():
    test_order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o2"],  # o1 has 1 item, o2 has 2
            "product_id": ["A", "B", "C"],
        }
    )

    cases = build_eval_cases(test_order_items)

    # o1 contributes nothing; o2 contributes one case per item in the basket
    assert len(cases) == 2
    assert ("B", {"C"}) in cases
    assert ("C", {"B"}) in cases


def test_precision_recall_hit_rate_at_k():
    recommended = ["A", "B", "C", "D"]
    relevant = {"B", "D", "Z"}

    assert precision_at_k(recommended, relevant, k=4) == 2 / 4
    assert recall_at_k(recommended, relevant, k=4) == 2 / 3
    assert hit_rate_at_k(recommended, relevant, k=4) == 1.0

    assert hit_rate_at_k(["A", "C"], relevant, k=2) == 0.0


class _StubModel:
    def __init__(self, results: list[str]) -> None:
        self._results = results

    def recommend(self, product_id: str, k: int = 10) -> list[str]:
        return self._results[:k]


def test_evaluate_model_averages_across_cases():
    # every case has the same ground truth, model always returns the same list
    model = _StubModel(["A", "B"])
    eval_cases = [("q1", {"A"}), ("q2", {"Z"})]  # 1 hit, 1 miss

    result = evaluate_model(model, eval_cases, k=2)

    assert result["n_eval_cases"] == 2
    assert result["hit_rate_at_k"] == 0.5
