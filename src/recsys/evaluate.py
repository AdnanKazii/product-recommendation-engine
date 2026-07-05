"""Offline evaluation via leave-one-out basket completion.

For every multi-item order in the held-out test set, each product in the
basket takes a turn as the "query" and the rest of the basket becomes the
ground-truth relevant set. A model is scored by whether/how well its
recommend(query, k) call surfaces those other basket items. Only baskets
with 2+ items can be used - Day 2 showed that's ~10% of orders.
"""

from __future__ import annotations

import pandas as pd


def build_eval_cases(test_order_items: pd.DataFrame) -> list[tuple[str, set[str]]]:
    baskets = test_order_items.groupby("order_id")["product_id"].apply(
        lambda s: list(dict.fromkeys(s))  # de-dupe while preserving order
    )

    cases = []
    for products in baskets:
        if len(products) < 2:
            continue
        for i, query in enumerate(products):
            relevant = set(products[:i] + products[i + 1 :])
            cases.append((query, relevant))
    return cases


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    # divide by k, not by however many items were actually returned - a model
    # that returns fewer than k items should be penalized for the empty slots,
    # not rewarded with a smaller denominator
    top_k = recommended[:k]
    hits = sum(1 for p in top_k if p in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    return sum(1 for p in top_k if p in relevant) / len(relevant)


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    return 1.0 if any(p in relevant for p in top_k) else 0.0


def evaluate_model(model, eval_cases: list[tuple[str, set[str]]], k: int = 10) -> dict:
    precisions, recalls, hits = [], [], []
    for query, relevant in eval_cases:
        recommended = model.recommend(query, k=k)
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        hits.append(hit_rate_at_k(recommended, relevant, k))

    n = len(eval_cases)
    return {
        "precision_at_k": sum(precisions) / n,
        "recall_at_k": sum(recalls) / n,
        "hit_rate_at_k": sum(hits) / n,
        "n_eval_cases": n,
    }
