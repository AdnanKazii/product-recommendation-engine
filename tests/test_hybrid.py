from src.recsys.hybrid import HybridRecommender


class _StubModel:
    """Returns whatever list was configured, ignoring product_id."""

    def __init__(self, results: list[str]) -> None:
        self._results = results

    def recommend(self, product_id: str, k: int = 10) -> list[str]:
        return self._results[:k]


def test_uses_collaborative_results_when_they_fill_k():
    hybrid = HybridRecommender(
        collaborative=_StubModel(["c1", "c2", "c3"]),
        content=_StubModel(["x1", "x2", "x3"]),
    )

    assert hybrid.recommend("p", k=3) == ["c1", "c2", "c3"]


def test_backfills_with_content_when_collaborative_is_short():
    hybrid = HybridRecommender(
        collaborative=_StubModel(["c1"]),
        content=_StubModel(["c1", "x1", "x2", "x3"]),  # c1 overlaps, must be skipped
    )

    result = hybrid.recommend("p", k=3)

    assert result == ["c1", "x1", "x2"]


def test_falls_back_entirely_to_content_when_collaborative_is_empty():
    hybrid = HybridRecommender(
        collaborative=_StubModel([]),
        content=_StubModel(["x1", "x2"]),
    )

    assert hybrid.recommend("p", k=2) == ["x1", "x2"]
