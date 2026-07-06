import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_service
from app.service import RecommenderService


class _StubHybrid:
    def recommend(self, product_id: str, k: int = 10) -> list[str]:
        return ["p2", "p3"][:k]


@pytest.fixture
def client():
    product_lookup = pd.DataFrame(
        {
            "category": ["electronics", "electronics", "furniture"],
            "price": [100.0, 110.0, 900.0],
            "avg_review_score": [4.5, 4.6, 2.0],
            "purchase_count": [5, 20, 1],
        },
        index=["p1", "p2", "p3"],
    )
    fake_service = RecommenderService(hybrid=_StubHybrid(), product_lookup=product_lookup)

    app.dependency_overrides[get_service] = lambda: fake_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_products_sorted_by_popularity(client):
    response = client.get("/products?limit=2")
    assert response.status_code == 200
    body = response.json()
    assert [p["product_id"] for p in body] == ["p2", "p1"]  # purchase_count 20, then 5


def test_list_products_filters_by_category(client):
    response = client.get("/products?category=furniture")
    assert response.status_code == 200
    body = response.json()
    assert [p["product_id"] for p in body] == ["p3"]


def test_recommendations_for_known_product(client):
    response = client.get("/recommendations/p1?k=2")
    assert response.status_code == 200
    body = response.json()
    assert body["query_product_id"] == "p1"
    assert [p["product_id"] for p in body["recommendations"]] == ["p2", "p3"]


def test_recommendations_for_unknown_product_returns_404(client):
    response = client.get("/recommendations/does-not-exist")
    assert response.status_code == 404
