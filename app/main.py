from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.schemas import ProductOut, RecommendationResponse
from app.service import RecommenderService, build_recommender_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = build_recommender_service()
    yield


app = FastAPI(title="Product Recommendation Engine", lifespan=lifespan)


def get_service(request: Request) -> RecommenderService:
    return request.app.state.service


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/products", response_model=list[ProductOut])
def list_products(
    limit: int = 20,
    category: str | None = None,
    service: RecommenderService = Depends(get_service),
) -> list[ProductOut]:
    return service.top_products(limit=limit, category=category)


@app.get("/recommendations/{product_id}", response_model=RecommendationResponse)
def get_recommendations(
    product_id: str,
    k: int = 10,
    service: RecommenderService = Depends(get_service),
) -> RecommendationResponse:
    recommendations = service.recommend(product_id, k=k)
    if recommendations is None:
        raise HTTPException(status_code=404, detail=f"unknown product_id: {product_id}")
    return RecommendationResponse(query_product_id=product_id, recommendations=recommendations)
