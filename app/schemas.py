from pydantic import BaseModel


class ProductOut(BaseModel):
    product_id: str
    category: str
    price: float
    avg_review_score: float | None = None


class RecommendationResponse(BaseModel):
    query_product_id: str
    recommendations: list[ProductOut]
