from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.data_bootstrap import ensure_data_available
from app.schemas import ProductOut, RecommendationResponse
from app.service import DATA_DIR, RecommenderService, build_recommender_service

APP_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_available(DATA_DIR)
    app.state.service = build_recommender_service()
    yield


app = FastAPI(title="Product Recommendation Engine", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def get_service(request: Request) -> RecommenderService:
    return request.app.state.service


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


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
