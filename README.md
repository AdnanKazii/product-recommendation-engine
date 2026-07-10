# Product Recommendation Engine

A hybrid product recommender — collaborative filtering + content-based fallback — built end-to-end on a real e-commerce dataset: EDA, model design, offline evaluation, a FastAPI backend, a web demo, and a live deployment.

**Live demo:** [product-recommendation-engine-t575.onrender.com](https://product-recommendation-engine-t575.onrender.com/)
*(Hosted on Render's free tier, which spins down after ~15 minutes idle — the first request after a while may take 30-60s to wake up and re-fetch the dataset. Give it a minute.)*

## The problem

Given a product, recommend others a customer is likely to want alongside it — the "customers who bought this also bought…" pattern that drives a meaningful share of e-commerce revenue.

## Why this dataset breaks the textbook approach

Built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~99k real orders, 2016-2018). EDA surfaced two hard constraints that shaped every modeling decision:

- **96.9% of customers only ever ordered once** (measured correctly on the person-level `customer_unique_id`, not the per-order `customer_id` — a schema gotcha in this dataset). Classic user-based collaborative filtering ("people like you also liked...") has almost nothing to work with.
- **90% of orders contain exactly one item.** Item-item co-purchase signal is real but sparse — it only exists in the remaining 10%.

Rather than force a standard approach onto data that doesn't support it, the model is designed around these constraints directly.

## Model design

| Model | Approach |
|---|---|
| **Popularity baseline** | Ignores the query product, returns global bestsellers. The floor every other model must beat. |
| **Content-based** | Cosine similarity over category (one-hot), log-scaled price, and average review score. Served via `sklearn.neighbors.NearestNeighbors` rather than a full ~33k×33k similarity matrix (would need several GB just to store). Works for any product regardless of purchase history — the cold-start fallback. |
| **Collaborative (item-item)** | Built from *baskets*, not customer history — order co-occurrence, not user history — since basket data exists even for one-time buyers. Sparse product×order matrix, similarity via sparse matrix multiplication (only ~33k non-zero entries vs. 709M if dense). Only has signal for the ~15% of products that were ever co-purchased with something else. |
| **Hybrid** | Collaborative results first (reflects real buying behavior), content-based backfill for any remaining slots. A cascade rather than a weighted score blend, since the two similarity spaces aren't on comparable scales without calibration. |

## Evaluation

Offline evaluation via **leave-one-out basket completion**: for every multi-item order in a *temporally held-out* test set (last 20% of orders by timestamp — a random split would leak future co-purchase data into training), each item takes a turn as the query and the rest of the basket is the ground truth. Only multi-item orders are usable, consistent with the ~10% basket-size constraint above.

Results at k=10, real held-out data:

| Model | Precision@10 | Recall@10 | Hit Rate@10 |
|---|---|---|---|
| Popularity baseline | 0.0014 | 0.0127 | 0.0141 |
| Content-based | 0.0041 | 0.0324 | 0.0394 |
| Collaborative | 0.0072 | 0.0571 | 0.0661 |
| **Hybrid** | **0.0097** | **0.0763** | **0.0879** |

The hybrid beats every individual model on every metric — **hit rate is ~6.2x the popularity baseline**. These are small absolute numbers, which is expected and worth stating plainly: this is a genuinely hard, sparse dataset, and the point of the evaluation is the *relative* lift from each modeling decision, not chasing a large absolute score.

## Architecture

```
Olist CSVs → src/recsys (data prep, 4 models, evaluation)
                    ↓
        app/service.py (fits models on full data at startup)
                    ↓
        app/main.py (FastAPI: /products, /recommendations/{id}, /)
                    ↓
        Vanilla HTML/CSS/JS frontend (Jinja2 + fetch)
                    ↓
        Docker → Render (dataset fetched from Kaggle at container startup)
```

## Tech stack

Python, pandas, scikit-learn, scipy (sparse matrices), FastAPI, Jinja2, pytest, Docker, GitHub Actions, Render.

## Running locally

```bash
python -m venv .venv
.venv/Scripts/activate  # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# requires a Kaggle account + API token at ~/.kaggle/kaggle.json (or KAGGLE_API_TOKEN env var)
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/ --unzip

uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

### Tests

```bash
pytest tests/           # fast unit + API tests, no data required
ruff check app/ src/ tests/
```

`tests/test_integration.py` additionally runs the full pipeline against the real dataset and checks the hybrid model beats the baseline — it skips itself automatically if `data/` isn't populated.

### Docker

```bash
docker build -t product-recommender .
docker run -p 7860:7860 -e KAGGLE_API_TOKEN=your_token product-recommender
```

## Project structure

```
src/recsys/       # data prep, baseline/content/collaborative/hybrid models, evaluation
app/               # FastAPI backend + service layer + frontend (templates/static)
tests/             # unit tests (fixtures), API tests, integration test
notebooks/         # exploratory EDA
render.yaml        # infrastructure-as-code deploy blueprint
Dockerfile
```

## Known limitations & how I'd scale this further

- **Collaborative coverage is only ~15%** of the catalog — the honest constraint of this dataset, not a modeling bug. A larger dataset with denser baskets would close most of the gap between the collaborative and hybrid numbers above.
- **Retrained on every startup** rather than loading a persisted artifact — fine at this scale (fits in well under a second), but at real scale I'd train offline on a schedule and serve a versioned, pre-fitted model artifact instead.
- **NearestNeighbors over ~33k products** is fast enough here; at millions of items I'd reach for an approximate nearest-neighbor index (e.g. FAISS) instead of exact brute-force search.
- **Offline metrics only** — in production I'd want online A/B testing (click-through / add-to-cart rate on recommended items) alongside these offline numbers, since offline metrics on historical data don't always translate directly to live user behavior.

## Data license

The Olist dataset is CC-BY-NC-SA-4.0 (non-commercial, share-alike). Rather than redistribute a copy of it, the deployed app fetches it directly from Kaggle at container startup — `data/` is gitignored and not included in this repo or its Docker image.
