# Backend

The backend serves the shopping assistant API, product image files, and retrieval pipeline.

## Responsibilities

- conversational routing and response generation
- text-based product retrieval
- image-based product retrieval
- multimodal retrieval
- index artifact management
- lightweight in-memory conversation history
- lightweight persistent personalization via SQLite
- health and readiness endpoints

## Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/healthz` | GET | Liveness check |
| `/readyz` | GET | Readiness check including retrieval indexes |
| `/chat` | POST | Text chat and product recommendation |
| `/search-image` | POST | Image-based product search |
| `/search-similar-product` | POST | Reuse a catalog product image for similar-item search |
| `/chat-multimodal` | POST | Text + image multimodal search |
| `/reset-memory` | POST | Reset conversation memory for the current session |

## Configuration

Environment variables are read from the project root `.env`.

Important settings:

- `OPENAI_API_KEY`
- `USE_MOCK_OPENAI`
- `CHAT_MODEL`
- `TEXT_EMBEDDING_MODEL`
- `CLIP_MODEL_NAME`
- `HF_TOKEN`
- `THIRD_PARTY_LOG_LEVEL`
- `ALLOWED_ORIGINS`
- `AUTO_BUILD_INDEXES`
- `FORCE_REBUILD_INDEXES`
- `ENABLE_BACKGROUND_WARMUP`
- `WARMUP_CLIP_MODEL`
- `ENABLE_PERSONALIZATION`

Start from the root `.env.example`.

## Local Setup

```bash
pip install -r requirements.txt
python -m backend.prepare_kaggle_catalog
python -m backend.build_indexes
python -m uvicorn backend.main:app
```

## Retrieval Indexes

Indexes are built into `backend/.artifacts/indexes/`.

- `python -m backend.build_indexes` builds them when missing or stale
- `python -m backend.build_indexes --force` rebuilds them explicitly
- app startup stays lightweight; indexes load lazily on readiness checks or first retrieval request
- by default, the backend also starts a background warmup thread to preload retrieval resources after startup

## Catalog Data

The active catalog now lives at `backend/data/catalog/products.json`.

- `python -m backend.prepare_kaggle_catalog` generates a cleaned catalog from `dataset/styles.csv`
- default output is a balanced subset sized for local development and manageable embedding costs
- use `python -m backend.prepare_kaggle_catalog --max-products 3000` to scale up gradually
- the processed dataset includes richer metadata such as inferred brand, color, material, style, season, price band, ratings, and inventory status

## Notes

- Product images are served from `dataset/images/`.
- The current memory layer is lightweight; conversation turns stay in memory and personalization is persisted in SQLite.
- Product assets in this repo should be treated as demo assets unless you replace them with your own licensed files.
