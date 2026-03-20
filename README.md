# AI Commerce Agent

A full-stack AI shopping assistant built with React and FastAPI. It supports conversational product discovery, image-based retrieval, hybrid retrieval, multimodal search, and lightweight personalization on top of a processed fashion product catalog.

![AI Commerce Agent Demo](./demo.gif)

## Features

- conversational recommendation
- multimodal retrieval
- vector search with FAISS
- hybrid search over text and images
- lightweight personalization and memory
- a user-facing product assistant UI

## Architecture

### Frontend

- React + Vite
- Material UI
- framer-motion
- react-markdown

### Backend

- FastAPI
- OpenAI chat + embedding APIs
- CLIP image embeddings via Hugging Face Transformers
- FAISS similarity search
- file-based retrieval index artifacts
- SQLite-backed lightweight personalization

## Repository Layout

```text
.
|- backend/            FastAPI app, retrieval logic, config, data prep, index builder
|- frontend/           React UI
|- dataset/            Local Kaggle dataset drop-in folder
|- scripts/            Local dev scripts
|- docker-compose.yml  Multi-container local setup
|- Makefile            Common developer commands
`- README.md
```

## Quick Start

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` unless you want mock mode.
3. Install dependencies:

```bash
make install
```

4. Put the Kaggle dataset in `dataset/` so the repo contains:

```text
dataset/
|- styles.csv
`- images/
```

Dataset reference:

- `Fashion Product Images (Small)` on Kaggle
- original link: `https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small`

5. Prepare a cleaned catalog from the Kaggle dataset:

```bash
make prepare-catalog
make build-indexes
```

6. Start the backend and frontend:

```bash
make dev
```

You can still run the script directly:

```bash
./scripts/run.sh
```

The backend starts quickly and then warms up retrieval resources in the background. Text chat is usually ready first, while the first image-heavy request may still be a little slower on a cold start.

## Common Workflows

Small local test set:

```bash
make prepare-catalog-small
make build-indexes
make dev
```

Default Kaggle subset:

```bash
make prepare-catalog
make build-indexes
make dev
```

Larger local experiment:

```bash
make prepare-catalog-large
make rebuild-indexes
make dev
```

Bootstrap everything in one command:

```bash
PREPARE_CATALOG_ON_RUN=true make dev
```

## Run Script Automation

The run script supports lightweight automation through `.env`:

- `PREPARE_CATALOG_ON_RUN=true` to regenerate the catalog before startup
- `BUILD_INDEXES_ON_RUN=true` to build or refresh indexes before startup
- `FORCE_REBUILD_INDEXES_ON_RUN=true` for a forced rebuild
- `KAGGLE_MAX_PRODUCTS`, `KAGGLE_MAX_PER_ARTICLE`, `KAGGLE_MIN_ARTICLE_COUNT`, `KAGGLE_SAMPLE_SEED`
- `BACKEND_READY_PATH` and `BACKEND_READY_TIMEOUT` to control how `run.sh` waits for backend readiness
- `HF_TOKEN` to reduce Hugging Face rate-limit warnings and speed up cached model access
- `THIRD_PARTY_LOG_LEVEL` to quiet noisy library logs such as `httpx` and `huggingface_hub`

If you want backend hot reload during development:

```bash
BACKEND_RELOAD=true make dev-reload
```

## How Index Building Works

- `make dev` calls `backend.build_indexes` automatically by default
- if indexes are already current, the build step is skipped
- first run is slower because text embeddings and image embeddings have to be created
- `make rebuild-indexes` should be used only when you intentionally want a full rebuild

In practice:

- use `make build-indexes` for normal incremental behavior
- use `make rebuild-indexes` after major catalog or model changes
- avoid forced rebuilds unless something actually changed

## Data Pipeline

The active catalog lives at [backend/data/catalog/products.json](/c:/Users/wilso/Documents/GitHub/AI-Commerce-Agent/backend/data/catalog/products.json).

The Kaggle preparation step:

- filters the raw fashion dataset to categories that fit the shopping assistant better
- keeps only rows with available images
- samples a balanced subset instead of blindly ingesting all 44k items
- enriches products with inferred metadata such as brand, style, material, price band, rating, and inventory status

## Docker

Docker support is included for reproducible local setup and easier handoff, but the containers still expect the Kaggle dataset to be present in `dataset/`.

```bash
docker compose up --build
```

Frontend: `http://localhost:3000`  
Backend: `http://localhost:8000`
