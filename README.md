# ShopSense

A full-stack AI shopping assistant built with React and FastAPI. It supports conversational product discovery, image-based retrieval, hybrid retrieval, multimodal search, and lightweight personalization on top of a processed fashion product catalog.

![ShopSense Demo](./demo.gif)

## Highlights

- text, image, and multimodal product retrieval
- hybrid search over OpenAI text embeddings, CLIP image embeddings, and lexical matching
- second-stage reranking with intent alignment and optional LLM reranking
- conversational shopping UX with lightweight personalization and memory
- offline retrieval benchmark plus regression-style tests

## Benchmark Snapshot

Current baseline on the bundled 18-query benchmark:

| Metric | Score |
| --- | ---: |
| Strict Hit@3 | 0.944 |
| Strict Hit@5 | 0.944 |
| Strict MRR | 0.889 |
| Semantic Category Hit@3 | 1.000 |
| Semantic Subcategory Hit@3 | 1.000 |

More detail:

- [docs/results.md](./docs/results.md)
- [docs/architecture.md](./docs/architecture.md)
- [eval/eval_retrieval.py](./eval/eval_retrieval.py)
- [tests](./tests)

## What It Demonstrates

- conversational recommendation
- multimodal retrieval
- vector search with FAISS
- hybrid search over text and images
- second-stage reranking
- optional LLM-based reranking for multimodal or ambiguous retrieval cases
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

More implementation detail is documented in [docs/architecture.md](./docs/architecture.md).

## Demo Scenarios

- Text search: broad shopping queries like running shoes, jackets, backpacks, or shorts.
- Image search: upload a catalog-like product image and retrieve visually similar items.
- Multimodal search: start from an image and shift intent with text such as `better for running` or `better for basketball`.

## Repository Layout

```text
.
|- backend/            FastAPI app, retrieval logic, config, data prep, index builder
|- frontend/           React UI
|- eval/               Retrieval benchmark queries and evaluation script
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

Useful development toggles:

- `PREPARE_CATALOG_ON_RUN=true` to regenerate the catalog before startup
- `BUILD_INDEXES_ON_RUN=true` to build or refresh indexes before startup
- `FORCE_REBUILD_INDEXES_ON_RUN=true` for a forced rebuild
- `HF_TOKEN` to reduce Hugging Face warnings and speed up cached model access
- `THIRD_PARTY_LOG_LEVEL` to quiet noisy library logs

For backend hot reload during development:

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

The active catalog lives at [backend/data/catalog/products.json](./backend/data/catalog/products.json).

The Kaggle preparation step:

- filters the raw fashion dataset to categories that fit the shopping assistant better
- keeps only rows with available images
- samples a balanced subset instead of blindly ingesting all 44k items
- enriches products with inferred metadata such as brand, style, material, price band, rating, and inventory status

## Evaluation

The repository includes a lightweight offline retrieval benchmark under [eval/queries.jsonl](./eval/queries.jsonl) and [eval/eval_retrieval.py](./eval/eval_retrieval.py).

Run it with:

```bash
make eval
make test
```

The script evaluates text, image, and multimodal queries and reports:

- strict `Hit@3`
- strict `Hit@5`
- strict `MRR`
- semantic category hit rate
- semantic subcategory hit rate
- semantic keyword hit rate
- average latency

You can also save the full results to a JSON file:

```bash
PYTHONPATH=. python eval/eval_retrieval.py --output eval/results/latest.json
```

More detail is documented in [docs/results.md](./docs/results.md).

## Docker

Docker support is included for reproducible local setup and easier handoff, but the containers still expect the Kaggle dataset to be present in `dataset/`.

```bash
make docker-build
make docker-up
```

Frontend: `http://localhost:3000`  
Backend: `http://localhost:8000`

The backend image installs the CPU-only PyTorch wheel on purpose. That keeps the container build much smaller and more reliable than pulling the default Linux CUDA stack, which is unnecessary for the current Docker workflow.

The default flow is:

```bash
make docker-build
make docker-up
```

If Docker BuildKit is flaky in your WSL setup, the recommended fallback is to keep using the default `docker-up` target and switch only the build step:

```bash
make docker-build-nobuildkit
make docker-up
```
