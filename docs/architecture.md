# Architecture

This project is a multimodal shopping assistant built around a small full-stack retrieval system:

- a React frontend for chat, product browsing, and image upload
- a FastAPI backend for orchestration and retrieval
- offline catalog preparation and index building scripts
- local FAISS indexes for dense retrieval
- CLIP-based image embeddings for image and multimodal search
- a dedicated second-stage reranker over the merged candidate set
- an optional LLM reranker for multimodal or otherwise ambiguous cases
- lightweight session personalization stored in SQLite

## High-Level System

```text
Kaggle dataset
  -> catalog preparation
  -> processed product catalog
  -> index build pipeline
  -> text/image embedding artifacts

Frontend (React + Vite)
  -> /chat
  -> /search-image
  -> /chat-multimodal
  -> /search-similar-product

Backend (FastAPI)
  -> agent orchestration
  -> retrieval fusion
  -> recommendation explanation
  -> personalization updates

Storage
  -> backend/.artifacts/indexes
  -> backend/.artifacts/personalization.db
```

## Main Components

### Frontend

The frontend lives under [frontend/](../frontend/).

Key responsibilities:

- render the chat-style shopping assistant UI
- support text chat, image upload, and multimodal messages
- display recommendation cards
- trigger `Search Similar` flows from existing product cards

Important user flows:

- text-only recommendation query
- image-only search
- text + image multimodal search
- “search similar” from a returned product card

### Backend API

The backend entrypoint is [backend/main.py](../backend/main.py).

Current API surface:

- `POST /chat`
- `POST /search-image`
- `POST /chat-multimodal`
- `POST /search-similar-product`
- `POST /reset-memory`
- `GET /healthz`
- `GET /readyz`

`/readyz` verifies that retrieval indexes can be loaded before the frontend starts.

### Agent Layer

The main orchestration logic lives in [backend/agent.py](../backend/agent.py).

Responsibilities:

- classify text requests into recommendation vs general chat
- rewrite vague shopping queries into clearer retrieval queries
- dispatch text, image, and multimodal retrieval
- generate shopper-facing recommendation summaries
- update short conversation memory
- update lightweight personalization state

The backend currently uses:

- OpenAI chat completions for text intent classification, query clarification, and recommendation summaries
- grounded deterministic summaries for image-heavy and multimodal retrieval explanations where consistency matters more than freeform phrasing

### Retrieval Layer

The retrieval and scoring logic lives in [backend/retrieval.py](../backend/retrieval.py).

The current retrieval stack combines:

- lexical matching over product text
- dense text retrieval from OpenAI embeddings
- image retrieval from CLIP image embeddings
- weighted hybrid fusion
- heuristic reranking bonuses from matched features/tags/use cases
- query-intent alignment rules
- lightweight personalization bonuses

The retrieval layer supports three modes:

- `text`
- `image`
- `multimodal`

For multimodal retrieval, the scorer changes its weighting depending on whether the text looks like a generic visual query such as `find something like this` or a stronger intent shift such as `better for running`.

The current architecture is a two-stage retrieval pipeline:

1. first-stage candidate generation through lexical, dense text, and image retrieval
2. second-stage reranking with query-intent alignment, candidate filtering heuristics, grounded match reasons, and lightweight personalization

An optional third refinement step can be enabled through configuration:

3. LLM reranking over the top candidate subset for multimodal or ambiguous retrieval paths

### Embeddings and Indexes

Indexing and embedding utilities live in [backend/embeddings.py](../backend/embeddings.py).

Responsibilities:

- load the processed catalog
- resolve catalog image paths
- generate text embeddings
- generate CLIP image embeddings
- build and save FAISS indexes
- load existing index artifacts lazily at runtime
- warm up retrieval resources in the background

Artifacts are written to:

- `backend/.artifacts/indexes/`

The backend avoids rebuilding indexes on every startup. Instead:

- offline build is handled by [backend/build_indexes.py](../backend/build_indexes.py)
- runtime startup only checks and loads existing artifacts
- model warmup happens in the background after the app starts

## Data Pipeline

The raw fashion data is expected in `dataset/`, based on the Kaggle dataset used by the project.

The preparation step is implemented in [backend/prepare_kaggle_catalog.py](../backend/prepare_kaggle_catalog.py).

Pipeline stages:

1. Read `styles.csv` and image files from `dataset/`
2. Filter to a catalog shape that fits the shopping assistant use case
3. Normalize fields into the project schema
4. Enrich products with inferred metadata
5. Save the processed catalog to [backend/data/catalog/products.json](../backend/data/catalog/products.json)
6. Build text and image indexes from that processed catalog

Current product metadata includes fields such as:

- `brand`
- `category`
- `subcategory`
- `description`
- `visual_description`
- `features`
- `tags`
- `use_cases`
- `price_band`
- `rating`
- `inventory_status`

## Personalization and Memory

Lightweight personalization lives in [backend/personalization.py](../backend/personalization.py).

Current profile state includes:

- preferred categories
- preferred tags
- preferred use cases
- coarse price band
- last query

Profiles are stored in:

- `backend/.artifacts/personalization.db`

This layer is intentionally lightweight. It is designed to make repeated shopping interactions feel more consistent without introducing a separate recommendation service or long-term user profile system.

Short chat memory is kept in-process per session and is mainly used for the conversational assistant layer.

## Startup and Developer Workflow

The main local startup path is [scripts/run.sh](../scripts/run.sh).

The script:

- loads `.env`
- optionally prepares the catalog
- optionally builds or refreshes indexes
- starts the backend
- waits for `/readyz`
- starts the frontend

This flow is mirrored by [Makefile](../Makefile) targets such as:

- `make dev`
- `make build-indexes`
- `make prepare-catalog`
- `make eval`

## Evaluation

Offline evaluation lives under:

- [eval/queries.jsonl](../eval/queries.jsonl)
- [eval/eval_retrieval.py](../eval/eval_retrieval.py)

The benchmark currently measures:

- strict retrieval accuracy against a small set of labeled positive IDs
- semantic category and subcategory consistency
- keyword alignment
- latency by modality

This gives the repository a practical regression harness for retrieval changes and a clearer portfolio story for multimodal search quality.

## Tradeoffs

The current design intentionally favors simplicity and explainability over maximum production complexity.

Current tradeoffs:

- local FAISS indexes instead of a hosted vector database
- heuristic reranking instead of a learned cross-encoder reranker
- lightweight SQLite personalization instead of a larger user-profile system
- small human-authored benchmark instead of a large-scale labeled evaluation suite

Those tradeoffs keep the project compact enough to understand end-to-end while still demonstrating retrieval, multimodal search, orchestration, evaluation, and developer tooling.
