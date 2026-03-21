PYTHON ?= python
NPM ?= npm
KAGGLE_MAX_PRODUCTS ?= 1200
KAGGLE_MAX_PER_ARTICLE ?= 60
KAGGLE_MIN_ARTICLE_COUNT ?= 20
KAGGLE_SAMPLE_SEED ?= 42

.PHONY: install install-backend install-frontend prepare-catalog prepare-catalog-small prepare-catalog-large build-indexes rebuild-indexes test eval eval-debug backend backend-reload frontend dev dev-reload dev-bootstrap docker-build docker-build-nobuildkit docker-up docker-down run

install: install-backend install-frontend

install-backend:
	$(PYTHON) -m pip install -r requirements.txt

install-frontend:
	cd frontend && $(NPM) install

prepare-catalog:
	$(PYTHON) -m backend.prepare_kaggle_catalog --max-products $(KAGGLE_MAX_PRODUCTS) --max-per-article $(KAGGLE_MAX_PER_ARTICLE) --min-article-count $(KAGGLE_MIN_ARTICLE_COUNT) --seed $(KAGGLE_SAMPLE_SEED)

prepare-catalog-small:
	$(PYTHON) -m backend.prepare_kaggle_catalog --max-products 400 --max-per-article 25 --min-article-count 20 --seed $(KAGGLE_SAMPLE_SEED)

prepare-catalog-large:
	$(PYTHON) -m backend.prepare_kaggle_catalog --max-products 2500 --max-per-article 100 --min-article-count 20 --seed $(KAGGLE_SAMPLE_SEED)

build-indexes:
	$(PYTHON) -m backend.build_indexes

rebuild-indexes:
	$(PYTHON) -m backend.build_indexes --force

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -v

eval:
	PYTHONPATH=. $(PYTHON) eval/eval_retrieval.py

eval-debug:
	PYTHONPATH=. $(PYTHON) eval/eval_retrieval.py --debug-query text_running_shoes --debug-query multimodal_running_from_formal --debug-query multimodal_basketball_from_sandal --debug-output eval/results/debug.json

backend:
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

backend-reload:
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && $(NPM) run dev

dev:
	bash ./scripts/run.sh

dev-bootstrap:
	PREPARE_CATALOG_ON_RUN=true bash ./scripts/run.sh

dev-reload:
	BACKEND_RELOAD=true bash ./scripts/run.sh

run: dev

docker-build:
	docker compose build

docker-build-nobuildkit:
	DOCKER_BUILDKIT=0 docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down
