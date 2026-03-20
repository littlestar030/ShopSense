from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()

_client: Any | None = None
_catalog: list[dict] | None = None
_text_index: Any | None = None
_image_index: Any | None = None
_clip_model: Any | None = None
_clip_processor: Any | None = None
_faiss: Any | None = None
_torch: Any | None = None
_image_embedding_dim: int | None = None
_device: str | None = None
_warmup_started = False
_warmup_lock = threading.Lock()
_clip_lock = threading.Lock()


def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as faiss_module

        _faiss = faiss_module
    return _faiss


def get_torch():
    global _torch
    if _torch is None:
        import torch as torch_module

        _torch = torch_module
    return _torch


def get_device() -> str:
    global _device
    if _device is None:
        torch = get_torch()
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_openai_client():
    global _client
    if settings.use_mock_openai:
        raise RuntimeError("OpenAI client should not be used in mock mode.")
    if _client is None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when USE_MOCK_OPENAI is false.")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def load_catalog() -> list[dict]:
    global _catalog
    if _catalog is None:
        with settings.catalog_path.open("r", encoding="utf-8") as file:
            _catalog = json.load(file)
    return _catalog


def resolve_catalog_image_path(image_path: str | None) -> str | None:
    """Resolve a catalog image path relative to the repository root."""
    if not image_path:
        return None
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "images":
        return str((settings.images_dir / path.name).resolve())
    return str((settings.catalog_path.parent.parent / path).resolve())


def get_clip_components():
    global _clip_model, _clip_processor, _image_embedding_dim
    if _clip_model is None or _clip_processor is None:
        with _clip_lock:
            if _clip_model is None or _clip_processor is None:
                from transformers import CLIPModel, CLIPProcessor
                from transformers.utils import logging as transformers_logging

                transformers_logging.set_verbosity_error()

                device = get_device()
                logger.info("Loading CLIP model '%s' on %s", settings.clip_model_name, device)
                _clip_model = CLIPModel.from_pretrained(settings.clip_model_name).to(device)
                _clip_model.eval()
                _clip_processor = CLIPProcessor.from_pretrained(settings.clip_model_name, use_fast=True)
                _image_embedding_dim = _clip_model.config.projection_dim
    return _clip_model, _clip_processor


def get_image_embedding_dim() -> int:
    global _image_embedding_dim
    if _image_embedding_dim is None:
        model, _ = get_clip_components()
        _image_embedding_dim = model.config.projection_dim
    return _image_embedding_dim


def _mock_text_embedding(text: str, dim: int = 1536) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return vector.tolist()


def get_text_embedding(text: str) -> list[float]:
    """Get a text embedding for a string using OpenAI or a deterministic mock."""
    if settings.use_mock_openai:
        return _mock_text_embedding(text)
    response = get_openai_client().embeddings.create(
        model=settings.text_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def get_image_embedding(image) -> np.ndarray:
    """Get an image embedding vector for a PIL Image using CLIP."""
    model, processor = get_clip_components()
    torch = get_torch()
    device = get_device()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    vector = features.squeeze(0).detach().cpu().numpy().astype("float32")
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return vector


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index for fast similarity search on embeddings."""
    faiss = get_faiss()
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array, got shape {embeddings.shape}")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def _catalog_hash() -> str:
    return hashlib.sha256(settings.catalog_path.read_bytes()).hexdigest()


def _artifact_paths() -> dict[str, Path]:
    return {
        "manifest": settings.index_dir / "manifest.json",
        "text_embeddings": settings.index_dir / "text_embeddings.npy",
        "image_embeddings": settings.index_dir / "image_embeddings.npy",
        "text_index": settings.index_dir / "text.index",
        "image_index": settings.index_dir / "image.index",
    }


def _catalog_text_repr(product: dict) -> str:
    return (
        f"Brand: {product.get('brand', '')} "
        f"Category: {product['category']} "
        f"Subcategory: {product.get('subcategory', '')} "
        f"Name: {product['name']} "
        f"Description: {product['description']} "
        f"Visual description: {product.get('visual_description', '')} "
        f"Features: {', '.join(product.get('features', []))} "
        f"Tags: {', '.join(product.get('tags', []))} "
        f"Use cases: {', '.join(product.get('use_cases', []))} "
        f"Color: {product.get('color', '')} "
        f"Material: {product.get('material', '')} "
        f"Style: {product.get('style', '')} "
        f"Gender: {product.get('gender', '')} "
        f"Season: {product.get('season', '')} "
        f"Price band: {product.get('price_band', '')}"
    )


def build_and_save_indexes(force: bool = False) -> None:
    global _text_index, _image_index
    settings.ensure_directories()
    catalog = load_catalog()
    artifacts = _artifact_paths()

    if not force and _manifest_is_current():
        logger.info("Index artifacts are current. Skipping rebuild.")
        return

    if not force and artifacts["manifest"].exists():
        logger.info("Index artifacts are stale. Rebuilding.")

    logger.info("Building retrieval indexes for %s catalog products", len(catalog))

    text_embeddings = np.array(
        [get_text_embedding(_catalog_text_repr(product)) for product in catalog],
        dtype="float32",
    )

    image_embeddings = []
    for product in catalog:
        image_path = resolve_catalog_image_path(product.get("image_path") or product.get("image"))
        if image_path and Path(image_path).exists():
            from PIL import Image

            with Image.open(image_path).convert("RGB") as image:
                embedding = get_image_embedding(image)
        else:
            embedding = np.zeros(get_image_embedding_dim(), dtype="float32")
        image_embeddings.append(embedding)
    image_embeddings_np = np.array(image_embeddings, dtype="float32")

    _text_index = build_faiss_index(text_embeddings)
    _image_index = build_faiss_index(image_embeddings_np)

    np.save(artifacts["text_embeddings"], text_embeddings)
    np.save(artifacts["image_embeddings"], image_embeddings_np)
    faiss = get_faiss()
    faiss.write_index(_text_index, str(artifacts["text_index"]))
    faiss.write_index(_image_index, str(artifacts["image_index"]))
    artifacts["manifest"].write_text(
        json.dumps(
            {
                "catalog_hash": _catalog_hash(),
                "catalog_size": len(catalog),
                "text_embedding_model": settings.text_embedding_model,
                "clip_model_name": settings.clip_model_name,
                "device": get_device(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved index artifacts to %s", settings.index_dir)


def _manifest_is_current() -> bool:
    artifacts = _artifact_paths()
    if not all(path.exists() for path in artifacts.values()):
        return False
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    return (
        manifest.get("catalog_hash") == _catalog_hash()
        and manifest.get("catalog_size") == len(load_catalog())
        and manifest.get("text_embedding_model") == settings.text_embedding_model
        and manifest.get("clip_model_name") == settings.clip_model_name
    )


def ensure_indexes_loaded(force_rebuild: bool = False) -> None:
    global _text_index, _image_index

    if force_rebuild:
        build_and_save_indexes(force=True)
        return

    if _text_index is not None and _image_index is not None:
        return

    artifacts = _artifact_paths()
    manifest_is_current = _manifest_is_current()

    if manifest_is_current:
        faiss = get_faiss()
        _text_index = faiss.read_index(str(artifacts["text_index"]))
        _image_index = faiss.read_index(str(artifacts["image_index"]))
        return

    if settings.auto_build_indexes:
        build_and_save_indexes(force=True)
        return

    raise RuntimeError(
        "Retrieval indexes are missing or stale. Run `python -m backend.build_indexes` first."
    )


def warmup_retrieval_resources() -> None:
    """Warm up retrieval resources in the background without blocking app startup."""
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    logger.info("Background warmup started")
    try:
        ensure_indexes_loaded(force_rebuild=False)
        if settings.warmup_clip_model:
            get_clip_components()
        logger.info("Background warmup completed")
    except Exception:
        logger.exception("Background warmup failed")


def find_similar_products_by_text(query_emb: list[float], top_k: int | None = None) -> list[dict]:
    """Find top-k products whose text embeddings are closest to the query."""
    ensure_indexes_loaded(force_rebuild=settings.force_rebuild_indexes)
    top_k = top_k or settings.text_top_k
    _, indices = _text_index.search(np.array([query_emb], dtype="float32"), top_k)
    catalog = load_catalog()
    return [catalog[i] for i in indices[0]]


def find_similar_products_by_image(query_emb: np.ndarray, top_k: int | None = None) -> list[dict]:
    """Find top-k products whose image embeddings are closest to the query."""
    ensure_indexes_loaded(force_rebuild=settings.force_rebuild_indexes)
    top_k = top_k or settings.image_top_k
    _, indices = _image_index.search(np.array([query_emb], dtype="float32"), top_k)
    catalog = load_catalog()
    return [catalog[i] for i in indices[0]]
