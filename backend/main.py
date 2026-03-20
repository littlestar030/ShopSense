from __future__ import annotations

import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import (
    CONVERSATION_MEMORY,
    get_agent_response,
    process_catalog_image_query,
    process_image_query,
    process_multimodal_query,
)
from .config import get_settings
from .embeddings import ensure_indexes_loaded, warmup_retrieval_resources


settings = get_settings()
logger = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
if settings.hf_token:
    os.environ.setdefault("HF_TOKEN", settings.hf_token)


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    third_party_level = getattr(logging, settings.third_party_log_level.upper(), logging.WARNING)
    for logger_name in [
        "httpx",
        "httpcore",
        "openai",
        "huggingface_hub",
        "transformers",
        "transformers.modeling_utils",
        "transformers.tokenization_utils_base",
        "faiss",
    ]:
        logging.getLogger(logger_name).setLevel(third_party_level)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    if settings.enable_background_warmup:
        threading.Thread(
            target=warmup_retrieval_resources,
            name="retrieval-warmup",
            daemon=True,
        ).start()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/images", StaticFiles(directory=str(settings.images_dir)), name="images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.normalized_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class SimilarProductRequest(BaseModel):
    product_id: str


def get_user_id(request: Request) -> str:
    session_id = request.headers.get(settings.session_header_key)
    if session_id:
        return session_id
    return request.client.host or "default"


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": settings.app_name}


@app.get("/readyz")
async def readyz():
    try:
        ensure_indexes_loaded(force_rebuild=False)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "index_dir": str(settings.index_dir)}


@app.post("/chat")
async def chat(request_data: ChatRequest, request: Request):
    user_id = get_user_id(request)
    result = get_agent_response(request_data.message, user_id)
    return result


@app.post("/search-image")
async def search_image(file: UploadFile, request: Request):
    user_id = get_user_id(request)
    content = await file.read()
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename or "upload.png")
    with open(file_path, "wb") as temp_file:
        temp_file.write(content)
    result = process_image_query(file_path, user_id)
    try:
        os.remove(file_path)
        os.rmdir(temp_dir)
    except OSError:
        logger.warning("Could not fully clean up temporary image upload directory: %s", temp_dir)
    return result


@app.post("/search-similar-product")
async def search_similar_product(request_data: SimilarProductRequest, request: Request):
    user_id = get_user_id(request)
    return process_catalog_image_query(request_data.product_id, user_id)


@app.post("/chat-multimodal")
async def chat_multimodal(
    request: Request,
    message: str = Form(default=""),
    file: UploadFile | None = File(default=None),
):
    user_id = get_user_id(request)
    if file is None:
        return get_agent_response(message, user_id)

    content = await file.read()
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename or "upload.png")
    with open(file_path, "wb") as temp_file:
        temp_file.write(content)

    try:
        return process_multimodal_query(message, file_path, user_id)
    finally:
        try:
            os.remove(file_path)
            os.rmdir(temp_dir)
        except OSError:
            logger.warning("Could not fully clean up temporary multimodal upload directory: %s", temp_dir)


@app.post("/reset-memory")
async def reset_memory(request: Request):
    user_id = get_user_id(request)
    CONVERSATION_MEMORY[user_id] = []
    return {"type": "chat", "response": f"Memory for {user_id} cleared.", "products": []}
