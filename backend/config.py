from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ShopSense"
    environment: str = "development"
    log_level: str = "INFO"
    third_party_log_level: str = "WARNING"

    openai_api_key: str | None = None
    use_mock_openai: bool = False
    hf_token: str | None = None
    chat_model: str = "gpt-4o-mini"
    text_embedding_model: str = "text-embedding-3-small"
    clip_model_name: str = "openai/clip-vit-base-patch32"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_dev_url: str = "http://localhost:3000"
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"

    catalog_path: Path = REPO_ROOT / "backend" / "data" / "catalog" / "products.json"
    raw_dataset_dir: Path = REPO_ROOT / "dataset"
    images_dir: Path = REPO_ROOT / "dataset" / "images"
    index_dir: Path = REPO_ROOT / "backend" / ".artifacts" / "indexes"
    profile_db_path: Path = REPO_ROOT / "backend" / ".artifacts" / "personalization.db"

    auto_build_indexes: bool = True
    force_rebuild_indexes: bool = False
    enable_background_warmup: bool = True
    warmup_clip_model: bool = True
    enable_personalization: bool = True
    enable_reranking: bool = True
    enable_llm_reranking: bool = False
    llm_rerank_model: str = "gpt-4o-mini"
    llm_rerank_top_n: int = 8
    llm_rerank_modes: str = "multimodal"
    llm_rerank_margin_threshold: float = 0.08
    rerank_candidate_pool_size: int = 12
    text_top_k: int = 3
    image_top_k: int = 3
    memory_turn_limit: int = 10
    session_header_name: str = "x-session-id"

    @field_validator(
        "openai_api_key",
        "hf_token",
        "chat_model",
        "text_embedding_model",
        "clip_model_name",
        "llm_rerank_model",
        "llm_rerank_modes",
        "backend_host",
        "frontend_dev_url",
        "allowed_origins",
        "session_header_name",
        mode="before",
    )
    @classmethod
    def strip_string_values(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator(
        "use_mock_openai",
        "auto_build_indexes",
        "force_rebuild_indexes",
        "enable_background_warmup",
        "warmup_clip_model",
        "enable_personalization",
        "enable_reranking",
        "enable_llm_reranking",
        mode="before",
    )
    @classmethod
    def parse_bool_values(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return value

    @property
    def normalized_allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @property
    def session_header_key(self) -> str:
        return self.session_header_name.lower()

    @property
    def normalized_llm_rerank_modes(self) -> set[str]:
        return {item.strip().lower() for item in self.llm_rerank_modes.split(",") if item.strip()}

    def ensure_directories(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.profile_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
