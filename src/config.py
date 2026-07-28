"""Centralised configuration.

All settings are read from environment variables (with a `.env` file loaded via
``python-dotenv``).  The :class:`Settings` object is a lazily-instantiated
singleton accessed through :func:`get_settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "./models")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    llm_repo_id: str = field(
        default_factory=lambda: os.getenv("LLM_REPO_ID", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    )
    llm_model_file: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    )
    llm_n_ctx: int = field(default_factory=lambda: int(os.getenv("LLM_N_CTX", "4096")))
    llm_n_threads: int = field(default_factory=lambda: int(os.getenv("LLM_N_THREADS", "4")))
    llm_n_gpu_layers: int = field(default_factory=lambda: int(os.getenv("LLM_N_GPU_LAYERS", "0")))

    mcp_transport: str = field(default_factory=lambda: os.getenv("MCP_TRANSPORT", "stdio"))

    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN") or None)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "research_library.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def llm_model_path(self) -> Path:
        return self.models_dir / self.llm_model_file

    def ensure_dirs(self) -> None:
        """Create the data and model directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton."""
    return Settings()
