"""Model download script.

Downloads:
  * BGE embedding + reranker models (cached automatically by
    ``sentence-transformers`` on first use — this pre-warms the cache).
  * Qwen2.5-1.5B-Instruct GGUF (Q4_K_M) into ``./models/``.

Uses ``HF_TOKEN`` from the environment for authenticated downloads.

Usage::

    python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings


def download_gguf() -> Path:
    """Download the GGUF LLM model file via huggingface_hub."""
    from huggingface_hub import hf_hub_download

    settings = get_settings()
    settings.ensure_dirs()
    token = settings.hf_token
    if not token:
        print("[download] WARNING: HF_TOKEN not set — public models may still work.")

    print(f"[download] GGUF: {settings.llm_repo_id} / {settings.llm_model_file}")
    path = hf_hub_download(
        repo_id=settings.llm_repo_id,
        filename=settings.llm_model_file,
        local_dir=str(settings.models_dir),
        token=token,
    )
    print(f"[download] GGUF saved → {path}")
    return Path(path)


def prewarm_sentence_transformers() -> None:
    """Pre-download the embedding + reranker models into the HF cache."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    settings = get_settings()
    print(f"[download] Embedder: {settings.embedding_model}")
    SentenceTransformer(settings.embedding_model)
    print(f"[download] Reranker: {settings.reranker_model}")
    CrossEncoder(settings.reranker_model)
    print("[download] sentence-transformers models cached.")


def main() -> int:
    print("[download] Starting model downloads…")
    try:
        prewarm_sentence_transformers()
    except Exception as exc:
        print(f"[download] sentence-transformers prewarm failed: {exc}")
    try:
        download_gguf()
    except Exception as exc:
        print(f"[download] GGUF download failed: {exc}")
        return 1
    print("[download] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
