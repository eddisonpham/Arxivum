# Research Library MCP

A **local-first** Model Context Protocol (MCP) server that lets coding agents
(Cursor, Claude Code, Freebuff, …) discover, summarise, organise, and ideate
over arXiv research papers — entirely on your own machine.

## Features

- **arXiv search** with Semantic Scholar citation / venue enrichment.
- **Vector library** (ChromaDB + BGE embeddings) with hybrid retrieval.
- **Structured summaries** — problem statement, methodology, findings,
  ablations, discussion, limitations.
- **Idea generation** grounded in each paper's constraints & biases.
- **Novelty re-verification** against the local library and arXiv.
- **Visual supervision panel** — a clean web UI to inspect the library and
  every action the agent performed.
- **CPU / small-GPU friendly** — quantized GGUF models via `llama-cpp-python`.

## Quick start

```bash
# 1. Clone & enter
git clone <repo>
cd arxivum

# 2. Create venv & install (add [llm] for local LLM inference)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm]"
# Note: [llm] installs llama-cpp-python (needs CMake + C++ compiler).
# Without [llm], everything works except local LLM generation.

# 3. Configure
cp .env.example .env               # add your HF_TOKEN (for model download only)

# 4. Download models (~1.5 GB)
python scripts/download_models.py

# 5. Initialise database
python scripts/migrate.py
```

## Running

```bash
# MCP server (stdio — for agents like Claude Code / Cursor)
python -m src.mcp_server

# Web API + visual panel
python -m src.api.main
# → http://localhost:8000       (visual panel)
# → http://localhost:8000/demo  (demo page)
# → http://localhost:8000/docs  (OpenAPI)
```

## Testing

```bash
# Unit + component + integration tests (mocked, offline)
pytest

# Smoke tests (gitignored — real local GGUF model, network calls)
# See tests/smoke/ after copying the smoke template.
```

## Architecture

```
Coding Agent ──MCP stdio──▶ MCP Server ──▶ arXiv API + Semantic Scholar
                                │
                    FastAPI + Visual Panel
                                │
                ┌───────────────┴───────────────┐
            ChromaDB                        SQLite
         (vectors)                     (metadata/ideas)
                                │
              llama-cpp-python (Qwen2.5-1.5B GGUF)
              sentence-transformers (BGE embed/rerank)
```

## Scope

This is a **local-only POC**. All models, databases, and services run on the
user's machine. Cloud / HPC scaling is future work.
