"""Shared pytest fixtures.

All tests use in-memory SQLite, ephemeral ChromaDB, a stub embedder/reranker
(no model downloads), and a stub LLM.  No network calls, no heavy ML deps.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

# ── Stub heavy ML modules before any src imports ─────────────────────────────
# llama_cpp requires a C++ build; stub it so tests run without it.
if "llama_cpp" not in sys.modules:
    llama_stub = types.ModuleType("llama_cpp")

    class _StubLlama:
        def __init__(self, **kw): pass
        def create_chat_completion(self, **kw):
            return {"choices": [{"message": {"content": '{"overall": "stub"}'}}]}

    llama_stub.Llama = _StubLlama
    sys.modules["llama_cpp"] = llama_stub


# ── Stub embedder / reranker (no model download) ─────────────────────────────

class StubEmbedder:
    """Deterministic embedder that returns fixed-size random-ish vectors."""

    def __init__(self, model_name: str = "stub") -> None:
        self._model_name = model_name
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        self._loaded = True

    def embed(self, texts: Sequence[str], is_query: bool = False, normalize: bool = True) -> list[list[float]]:
        self._loaded = True
        # Deterministic pseudo-embedding based on text hash → 384-d.
        import hashlib
        results = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [(b / 255.0 - 0.5) * 2 for b in h * 24]  # 768 → take 384
            results.append(vec[:384])
        return results

    def embed_one(self, text: str, is_query: bool = False, normalize: bool = True) -> list[float]:
        return self.embed([text], is_query, normalize)[0]

    @property
    def dim(self) -> int:
        self._loaded = True
        return 384

    def unload(self) -> None:
        self._loaded = False


class StubReranker:
    """Deterministic reranker that preserves input order."""

    def __init__(self, model_name: str = "stub") -> None:
        self._model_name = model_name
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        self._loaded = True

    def rerank(self, query: str, candidates: Sequence[str], top_k: int | None = None) -> list[tuple[int, float]]:
        self._loaded = True
        capped = list(candidates[:10])
        ranked = [(i, 1.0 - i * 0.1) for i in range(len(capped))]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def unload(self) -> None:
        self._loaded = False


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def stub_embedder():
    return StubEmbedder()


@pytest.fixture
def stub_reranker():
    return StubReranker()


@pytest.fixture
def stub_llm():
    """StubLLM that returns context-appropriate canned JSON responses.

    Detects the prompt type (summary, constraints, ideas, novelty) from the
    user message content and returns the matching JSON shape.
    """
    from src.inference.llm import StubLLM

    summary_json = json.dumps({
        "problem_statement": "The paper addresses X.",
        "methodology": "Uses method Y.",
        "findings": "Achieves result Z.",
        "ablations": "Tested variants A and B.",
        "discussion": "Implications for field W.",
        "limitations": "Limited to dataset D.",
        "overall": "A paper about X using Y achieving Z.",
    })
    constraints_json = json.dumps({
        "assumptions": ["data is iid"],
        "inductive_biases": ["linear separability"],
        "limitations": ["small dataset"],
        "domain": "NLP",
        "key_method": "transformer",
    })
    ideas_json = json.dumps([
        {
            "title": "Idea 1",
            "summary": "A novel approach using method A.",
            "extension": "Extends the paper by applying A.",
            "next_steps": ["step1", "step2"],
            "search_queries": ["method A novel", "approach A new"],
        },
        {
            "title": "Idea 2",
            "summary": "An alternative using method B.",
            "extension": "Contradicts by using B instead.",
            "next_steps": ["step3"],
            "search_queries": ["method B alternative"],
        },
        {
            "title": "Idea 3",
            "summary": "A hybrid approach combining A and B.",
            "extension": "Builds on both methods.",
            "next_steps": ["step4", "step5"],
            "search_queries": ["hybrid A B"],
        },
    ])
    novelty_json = json.dumps({
        "verdict": "likely_novel",
        "reason": "No similar work found.",
    })

    def responder(msgs):
        content = msgs[-1]["content"] if msgs else ""
        if "JSON array" in content and "ideas" in content:
            return ideas_json
        if "assumptions" in content and "inductive_biases" in content:
            return constraints_json
        if "verdict" in content and ("likely_novel" in content or "similar_exists" in content):
            return novelty_json
        # Default: summary format
        return summary_json

    return StubLLM(responder=responder, default=summary_json)


@pytest.fixture
def db():
    """In-memory SQLite database."""
    from src.db.models import Database
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture
def chroma_store():
    """Ephemeral ChromaDB store (in-memory)."""
    from src.db.chroma_store import ChromaStore
    store = ChromaStore(path="")  # ephemeral
    yield store
    store.reset()


@pytest.fixture
def mock_arxiv_results():
    """Sample arXiv Result-like objects for mocking."""
    from src.clients.arxiv_client import ArxivPaper
    return [
        ArxivPaper(
            arxiv_id="2106.00001",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer.",
            published="2017-06-12T00:00:00",
            updated="2017-06-12T00:00:00",
            categories=["cs.CL"],
            primary_category="cs.CL",
            pdf_url="http://arxiv.org/pdf/2106.00001",
            abs_url="http://arxiv.org/abs/2106.00001",
        ),
        ArxivPaper(
            arxiv_id="2106.00002",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Jacob Devlin", "Ming-Wei Chang"],
            abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            published="2018-10-11T00:00:00",
            updated="2019-05-29T00:00:00",
            categories=["cs.CL"],
            primary_category="cs.CL",
            pdf_url="http://arxiv.org/pdf/2106.00002",
            abs_url="http://arxiv.org/abs/2106.00002",
        ),
    ]


@pytest.fixture
def mock_arxiv_client(mock_arxiv_results):
    """Mocked ArxivClient that returns sample papers."""
    from src.clients.arxiv_client import ArxivClient
    client = MagicMock(spec=ArxivClient)
    client.search.return_value = mock_arxiv_results
    client.get_paper.return_value = mock_arxiv_results[0]
    return client


@pytest.fixture
def mock_s2_client():
    """Mocked SemanticScholarClient."""
    from src.clients.s2_client import S2Metrics, SemanticScholarClient
    client = MagicMock(spec=SemanticScholarClient)

    async def _fetch_paper(arxiv_id):
        return S2Metrics(
            arxiv_id=arxiv_id,
            citation_count=1000,
            influential_citation_count=50,
            venue="NeurIPS",
            s2_paper_id="abc123",
            raw={"paperId": "abc123", "citationCount": 1000},
        )

    async def _fetch_batch(ids):
        return {aid: S2Metrics(arxiv_id=aid, citation_count=500, venue="ICML") for aid in ids}

    async def _aclose():
        pass

    client.fetch_paper = _fetch_paper
    client.fetch_batch = _fetch_batch
    client.aclose = _aclose
    return client


@pytest.fixture
def model_manager(stub_embedder, stub_reranker, stub_llm):
    """ModelManager with stub models injected."""
    from src.inference.manager import ModelManager
    mgr = ModelManager(constrained_memory=False)
    # Inject stubs directly
    mgr._embedder = stub_embedder
    mgr._reranker = stub_reranker
    mgr.set_llm(stub_llm)
    yield mgr
    mgr.shutdown()


@pytest.fixture
def app_context(db, chroma_store, mock_arxiv_client, mock_s2_client, model_manager):
    """Fully-wired AppContext with all stubs/mocks."""
    from src.app import AppContext
    from src.services.ideas import IdeaService
    from src.services.library import LibraryService
    from src.services.novelty import NoveltyService
    from src.services.summarizer import SummarizerService

    library = LibraryService(db, chroma_store, mock_arxiv_client, mock_s2_client, model_manager)
    summarizer = SummarizerService(db, model_manager, library)
    ideas = IdeaService(db, model_manager)
    novelty = NoveltyService(db, model_manager, mock_arxiv_client,
                             library_query_fn=library.query_library)

    ctx = AppContext(
        db=db, chroma=chroma_store, arxiv_client=mock_arxiv_client,
        s2_client=mock_s2_client, models=model_manager,
        library=library, summarizer=summarizer, ideas=ideas, novelty=novelty,
    )
    yield ctx


@pytest.fixture
def imported_paper(app_context):
    """Import a single paper into the library for tests."""
    ctx = app_context
    results = ctx.library.search_and_import("attention", max_results=1)
    return results[0] if results else None
