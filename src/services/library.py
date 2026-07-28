"""Core library service.

Orchestrates:
  * arXiv search + import,
  * Semantic Scholar enrichment (async, batched),
  * SQLite + ChromaDB persistence,
  * hybrid vector + metadata retrieval with reranking,
  * paper removal.

This is the single place where all the low-level clients, the database,
and the embedding model are wired together.  Both the MCP server and the
FastAPI layer call into this service.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.clients.arxiv_client import ArxivClient, ArxivPaper, strip_version
from src.clients.s2_client import S2Metrics, SemanticScholarClient
from src.db.chroma_store import CHUNK_ABSTRACT, CHUNK_TITLE, Chunk, ChromaStore
from src.db.models import ActivityRow, Database, MetricsRow, PaperRow
from src.inference.manager import ModelManager
from src.utils import run_async

logger = logging.getLogger(__name__)


@dataclass
class QueryResultItem:
    """A single library query result."""

    arxiv_id: str
    title: str
    score: float
    abstract_snippet: str
    citation_count: int = -1
    venue: str | None = None
    primary_category: str | None = None
    authors: list[str] = field(default_factory=list)
    chunk_type: str = ""

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "score": round(self.score, 4),
            "abstract_snippet": self.abstract_snippet,
            "citation_count": self.citation_count,
            "venue": self.venue,
            "primary_category": self.primary_category,
            "authors": self.authors,
            "chunk_type": self.chunk_type,
        }


@dataclass
class SearchResultItem:
    """A single arXiv search result after import."""

    arxiv_id: str
    title: str
    citation_count: int = -1
    venue: str | None = None
    imported: bool = True

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "citation_count": self.citation_count,
            "venue": self.venue,
            "imported": self.imported,
        }


class LibraryService:
    """High-level research library operations."""

    def __init__(
        self,
        db: Database,
        chroma: ChromaStore,
        arxiv_client: ArxivClient,
        s2_client: SemanticScholarClient,
        models: ModelManager,
    ) -> None:
        self.db = db
        self.chroma = chroma
        self.arxiv = arxiv_client
        self.s2 = s2_client
        self.models = models

    # ── search + import ────────────────────────────────────────────────
    def search_and_import(
        self,
        query: str,
        max_results: int = 10,
        primary_category: str | None = None,
        sort_by: str = "relevance",
        auto_enrich: bool = False,
    ) -> list[SearchResultItem]:
        """Search arXiv, import all results into the library.

        If ``auto_enrich`` is True, Semantic Scholar metrics are fetched
        synchronously (blocking).  When False, papers are stored with
        sentinel metrics and enrichment can be triggered separately via
        :meth:`enrich_paper`.
        """
        log_id = self.db.log_activity(ActivityRow(
            id=None, action_type="search", query=query, status="started",
            metadata_json={"max_results": max_results, "category": primary_category},
        ))
        try:
            papers = self.arxiv.search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                primary_category=primary_category,
            )
            results: list[SearchResultItem] = []
            for p in papers:
                self._import_paper(p, enrich=auto_enrich)
                metrics = self.db.get_metrics(p.arxiv_id)
                results.append(SearchResultItem(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    citation_count=metrics.citation_count if metrics else -1,
                    venue=metrics.venue if metrics else None,
                    imported=True,
                ))
            self.db.update_activity_status(log_id, "completed")
            return results
        except Exception:
            self.db.update_activity_status(log_id, "failed")
            raise

    def _import_paper(self, paper: ArxivPaper, enrich: bool = False) -> None:
        """Persist a paper to SQLite + ChromaDB and optionally enrich."""
        # 1. SQLite
        row = PaperRow(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            published=paper.published,
            updated=paper.updated,
            categories=paper.categories,
            primary_category=paper.primary_category,
            pdf_url=paper.pdf_url,
            abs_url=paper.abs_url,
            doi=paper.doi,
            journal_ref=paper.journal_ref,
            comment=paper.comment,
        )
        self.db.upsert_paper(row)
        self.db.log_activity(ActivityRow(
            id=None, action_type="import", arxiv_id=paper.arxiv_id,
            status="completed",
        ))

        # 2. Enrich via S2 (sync wrapper if requested)
        if enrich:
            metrics = run_async(self._enrich_async(paper.arxiv_id))
        else:
            # Write a placeholder metrics row so queries don't break.
            self.db.upsert_metrics(MetricsRow(arxiv_id=paper.arxiv_id))

        # 3. ChromaDB — embed abstract + title
        self._index_paper(paper)

    def _index_paper(self, paper: ArxivPaper) -> None:
        """Embed and store the paper's abstract and title chunks."""
        embedder = self.models.embedder
        metrics = self.db.get_metrics(paper.arxiv_id)
        meta_citation = metrics.citation_count if metrics else -1
        meta_venue = metrics.venue if metrics else None

        texts = [paper.abstract, paper.title]
        vectors = embedder.embed(texts, is_query=False)

        chunks = [
            Chunk(
                id=ChromaStore.chunk_id(paper.arxiv_id, CHUNK_ABSTRACT),
                document=paper.abstract,
                arxiv_id=paper.arxiv_id,
                chunk_type=CHUNK_ABSTRACT,
                title=paper.title,
                venue=meta_venue,
                citation_count=meta_citation,
                primary_category=paper.primary_category,
                embedding=vectors[0],
            ),
            Chunk(
                id=ChromaStore.chunk_id(paper.arxiv_id, CHUNK_TITLE),
                document=paper.title,
                arxiv_id=paper.arxiv_id,
                chunk_type=CHUNK_TITLE,
                title=paper.title,
                venue=meta_venue,
                citation_count=meta_citation,
                primary_category=paper.primary_category,
                embedding=vectors[1],
            ),
        ]
        self.chroma.upsert_chunks(chunks)

    def index_summary_section(
        self, arxiv_id: str, section: str, content: str
    ) -> None:
        """Embed and store a generated summary section as a chunk."""
        paper = self.db.get_paper(arxiv_id)
        if not paper:
            raise ValueError(f"Paper {arxiv_id} not in library")
        embedder = self.models.embedder
        vec = embedder.embed_one(content, is_query=False)
        metrics = self.db.get_metrics(arxiv_id)
        chunk = Chunk(
            id=ChromaStore.chunk_id(arxiv_id, "summary", section),
            document=content,
            arxiv_id=arxiv_id,
            chunk_type=f"summary_{section}",
            title=paper.title,
            venue=metrics.venue if metrics else None,
            citation_count=metrics.citation_count if metrics else -1,
            primary_category=paper.primary_category,
            section=section,
            embedding=vec,
        )
        self.chroma.upsert_chunk(chunk)

    # ── S2 enrichment ──────────────────────────────────────────────────
    async def _enrich_async(self, arxiv_id: str) -> S2Metrics:
        """Fetch S2 metrics and persist them."""
        try:
            metrics = await self.s2.fetch_paper(strip_version(arxiv_id))
        except Exception as exc:
            logger.warning("S2 enrichment failed for %s: %s", arxiv_id, exc)
            metrics = S2Metrics(arxiv_id=arxiv_id)
        self._store_metrics(arxiv_id, metrics)
        # Re-index so Chroma metadata has the real citation_count/venue.
        paper = self.db.get_paper(arxiv_id)
        if paper:
            arxiv_paper = ArxivPaper(
                arxiv_id=paper.arxiv_id, title=paper.title,
                authors=paper.authors, abstract=paper.abstract,
                primary_category=paper.primary_category,
            )
            self._index_paper(arxiv_paper)
        return metrics

    async def enrich_paper(self, arxiv_id: str) -> S2Metrics:
        """Public async enrichment entry point."""
        log_id = self.db.log_activity(ActivityRow(
            id=None, action_type="enrich", arxiv_id=arxiv_id, status="started",
        ))
        try:
            metrics = await self._enrich_async(arxiv_id)
            self.db.update_activity_status(log_id, "completed")
            return metrics
        except Exception:
            self.db.update_activity_status(log_id, "failed")
            raise

    def enrich_paper_sync(self, arxiv_id: str) -> S2Metrics:
        """Sync wrapper for :meth:`enrich_paper`.

        Safe to call from both synchronous (MCP stdio) and asynchronous
        (FastAPI) contexts — see :func:`src.utils.run_async`.
        """
        return run_async(self.enrich_paper(arxiv_id))

    def _store_metrics(self, arxiv_id: str, m: S2Metrics) -> None:
        self.db.upsert_metrics(MetricsRow(
            arxiv_id=arxiv_id,
            citation_count=m.citation_count,
            influential_citation_count=m.influential_citation_count,
            venue=m.venue,
            s2_paper_id=m.s2_paper_id,
            raw_s2_json=json.dumps(m.raw) if m.raw else None,
        ))

    # ── hybrid retrieval ───────────────────────────────────────────────
    def query_library(
        self,
        query: str,
        top_k: int = 5,
        min_citations: int | None = None,
        venue: str | None = None,
        primary_category: str | None = None,
        rerank: bool = True,
    ) -> list[QueryResultItem]:
        """Hybrid vector + metadata search over the local library.

        Steps:
          1. Embed the query (with BGE query prefix).
          2. ChromaDB vector search with metadata pre-filter.
          3. Deduplicate by arxiv_id (keep best-scoring chunk).
          4. Optionally rerank with the cross-encoder.
          5. Return top-k.
        """
        log_id = self.db.log_activity(ActivityRow(
            id=None, action_type="query", query=query, status="started",
            metadata_json={"top_k": top_k, "min_citations": min_citations,
                           "venue": venue, "rerank": rerank},
        ))
        try:
            embedder = self.models.embedder
            q_vec = embedder.embed_one(query, is_query=True)

            # Build Chroma metadata filter.
            where: dict[str, Any] = {}
            if primary_category:
                where["primary_category"] = primary_category

            # Over-fetch so we have enough after dedup.
            fetch_n = min(top_k * 4, 50)
            raw_results = self.chroma.query(
                query_embedding=q_vec, top_k=fetch_n, where=where or None,
            )

            # Deduplicate by arxiv_id, keeping the highest-scoring chunk.
            best_by_paper: dict[str, Any] = {}
            for r in raw_results:
                # Apply post-filters that Chroma can't do natively.
                if min_citations is not None and r.citation_count < min_citations:
                    continue
                if venue and (r.venue is None or venue.lower() not in r.venue.lower()):
                    continue
                existing = best_by_paper.get(r.arxiv_id)
                if existing is None or r.score > existing.score:
                    best_by_paper[r.arxiv_id] = r

            candidates = list(best_by_paper.values())
            if not candidates:
                self.db.update_activity_status(log_id, "completed")
                return []

            # Rerank with cross-encoder (capped at 10 by Reranker).
            if rerank and len(candidates) > 1:
                reranker = self.models.reranker
                texts = [c.document for c in candidates]
                ranked_idx = reranker.rerank(query, texts, top_k=top_k)
                # Map back; cross-encoder scores are raw logits.
                reranked = []
                for idx, _score in ranked_idx:
                    c = candidates[idx]
                    reranked.append(c)
                candidates = reranked

            # Trim to top_k and enrich with DB metadata.
            results: list[QueryResultItem] = []
            for c in candidates[:top_k]:
                paper = self.db.get_paper(c.arxiv_id)
                metrics = self.db.get_metrics(c.arxiv_id)
                snippet = c.document[:300]
                results.append(QueryResultItem(
                    arxiv_id=c.arxiv_id,
                    title=c.title or (paper.title if paper else ""),
                    score=c.score,
                    abstract_snippet=snippet,
                    citation_count=metrics.citation_count if metrics else -1,
                    venue=metrics.venue if metrics else None,
                    primary_category=paper.primary_category if paper else None,
                    authors=paper.authors if paper else [],
                    chunk_type=c.chunk_type,
                ))

            self.db.update_activity_status(log_id, "completed")
            return results
        except Exception:
            self.db.update_activity_status(log_id, "failed")
            raise

    # ── paper removal ──────────────────────────────────────────────────
    def remove_paper(self, arxiv_id: str) -> bool:
        """Remove a paper and all derived data from the library."""
        log_id = self.db.log_activity(ActivityRow(
            id=None, action_type="remove", arxiv_id=arxiv_id, status="started",
        ))
        try:
            self.chroma.delete_by_arxiv_id(arxiv_id)
            deleted = self.db.delete_paper(arxiv_id)
            self.db.update_activity_status(
                log_id, "completed" if deleted else "completed"
            )
            return deleted
        except Exception:
            self.db.update_activity_status(log_id, "failed")
            raise

    # ── details ────────────────────────────────────────────────────────
    def get_paper_detail(self, arxiv_id: str) -> dict | None:
        """Return full metadata + metrics + summaries for a paper."""
        paper = self.db.get_paper(arxiv_id)
        if not paper:
            return None
        metrics = self.db.get_metrics(arxiv_id)
        summaries = self.db.get_summaries(arxiv_id)
        ideas = self.db.list_ideas(arxiv_id)
        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "published": paper.published,
            "updated": paper.updated,
            "categories": paper.categories,
            "primary_category": paper.primary_category,
            "pdf_url": paper.pdf_url,
            "abs_url": paper.abs_url,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comment": paper.comment,
            "metrics": metrics.__dict__ if metrics else None,
            "summaries": [
                {"section": s.section, "content": s.content, "model_used": s.model_used}
                for s in summaries
            ],
            "ideas": [
                {"id": i.id, "idea_text": i.idea_text, "status": i.status,
                 "search_queries": i.search_queries}
                for i in ideas
            ],
        }

    # ── listing ────────────────────────────────────────────────────────
    def list_library(
        self,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        primary_category: str | None = None,
        min_citations: int | None = None,
        venue: str | None = None,
    ) -> dict:
        """List papers with pagination and filters."""
        papers = self.db.list_papers(
            limit=limit, offset=offset, sort_by=sort_by,
            primary_category=primary_category,
            min_citations=min_citations, venue=venue,
        )
        total = self.db.count_papers()
        return {
            "papers": [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "authors": p.authors,
                    "primary_category": p.primary_category,
                    "published": p.published,
                    "created_at": p.created_at,
                }
                for p in papers
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
