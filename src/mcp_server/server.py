"""MCP server — exposes research tools to coding agents.

Uses the official ``mcp`` Python SDK's high-level ``MCPServer`` API.
All tools are prefixed with ``research_`` to avoid collisions.

Run via::

    python -m src.mcp_server          # stdio (default)
    MCP_TRANSPORT=sse python -m src.mcp_server   # SSE
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from src.app import AppContext, create_app, shutdown_app
from src.config import get_settings

logger = logging.getLogger(__name__)

TOOL_PREFIX = "research_"

def create_mcp_server(ctx: AppContext):
    """Create an MCP server with all research tools registered.

    Returns the ``MCPServer`` instance.  The caller runs it via
    ``mcp.run(transport=...)``.
    """
    from mcp.server import MCPServer

    mcp = MCPServer("research-library")
    lib = ctx.library
    summarizer = ctx.summarizer
    ideas_svc = ctx.ideas
    novelty_svc = ctx.novelty

    @mcp.tool()
    def research_search_papers(
        query: str,
        max_results: int = 10,
        primary_category: str = "",
        auto_enrich: bool = False,
        summarize: bool = False,
    ) -> str:
        """Search arXiv for papers matching *query* and import them into the
        local library.  Returns a JSON list of imported papers with their
        arXiv IDs, titles, and citation counts (if enriched).

        Args:
            query: arXiv search query (natural language or field-specific).
            max_results: Maximum papers to return (1–50, default 10).
            primary_category: Optional arXiv category filter (e.g. cs.LG).
            auto_enrich: If True, fetch Semantic Scholar metrics (slower).
            summarize: If True, generate summaries after import (much slower).
        """
        max_results = min(max(max_results, 1), 50)
        results = lib.search_and_import(
            query=query,
            max_results=max_results,
            primary_category=primary_category or None,
            auto_enrich=auto_enrich,
        )
        out = {
            "papers": [r.to_dict() for r in results],
            "total_found": len(results),
            "message": f"Imported {len(results)} paper(s) into the library.",
        }
        if summarize and results:
            for r in results[:5]:
                try:
                    summarizer.summarize(r.arxiv_id)
                except Exception as exc:
                    logger.warning("Summary failed for %s: %s", r.arxiv_id, exc)
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def research_query_library(
        query: str,
        top_k: int = 5,
        min_citations: int = 0,
        venue: str = "",
        primary_category: str = "",
        rerank: bool = True,
    ) -> str:
        """Search the *local* library for papers relevant to *query* using
        hybrid vector + metadata retrieval.  Returns JSON with scored
        results including abstract snippets and citation counts.

        Args:
            query: Natural-language query.
            top_k: Number of results (default 5).
            min_citations: Filter out papers with fewer citations (0 = no filter).
            venue: Partial venue/conference name filter (e.g. "NeurIPS").
            primary_category: arXiv category filter (e.g. cs.LG).
            rerank: Apply cross-encoder reranking (default True).
        """
        results = lib.query_library(
            query=query,
            top_k=top_k,
            min_citations=min_citations or None,
            venue=venue or None,
            primary_category=primary_category or None,
            rerank=rerank,
        )
        return json.dumps(
            {"results": [r.to_dict() for r in results]},
            ensure_ascii=False,
        )

    extractor_svc = ctx.extractor

    @mcp.tool()
    def research_extract_paper(arxiv_id: str, force: bool = False) -> str:
        """Extract a structured bibliographic schema for a paper.

        Returns a JSON object with method, datasets, baselines, headline
        metric, contribution, limitations, domain, and bibcode. Schema
        follows OpenAlex / SPECTER conventions so it round-trips into
        BibTeX and LaTeX pipelines. Cached on first call.

        Args:
            arxiv_id: arXiv ID of the paper.
            force: Regenerate even if cached.
        """
        try:
            schema = extractor_svc.extract(arxiv_id, force=force)
            return json.dumps(
                {"arxiv_id": arxiv_id, "extraction": schema},
                ensure_ascii=False,
            )
        except ValueError as exc:
            return json.dumps(
                {"error": "not_found", "message": str(exc)},
                ensure_ascii=False,
            )

    @mcp.tool()
    def research_get_paper_details(arxiv_id: str) -> str:
        """Get full metadata, citation metrics, summaries, and ideas for a
        paper in the local library.

        Args:
            arxiv_id: Normalized arXiv ID (e.g. 2106.00001).
        """
        detail = lib.get_paper_detail(arxiv_id)
        if not detail:
            return json.dumps(
                {"error": "not_found", "message": f"Paper {arxiv_id} not in library."},
                ensure_ascii=False,
            )
        return json.dumps(detail, ensure_ascii=False)

    @mcp.tool()
    def research_remove_paper(arxiv_id: str, delete_files: bool = True) -> str:
        """Remove a paper and all derived data (summaries, ideas, embeddings)
        from the local library.

        Args:
            arxiv_id: arXiv ID of the paper to remove.
            delete_files: Also delete cached files (default True).
        """
        removed = lib.remove_paper(arxiv_id)
        return json.dumps(
            {"removed": removed, "arxiv_id": arxiv_id},
            ensure_ascii=False,
        )

    @mcp.tool()
    def research_generate_summary(
        arxiv_id: str,
        sections: list[str] | None = None,
        force: bool = False,
    ) -> str:
        """Generate or retrieve a structured summary of a paper.  Sections:
        problem_statement, methodology, findings, ablations, discussion,
        limitations, overall.

        Args:
            arxiv_id: arXiv ID of the paper.
            sections: Which sections to generate (default: all).
            force: Regenerate even if cached (default False).
        """
        try:
            result = summarizer.summarize(arxiv_id, sections=sections, force=force)
            return json.dumps(
                {"arxiv_id": arxiv_id, "summaries": result},
                ensure_ascii=False,
            )
        except ValueError as exc:
            return json.dumps(
                {"error": "not_found", "message": str(exc)},
                ensure_ascii=False,
            )

    @mcp.tool()
    def research_generate_ideas(
        arxiv_id: str,
        num_ideas: int = 3,
        focus_area: str = "methodological",
    ) -> str:
        """Generate novel research ideas based on a paper's constraints and
        inductive biases.  Each idea includes suggested search queries for
        novelty verification.

        Args:
            arxiv_id: Source paper's arXiv ID.
            num_ideas: Number of ideas (1–5, default 3).
            focus_area: theoretical, applied, methodological, or hybrid.
        """
        try:
            result = ideas_svc.generate_ideas(
                arxiv_id, num_ideas=num_ideas, focus_area=focus_area,
            )
            return json.dumps(
                {"arxiv_id": arxiv_id, "ideas": result},
                ensure_ascii=False,
            )
        except ValueError as exc:
            return json.dumps(
                {"error": "not_found", "message": str(exc)},
                ensure_ascii=False,
            )

    @mcp.tool()
    def research_verify_novelty(
        idea_id: int,
        search_query: str = "",
    ) -> str:
        """Run a novelty re-verification on a previously generated idea.
        Checks the local library and arXiv for similar work, then uses the
        LLM to judge overlap.  Returns a verdict: likely_novel,
        needs_review, or similar_exists.

        Args:
            idea_id: Database ID of the idea (from generate_ideas output).
            search_query: Optional override query for the arXiv check.
        """
        try:
            result = novelty_svc.verify_novelty(
                idea_id, search_query=search_query or None,
            )
            return json.dumps(result, ensure_ascii=False)
        except ValueError as exc:
            return json.dumps(
                {"error": "not_found", "message": str(exc)},
                ensure_ascii=False,
            )
    @mcp.tool()
    def research_list_library(
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        primary_category: str = "",
        min_citations: int = 0,
        venue: str = "",
    ) -> str:
        """List papers in the local library with pagination and optional filters.

        Args:
            limit: Page size (default 20).
            offset: Pagination offset (default 0).
            sort_by: Sort key — citation_count, published, or created_at.
            primary_category: Optional arXiv category filter (e.g. cs.LG).
            min_citations: Filter out papers with fewer citations (0 = no filter).
            venue: Partial venue/conference name filter (e.g. "NeurIPS").
        """
        result = lib.list_library(
            limit=limit, offset=offset, sort_by=sort_by,
            primary_category=primary_category or None,
            min_citations=min_citations or None,
            venue=venue or None,
        )
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    def research_get_activity_log(limit: int = 50, action_type: str = "") -> str:
        """Return recent agent actions from the activity log for supervision.

        Args:
            limit: Number of entries (default 50).
            action_type: Optional filter — search, import, summarize, idea,
                novelty, query, remove, enrich.
        """
        rows = ctx.db.list_activity(limit=limit, action_type=action_type or None)
        return json.dumps(
            {
                "activities": [
                    {
                        "id": r.id,
                        "action_type": r.action_type,
                        "arxiv_id": r.arxiv_id,
                        "query": r.query,
                        "status": r.status,
                        "metadata": r.metadata_json,
                        "created_at": r.created_at,
                    }
                    for r in rows
                ]
            },
            ensure_ascii=False,
        )

    return mcp

def main() -> int:
    """Entry point for the MCP server."""
    settings = get_settings()
    settings.ensure_dirs()
    ctx = create_app()
    mcp = create_mcp_server(ctx)
    transport = settings.mcp_transport
    logger.info("Starting MCP server (transport=%s)", transport)
    try:
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_app(ctx)
    return 0

if __name__ == "__main__":
    sys.exit(main())
