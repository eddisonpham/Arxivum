"""MCP server — exposes research tools to coding agents.

Uses the official ``mcp`` Python SDK's high-level ``MCPServer`` API.
The server exposes exactly three tools (down from nine in the
pre-refactor surface) so that coding-agent tool selection stays
disambiguated at the name level rather than through argument
inference:

  * ``library.search``  — search arXiv, import results, optionally
                         enrich / summarise / extract on the same
                         call. Replaces the old ``research_search_papers``
                         and folds post-import actions into a single
                         request.
  * ``library.paper``   — get the full detail of one paper, or run a
                         single mutation (summary / extract / ideas /
                         verify_novelty / approve_idea / reject_idea /
                         remove) by setting ``action``. Replaces the
                         seven old ``research_get_paper_details`` /
                         ``_extract_paper`` / ``_generate_summary`` /
                         ``_generate_ideas`` / ``_verify_novelty`` /
                         ``_remove_paper`` / ``_approve_idea`` tools.
  * ``library.find``    — query the local library with RAG (``mode='rag'``)
                         or list papers (``mode='list'``) or return
                         the activity log (``mode='activity'``).
                         Replaces the three old ``research_query_library`` /
                         ``_list_library`` / ``_get_activity_log`` tools.

Run via::

    python -m src.mcp_server          # stdio (default)
    MCP_TRANSPORT=sse python -m src.mcp_server   # SSE
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Literal

from src.app import AppContext, create_app, shutdown_app
from src.config import get_settings

logger = logging.getLogger(__name__)

# ── Pure dispatch helpers (unit-testable, no MCP dependency) ─────────────

# library.paper actions
PAPER_ACTION_DETAIL = "detail"
PAPER_ACTION_SUMMARY = "summary"
PAPER_ACTION_EXTRACT = "extract"
PAPER_ACTION_IDEAS = "ideas"
PAPER_ACTION_VERIFY = "verify_novelty"
PAPER_ACTION_APPROVE = "approve_idea"
PAPER_ACTION_REJECT = "reject_idea"
PAPER_ACTION_REMOVE = "remove"

PAPER_ACTIONS = (
    PAPER_ACTION_DETAIL, PAPER_ACTION_SUMMARY, PAPER_ACTION_EXTRACT,
    PAPER_ACTION_IDEAS, PAPER_ACTION_VERIFY, PAPER_ACTION_APPROVE,
    PAPER_ACTION_REJECT, PAPER_ACTION_REMOVE,
)

# library.find modes
FIND_MODE_RAG = "rag"
FIND_MODE_LIST = "list"
FIND_MODE_ACTIVITY = "activity"

FIND_MODES = (FIND_MODE_RAG, FIND_MODE_LIST, FIND_MODE_ACTIVITY)


def search_dispatch(
    ctx: AppContext,
    query: str,
    max_results: int = 10,
    primary_category: str = "",
    auto_enrich: bool = False,
    summarize: bool = False,
    extract: bool = False,
) -> dict:
    """Search arXiv, import results, optionally run post-import actions.

    Returns a dict with the imported papers plus optional per-paper
    summaries and structured extractions.
    """
    max_results = min(max(max_results, 1), 50)
    results = ctx.library.search_and_import(
        query=query,
        max_results=max_results,
        primary_category=primary_category or None,
        auto_enrich=auto_enrich,
    )
    papers = [r.to_dict() for r in results]
    if summarize and results:
        for r in results[:5]:
            try:
                ctx.summarizer.summarize(r.arxiv_id)
            except Exception as exc:
                logger.warning("Summary failed for %s: %s", r.arxiv_id, exc)
    if extract and results:
        for r in results[:5]:
            try:
                ctx.extractor.extract(r.arxiv_id)
            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", r.arxiv_id, exc)
    return {
        "papers": papers,
        "total_found": len(results),
        "message": f"Imported {len(results)} paper(s) into the library.",
        "indexed_actions": [
            *(["enrich"] if auto_enrich else []),
            *(["summarize"] if summarize else []),
            *(["extract"] if extract else []),
        ],
    }


def paper_dispatch(
    ctx: AppContext,
    arxiv_id: str,
    action: str = PAPER_ACTION_DETAIL,
    idea_id: int | None = None,
    num_ideas: int = 3,
    focus_area: str = "methodological",
    status: str = "approved",
    search_query: str = "",
    force: bool = False,
    include: list[str] | None = None,
) -> dict:
    """Get a paper's detail, or run a single mutation action on it.

    ``action`` is one of: detail (default), summary, extract, ideas,
    verify_novelty, approve_idea, reject_idea, remove. Each action is
    a single, atomic operation; the response shape matches the action.
    """
    if action not in PAPER_ACTIONS:
        raise ValueError(
            f"action={action!r} not in {PAPER_ACTIONS}"
        )

    if action == PAPER_ACTION_DETAIL:
        detail = ctx.library.get_paper_detail(arxiv_id)
        if not detail:
            return {"error": "not_found",
                    "message": f"Paper {arxiv_id} not in library."}
        if include:
            keys = set(include) & set(detail.keys())
            return {k: detail[k] for k in keys}
        return detail

    if action == PAPER_ACTION_SUMMARY:
        return {
            "arxiv_id": arxiv_id,
            "summaries": ctx.summarizer.summarize(
                arxiv_id,
                sections=None,
                force=force,
            ),
        }

    if action == PAPER_ACTION_EXTRACT:
        schema = ctx.extractor.extract(arxiv_id, force=force)
        return {"arxiv_id": arxiv_id, "extraction": schema}

    if action == PAPER_ACTION_IDEAS:
        ideas = ctx.ideas.generate_ideas(
            arxiv_id, num_ideas=num_ideas, focus_area=focus_area,
        )
        return {"arxiv_id": arxiv_id, "ideas": ideas}

    if action == PAPER_ACTION_VERIFY:
        if idea_id is None:
            raise ValueError("verify_novelty requires idea_id")
        return ctx.novelty.verify_novelty(
            idea_id, search_query=search_query or None,
        )

    if action in (PAPER_ACTION_APPROVE, PAPER_ACTION_REJECT):
        if idea_id is None:
            raise ValueError(f"{action} requires idea_id")
        target_status = "approved" if action == PAPER_ACTION_APPROVE else "rejected"
        if status not in ("approved", "rejected"):
            target_status = status
        ok = ctx.ideas.update_status(idea_id, target_status)
        return {"idea_id": idea_id, "status": target_status,
                "ok": ok, "action": action}

    if action == PAPER_ACTION_REMOVE:
        removed = ctx.library.remove_paper(arxiv_id)
        return {"removed": removed, "arxiv_id": arxiv_id, "action": action}

    raise AssertionError(f"unreachable action: {action!r}")  # pragma: no cover


def find_dispatch(
    ctx: AppContext,
    mode: str = FIND_MODE_RAG,
    query: str = "",
    top_k: int = 5,
    min_citations: int = 0,
    venue: str = "",
    primary_category: str = "",
    rerank: bool = True,
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "created_at",
    action_type: str = "",
) -> dict:
    """Three-mode dispatcher: local RAG, list, or activity log."""
    if mode not in FIND_MODES:
        raise ValueError(f"mode={mode!r} not in {FIND_MODES}")

    if mode == FIND_MODE_RAG:
        results = ctx.library.query_library(
            query=query or "",
            top_k=top_k,
            min_citations=min_citations or None,
            venue=venue or None,
            primary_category=primary_category or None,
            rerank=rerank,
        )
        return {"mode": "rag", "results": [r.to_dict() for r in results]}

    if mode == FIND_MODE_LIST:
        result = ctx.library.list_library(
            limit=limit, offset=offset, sort_by=sort_by,
            primary_category=primary_category or None,
            min_citations=min_citations or None,
            venue=venue or None,
        )
        return {"mode": "list", **result}

    # mode == FIND_MODE_ACTIVITY
    rows = ctx.db.list_activity(limit=limit, action_type=action_type or None)
    return {
        "mode": "activity",
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
        ],
    }


# ── MCP tool registration (only 3 tools) ─────────────────────────────────

def create_mcp_server(ctx: AppContext):
    """Register exactly three tools on the MCP server."""
    from mcp.server import MCPServer

    mcp = MCPServer("research-library")

    @mcp.tool()
    def library_search(
        query: str,
        max_results: int = 10,
        primary_category: str = "",
        auto_enrich: bool = False,
        summarize: bool = False,
        extract: bool = False,
    ) -> str:
        """Search arXiv and import results into the local library.

        Args:
            query: arXiv search query (natural language or field-tagged).
            max_results: 1-50, default 10.
            primary_category: Optional filter (e.g. cs.LG).
            auto_enrich: Fetch Semantic Scholar metrics on import.
            summarize: Also generate 7-section summary on each result.
            extract: Also generate structured bibliographic schema for each.

        Returns:
            JSON object with imported papers and the per-paper actions
            that ran ("enrich", "summarize", "extract").
        """
        out = search_dispatch(
            ctx, query=query,
            max_results=max_results,
            primary_category=primary_category,
            auto_enrich=auto_enrich,
            summarize=summarize,
            extract=extract,
        )
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def library_paper(
        arxiv_id: str,
        action: Literal[
            "detail", "summary", "extract", "ideas",
            "verify_novelty", "approve_idea", "reject_idea", "remove",
        ] = "detail",
        idea_id: int | None = None,
        num_ideas: int = 3,
        focus_area: str = "methodological",
        status: Literal["approved", "rejected"] = "approved",
        search_query: str = "",
        force: bool = False,
        include: list[str] | None = None,
    ) -> str:
        """Get or mutate one paper in the local library.

        Args:
            arxiv_id: arXiv ID of the paper.
            action: One of the eight actions listed above. Default
                ``detail`` returns the full record; the others trigger
                a single mutation. Each action returns a JSON object
                shaped to that action (not a generic envelope).
            idea_id: Required for verify_novelty / approve_idea / reject_idea.
            num_ideas: How many ideas to generate on ``action='ideas'``
                (1-5, default 3).
            focus_area: theoretical / applied / methodological / hybrid.
            status: approved / rejected for idea-status actions.
            search_query: Optional override for ``action='verify_novelty'``.
            force: Re-generate even if cached (summary / extract).
            include: If set, restrict the detail response to these
                top-level keys (``ideas``, ``summaries``, ``extraction``,
                ``metrics``, ...).

        Returns:
            JSON object whose shape depends on ``action``.
        """
        try:
            out = paper_dispatch(
                ctx, arxiv_id=arxiv_id, action=action,
                idea_id=idea_id, num_ideas=num_ideas,
                focus_area=focus_area, status=status,
                search_query=search_query, force=force,
                include=include,
            )
            return json.dumps(out, ensure_ascii=False)
        except ValueError as exc:
            return json.dumps(
                {"error": "not_found_or_invalid", "message": str(exc)},
                ensure_ascii=False,
            )

    @mcp.tool()
    def library_find(
        mode: Literal["rag", "list", "activity"] = "rag",
        query: str = "",
        top_k: int = 5,
        min_citations: int = 0,
        venue: str = "",
        primary_category: str = "",
        rerank: bool = True,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        action_type: str = "",
    ) -> str:
        """Query the local library in one of three modes.

        Args:
            mode: One of:
                ``rag`` — vector search + metadata filter over the
                local library. Use the natural-language ``query`` and
                optional filters (``top_k``, ``min_citations``,
                ``venue``, ``primary_category``, ``rerank``).
                ``list`` — paginated browsing of all papers in the
                library. Uses ``limit``, ``offset``, ``sort_by`` and
                the same filters as a hard-where.
                ``activity`` — agent action log. Uses ``limit`` and
                optional ``action_type`` filter.
            query: Natural-language query (rag mode).
            top_k: How many results to return (rag mode, default 5).
            min_citations: Filter out lower-cited papers (0 = no filter).
            venue: Partial venue name filter (e.g. "NeurIPS").
            primary_category: arXiv category filter (e.g. cs.LG).
            rerank: Apply cross-encoder reranking (rag mode).
            limit: Page size (list / activity, default 20).
            offset: Pagination offset (list mode).
            sort_by: Sort key for list mode (citation_count /
                published / created_at).
            action_type: Filter activity by action type.

        Returns:
            JSON object keyed by ``mode``.
        """
        try:
            out = find_dispatch(
                ctx, mode=mode, query=query, top_k=top_k,
                min_citations=min_citations, venue=venue,
                primary_category=primary_category, rerank=rerank,
                limit=limit, offset=offset, sort_by=sort_by,
                action_type=action_type,
            )
            return json.dumps(out, ensure_ascii=False)
        except ValueError as exc:
            return json.dumps(
                {"error": "invalid_mode", "message": str(exc)},
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
    logger.info("Starting MCP server (transport=%s) with 3 tools", transport)
    try:
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_app(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
