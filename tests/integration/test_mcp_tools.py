"""Integration tests for the 3-tool MCP surface.

These tests use the pure dispatchers (``search_dispatch``,
``paper_dispatch``, ``find_dispatch``) so they need not run an MCP
server end to end — the dispatch logic and the registration shape
are the parts that matter.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.db.models import IdeaRow
from src.mcp_server.server import (
    FIND_MODE_RAG,
    PAPER_ACTIONS,
    create_mcp_server,
    find_dispatch,
    paper_dispatch,
    search_dispatch,
)


def test_mcp_tool_count_is_three():
    """Tool surface reduction: 9 -> 3."""
    from src.app import create_app, shutdown_app

    ctx = create_app(db_path=":memory:", chroma_path="")
    try:
        mcp = create_mcp_server(ctx)
        tools = asyncio.run(mcp.list_tools())
        names = sorted(t.name for t in tools)
        assert tools is not None
        assert names == ["library_find", "library_paper", "library_search"], names
    finally:
        shutdown_app(ctx)


def test_tool_input_schemas_are_well_formed():
    from src.app import create_app, shutdown_app

    ctx = create_app(db_path=":memory:", chroma_path="")
    try:
        mcp = create_mcp_server(ctx)
        tools = asyncio.run(mcp.list_tools())
        for t in tools:
            schema = t.input_schema or {}
            assert schema.get("type") == "object"
            assert "properties" in schema
    finally:
        shutdown_app(ctx)


def test_paper_actions_constant_matches_old_capabilities():
    """Every old ``research_*`` capability is reachable through PAPER_ACTIONS."""
    expected = {
        "detail",        # research_get_paper_details
        "summary",       # research_generate_summary
        "extract",       # research_extract_paper
        "ideas",         # research_generate_ideas
        "verify_novelty",  # research_verify_novelty
        "approve_idea",  # (was approve via status endpoint)
        "reject_idea",
        "remove",        # research_remove_paper
    }
    assert set(PAPER_ACTIONS) == expected, set(PAPER_ACTIONS) ^ expected


def test_search_dispatch_imports_papers(app_context):
    """library.search equivalent: search_and_import returns papers."""
    from src.mcp_server.server import search_dispatch
    out = search_dispatch(
        app_context, query="attention", max_results=1,
    )
    assert out["total_found"] >= 1
    assert out["papers"][0]["arxiv_id"]


def test_search_dispatch_with_summarize_runs_summarizer(app_context, stub_llm):
    from src.mcp_server.server import search_dispatch
    app_context.models.set_llm(stub_llm)
    out = search_dispatch(
        app_context, query="attention", max_results=1,
        summarize=True,
    )
    assert out["indexed_actions"] == ["summarize"]


def test_search_dispatch_with_extract_runs_extractor(app_context, stub_llm):
    from src.mcp_server.server import search_dispatch
    app_context.models.set_llm(stub_llm)
    out = search_dispatch(
        app_context, query="attention", max_results=1,
        extract=True,
    )
    assert out["indexed_actions"] == ["extract"]


def test_paper_dispatch_detail_works(app_context, imported_paper):
    out = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id,
    )
    assert "title" in out
    assert out["arxiv_id"] == imported_paper.arxiv_id


def test_paper_dispatch_detail_include_filters_keys(app_context, imported_paper):
    out = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id,
        include=["ideas"],
    )
    assert "ideas" in out
    assert "summaries" not in out


def test_paper_dispatch_summary(app_context, imported_paper, stub_llm):
    app_context.models.set_llm(stub_llm)
    out = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id, action="summary",
    )
    assert "summaries" in out
    assert "problem_statement" in out["summaries"]


def test_paper_dispatch_extract(app_context, imported_paper, stub_llm):
    app_context.models.set_llm(stub_llm)
    out = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id, action="extract",
    )
    assert "extraction" in out


def test_paper_dispatch_ideas(app_context, imported_paper, stub_llm):
    app_context.models.set_llm(stub_llm)
    out = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id,
        action="ideas", num_ideas=3,
    )
    assert "ideas" in out
    assert len(out["ideas"]) == 3


def test_paper_dispatch_approve_idea(app_context, imported_paper, stub_llm):
    app_context.models.set_llm(stub_llm)
    ideas = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id,
        action="ideas", num_ideas=1,
    )
    idea_id = ideas["ideas"][0]["id"]
    out = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id,
        action="approve_idea", idea_id=idea_id,
    )
    assert out["status"] == "approved"
    assert out["ok"] is True


def test_paper_dispatch_verify_novelty_requires_idea_id(app_context, imported_paper):
    with pytest.raises(ValueError, match="idea_id"):
        paper_dispatch(
            app_context, arxiv_id=imported_paper.arxiv_id,
            action="verify_novelty",
        )


def test_paper_dispatch_remove(app_context, imported_paper):
    out = paper_dispatch(
        app_context, arxiv_id=imported_paper.arxiv_id, action="remove",
    )
    assert out["removed"] is True
    assert app_context.db.get_paper(imported_paper.arxiv_id) is None


def test_find_dispatch_rag_mode(app_context, imported_paper, stub_embedder):
    app_context.models._embedder = stub_embedder
    out = find_dispatch(
        app_context, mode="rag", query="attention", top_k=3,
    )
    assert out["mode"] == "rag"
    assert isinstance(out["results"], list)


def test_find_dispatch_list_mode(app_context, imported_paper):
    out = find_dispatch(
        app_context, mode="list", limit=5, sort_by="created_at",
    )
    assert out["mode"] == "list"
    assert "papers" in out


def test_find_dispatch_activity_mode(app_context, imported_paper):
    out = find_dispatch(
        app_context, mode="activity", limit=5,
    )
    assert out["mode"] == "activity"
    assert "activities" in out


def test_find_dispatch_invalid_mode_raises(app_context):
    with pytest.raises(ValueError, match="mode"):
        find_dispatch(app_context, mode="bogus")


def test_paper_dispatch_invalid_action_raises(app_context, imported_paper):
    with pytest.raises(ValueError, match="action"):
        paper_dispatch(
            app_context, arxiv_id=imported_paper.arxiv_id, action="bogus",
        )
