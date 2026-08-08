"""Integration test for the SSE summarise endpoint (Tier 1.5).

We exercise the route via Starlette's TestClient. The point of the
SSE contract is that the front-end sees a ``skeleton`` event almost
immediately, then ``section`` events as each field becomes visible.
The test uses an injected StubLLM that emits a known JSON output
token-by-token; the endpoint must deliver both events.
"""

from __future__ import annotations

import json

import pytest


def _make_stub_llm(per_token_chunks: list[str]):
    """Return a StubLLM whose stream_chat() yields the given chunks."""
    from src.inference.llm import StubLLM

    chunks = list(per_token_chunks)
    body = "".join(chunks)

    class _StreamingStub(StubLLM):
        def __init__(self):
            super().__init__(default=body)

        def stream_chat(self, messages, temperature=0.3, max_tokens=1024,
                          stop=None):
            for c in chunks:
                yield c

        def chat(self, messages, temperature=0.3, max_tokens=1024, stop=None):
            return body

    return _StreamingStub()


def _import_app():
    # Build a fresh AppContext bound to an in-memory DB + chroma
    # so this test stays offline.
    import os
    os.environ.setdefault("APP_ENV", "test")
    from src.app import create_app, shutdown_app
    from src.inference.llm import StubLLM
    from src.db.models import PaperRow
    from fastapi.testclient import TestClient

    from src.api.main import app as fastapi_app  # uses the running lifespan
    from src.api.main import get_ctx, broadcaster

    # Reset DB / chroma via create_app in test mode.
    ctx = create_app(db_path="", chroma_path="")
    ctx.db.upsert_paper(PaperRow(
        arxiv_id="2401.00001",
        title="MoE Scaling",
        authors=["A"],
        abstract="We study scaling laws for mixture-of-experts routing.",
    ))
    # Inject stub LLM with chunked output
    ctx.models.set_llm(_make_stub_llm([
        "{\n  \"problem_statement\": \"MoE scaling laws.\",\n",
        "  \"methodology\": \"Empirical sweep.\",\n",
        "  \"findings\": \"Power-law slope -0.05.\",\n",
        "  \"ablations\": \"N/A\",\n",
        "  \"discussion\": \"\",\n",
        "  \"limitations\": \"Single compute regime.\",\n",
        "  \"overall\": \"Clean empirical contribution.\"\n}\n",
    ]))
    fastapi_app.dependency_overrides[get_ctx] = lambda: ctx
    client = TestClient(fastapi_app)
    yield client, ctx
    fastapi_app.dependency_overrides.clear()
    shutdown_app(ctx)


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    """Build a fresh AppContext and patch it into the live FastAPI app.

    We don't go through the production lifespan because we need to
    inject our own LLM stub AND our own paper row; the production
    lifespan overwrites the context the moment TestClient starts.
    """
    import os
    os.environ.setdefault("APP_ENV", "test")
    from src.app import create_app, shutdown_app
    from src.db.models import PaperRow
    from fastapi.testclient import TestClient
    import src.api.main as api_main

    db_path = str(tmp_path / "fresh.db")
    ctx = create_app(db_path=db_path, chroma_path="")
    ctx.db.upsert_paper(PaperRow(
        arxiv_id="2401.00001",
        title="MoE Scaling",
        authors=["A"],
        abstract="We study scaling laws for mixture-of-experts routing.",
    ))
    ctx.models.set_llm(_make_stub_llm([
        "{\n  \"problem_statement\": \"MoE scaling laws.\",\n",
        "  \"methodology\": \"Empirical sweep.\",\n",
        "  \"findings\": \"Power-law slope -0.05.\",\n",
        "  \"ablations\": \"N/A\",\n",
        "  \"discussion\": \"\",\n",
        "  \"limitations\": \"Single compute regime.\",\n",
        "  \"overall\": \"Clean empirical contribution.\"\n}\n",
    ]))
    monkeypatch.setattr(api_main, "_ctx", ctx)
    client = TestClient(api_main.app)
    yield client, ctx
    shutdown_app(ctx)


def test_summarize_stream_emits_skeleton_and_done(app_client):
    """A request for a paper with NO cached sections must emit a
    ``skeleton`` event with empty sections, then per-section events,
    then a ``done`` event."""
    client, _ = app_client
    resp = client.post(
        "/api/v1/summaries/2401.00001/stream",
        json={},  # full default sections, no force
    )
    assert resp.status_code == 200
    # SSE streamed response is consumable via iter_lines
    text = resp.text
    events = [line for line in text.split("\n")
              if line.startswith("data: ")]
    assert events, f"no SSE events found in response: {text[:200]}"
    payloads = [json.loads(e[len("data: "):]) for e in events]
    assert "skeleton" in text
    assert payloads[0]["n_cached"] == 0
    section_names = {p["section"] for p in payloads
                     if "section" in p}
    for required in ("problem_statement", "methodology", "findings",
                       "ablations", "discussion", "limitations", "overall"):
        assert required in section_names, (
            f"{required} missing from stream events: {section_names}"
        )
    assert "done" in text


def test_summarize_stream_skeleton_first_and_sections_after(app_client):
    """The skeleton's first emission must precede the first section event."""
    client, _ = app_client
    resp = client.post("/api/v1/summaries/2401.00001/stream", json={})
    text = resp.text
    skeleton_pos = text.find('event: skeleton')
    first_section_pos = text.find('event: section')
    assert skeleton_pos >= 0
    assert first_section_pos >= 0
    assert skeleton_pos < first_section_pos, (
        "skeleton event must fire before any section event"
    )


def test_summarize_stream_cache_hit_only_skeleton_then_done(app_client):
    """A paper that already has all sections cached is fast: only
    ``skeleton`` + ``done`` with ``skipped: True``."""
    client, ctx = app_client
    from src.db.models import SummaryRow
    ctx.db.upsert_summary(SummaryRow(
        id=None, arxiv_id="2401.00001", section="problem_statement",
        content="x", model_used="stub",
    ))
    # pre-fill all 7 sections
    for s in ("problem_statement", "methodology", "findings",
              "ablations", "discussion", "limitations", "overall"):
        ctx.db.upsert_summary(SummaryRow(
            id=None, arxiv_id="2401.00001", section=s,
            content=f"cached {s}", model_used="stub",
        ))
    resp = client.post("/api/v1/summaries/2401.00001/stream", json={})
    text = resp.text
    assert '"skipped": true' in text
    # No section events because everything was cached.
    assert '"section"' not in text, f"unexpected section event: {text[:200]}"


def test_summarize_stream_unknown_paper_returns_404(app_client):
    client, _ = app_client
    resp = client.post("/api/v1/summaries/9999.99999/stream", json={})
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"