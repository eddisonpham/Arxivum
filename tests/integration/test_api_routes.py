"""Integration tests for the FastAPI API routes.

Uses httpx.AsyncClient with the FastAPI app, overriding the app context
with stubs/mocks so no real models or network calls are needed.
"""

import sys
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest_asyncio.fixture
async def api_client(app_context, monkeypatch):
    """Async HTTP client with the app context stubbed.

    Uses an async fixture so the client shares the test's event loop.
    Patches the module-level ``_ctx`` global via ``sys.modules`` to avoid
    the ``src.api.main`` name collision with the re-exported ``main`` function.
    """
    api_mod = sys.modules["src.api.main"]
    monkeypatch.setattr(api_mod, "_ctx", app_context)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, api_client):
        resp = await api_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "papers" in data
        assert "model_state" in data


class TestLibraryEndpoints:
    @pytest.mark.asyncio
    async def test_list_library_empty(self, api_client):
        resp = await api_client.get("/api/v1/library")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["papers"] == []

    @pytest.mark.asyncio
    async def test_search_and_list(self, api_client, app_context):
        # Search
        resp = await api_client.post("/api/v1/library/search", json={
            "query": "attention", "max_results": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 2
        # List
        resp = await api_client.get("/api/v1/library")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_get_paper_detail(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        resp = await api_client.get(f"/api/v1/library/{arxiv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["arxiv_id"] == arxiv_id

    @pytest.mark.asyncio
    async def test_get_paper_not_found(self, api_client):
        resp = await api_client.get("/api/v1/library/9999.99999")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_query_library(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "attention", "max_results": 2,
        })
        resp = await api_client.post("/api/v1/library/query", json={
            "query": "transformer", "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_remove_paper(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        resp = await api_client.delete(f"/api/v1/library/{arxiv_id}")
        assert resp.status_code == 200
        assert resp.json()["removed"] is True


class TestSummaryEndpoints:
    @pytest.mark.asyncio
    async def test_generate_summary(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        resp = await api_client.post(f"/api/v1/summaries/{arxiv_id}", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "summaries" in data
        assert "overall" in data["summaries"]

    @pytest.mark.asyncio
    async def test_get_cached_summaries(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        # Generate first
        await api_client.post(f"/api/v1/summaries/{arxiv_id}", json={})
        # Get cached
        resp = await api_client.get(f"/api/v1/summaries/{arxiv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "summaries" in data

    @pytest.mark.asyncio
    async def test_summary_not_found(self, api_client):
        resp = await api_client.post("/api/v1/summaries/9999.99999", json={})
        assert resp.status_code == 404


class TestIdeaEndpoints:
    @pytest.mark.asyncio
    async def test_generate_ideas(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        resp = await api_client.post(f"/api/v1/ideas/{arxiv_id}", json={
            "num_ideas": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["ideas"]) <= 2

    @pytest.mark.asyncio
    async def test_list_ideas(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        await api_client.post(f"/api/v1/ideas/{arxiv_id}", json={})
        resp = await api_client.get(f"/api/v1/ideas/{arxiv_id}")
        assert resp.status_code == 200
        assert len(resp.json()["ideas"]) > 0

    @pytest.mark.asyncio
    async def test_update_idea_status(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        ideas_resp = await api_client.post(f"/api/v1/ideas/{arxiv_id}", json={})
        idea_id = ideas_resp.json()["ideas"][0]["id"]
        resp = await api_client.post(f"/api/v1/ideas/{idea_id}/status", json={
            "status": "approved",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    @pytest.mark.asyncio
    async def test_idea_status_not_found(self, api_client):
        resp = await api_client.post("/api/v1/ideas/999/status", json={
            "status": "approved",
        })
        assert resp.status_code == 404


class TestActivityEndpoint:
    @pytest.mark.asyncio
    async def test_get_activity(self, api_client, app_context):
        await api_client.post("/api/v1/library/search", json={
            "query": "test", "max_results": 1,
        })
        resp = await api_client.get("/api/v1/activity")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["activities"]) > 0
