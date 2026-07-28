"""Unit tests for the Semantic Scholar client."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.clients.s2_client import (
    MISSING_CITATION, S2Metrics, SemanticScholarClient, _parse_paper,
)


class TestParsePaper:
    def test_parse_full_data(self):
        data = {
            "paperId": "abc123",
            "citationCount": 500,
            "influentialCitationCount": 25,
            "venue": "NeurIPS",
            "publicationVenue": {"name": "NeurIPS"},
        }
        m = _parse_paper("2106.00001", data)
        assert m.arxiv_id == "2106.00001"
        assert m.citation_count == 500
        assert m.influential_citation_count == 25
        assert m.venue == "NeurIPS"
        assert m.s2_paper_id == "abc123"

    def test_parse_null_data(self):
        m = _parse_paper("2106.00001", None)
        assert m.citation_count == MISSING_CITATION
        assert m.venue is None

    def test_parse_empty_venue_falls_back_to_publication_venue(self):
        data = {"venue": "", "publicationVenue": {"name": "ICML"}}
        m = _parse_paper("1.1", data)
        assert m.venue == "ICML"

    def test_parse_missing_fields(self):
        data = {"paperId": "x"}
        m = _parse_paper("1.1", data)
        assert m.citation_count == MISSING_CITATION
        assert m.venue is None

    def test_to_dict(self):
        m = S2Metrics(arxiv_id="1.1", citation_count=10, venue="ICLR")
        d = m.to_dict()
        assert d["arxiv_id"] == "1.1"
        assert d["citation_count"] == 10
        assert d["venue"] == "ICLR"


class TestS2Client:
    @pytest.mark.asyncio
    async def test_fetch_paper_success(self):
        """fetch_paper should parse S2 response correctly."""
        client = SemanticScholarClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "paperId": "abc",
            "citationCount": 100,
            "influentialCitationCount": 5,
            "venue": "NeurIPS",
            "publicationVenue": None,
        }
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            m = await client.fetch_paper("2106.00001")
        assert m.citation_count == 100
        assert m.venue == "NeurIPS"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_paper_not_found(self):
        """fetch_paper should return sentinel metrics on 404."""
        client = SemanticScholarClient()
        exc = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        with patch.object(client, "_request", new_callable=AsyncMock, side_effect=exc):
            m = await client.fetch_paper("9999.99999")
        assert m.citation_count == MISSING_CITATION
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_paper_network_error(self):
        """fetch_paper should return sentinel metrics on network error."""
        client = SemanticScholarClient()
        with patch.object(client, "_request", new_callable=AsyncMock,
                          side_effect=httpx.ConnectError("fail")):
            m = await client.fetch_paper("2106.00001")
        assert m.citation_count == MISSING_CITATION
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_batch_success(self):
        client = SemanticScholarClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"paperId": "a", "citationCount": 10, "venue": "ICML"},
            None,
        ]
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_batch(["2106.00001", "2106.00002"])
        assert len(results) == 2
        assert results["2106.00001"].citation_count == 10
        assert results["2106.00002"].citation_count == MISSING_CITATION
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_batch_empty(self):
        client = SemanticScholarClient()
        results = await client.fetch_batch([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_fetch_batch_network_error(self):
        client = SemanticScholarClient()
        with patch.object(client, "_request", new_callable=AsyncMock,
                          side_effect=httpx.ConnectError("fail")):
            results = await client.fetch_batch(["1.1", "2.2"])
        assert len(results) == 2
        assert all(v.citation_count == MISSING_CITATION for v in results.values())
        await client.aclose()

    @pytest.mark.asyncio
    async def test_batch_queue_enqueue_and_flush(self):
        """Queue should accumulate IDs and flush them in one batch."""
        client = SemanticScholarClient(flush_interval=0.1)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"paperId": "a", "citationCount": 50},
            {"paperId": "b", "citationCount": 30},
        ]
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            task1 = asyncio.create_task(client.enqueue("2106.00001"))
            task2 = asyncio.create_task(client.enqueue("2106.00002"))
            m1 = await task1
            m2 = await task2
        assert m1.citation_count == 50
        assert m2.citation_count == 30
        await client.aclose()
