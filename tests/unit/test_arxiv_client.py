"""Unit tests for the arXiv client."""

import pytest
from unittest.mock import MagicMock, patch

from src.clients.arxiv_client import (
    ArxivClient, ArxivPaper, normalize_arxiv_id, strip_version,
)


class TestArxivIDNormalization:
    def test_normalize_from_url(self):
        assert normalize_arxiv_id("http://arxiv.org/abs/2106.00001") == "2106.00001"

    def test_normalize_from_url_with_version(self):
        assert normalize_arxiv_id("http://arxiv.org/abs/2106.00001v2") == "2106.00001v2"

    def test_normalize_with_prefix(self):
        assert normalize_arxiv_id("arXiv:2106.00001") == "2106.00001"

    def test_normalize_bare_id(self):
        assert normalize_arxiv_id("2106.00001") == "2106.00001"

    def test_normalize_old_style(self):
        assert normalize_arxiv_id("hep-th/9901001") == "hep-th/9901001"

    def test_normalize_pdf_url(self):
        assert normalize_arxiv_id("http://arxiv.org/pdf/2106.00001") == "2106.00001"

    def test_normalize_invalid_raises(self):
        with pytest.raises(ValueError):
            normalize_arxiv_id("not an arxiv id at all")

    def test_strip_version(self):
        assert strip_version("2106.00001v3") == "2106.00001"
        assert strip_version("2106.00001") == "2106.00001"


class TestArxivPaper:
    def test_to_dict(self):
        p = ArxivPaper(
            arxiv_id="2106.00001", title="Test", authors=["A", "B"],
            abstract="Abstract text", primary_category="cs.LG",
        )
        d = p.to_dict()
        assert d["arxiv_id"] == "2106.00001"
        assert d["title"] == "Test"
        assert d["authors"] == ["A", "B"]
        assert d["abstract"] == "Abstract text"
        assert d["primary_category"] == "cs.LG"

    def test_from_result(self):
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2106.00001"
        mock_result.title = "Paper Title"
        mock_result.summary = "Abstract"
        mock_result.published = None
        mock_result.updated = None
        mock_result.categories = ["cs.CL", "cs.AI"]
        mock_result.primary_category = "cs.CL"
        mock_result.pdf_url = "http://arxiv.org/pdf/2106.00001"
        mock_result.doi = None
        mock_result.journal_ref = None
        mock_result.comment = None
        author1 = MagicMock(); author1.name = "Alice"
        author2 = MagicMock(); author2.name = "Bob"
        mock_result.authors = [author1, author2]

        p = ArxivPaper.from_result(mock_result)
        assert p.arxiv_id == "2106.00001"
        assert p.title == "Paper Title"
        assert p.authors == ["Alice", "Bob"]
        assert p.abstract == "Abstract"
        assert p.primary_category == "cs.CL"
        assert p.categories == ["cs.CL", "cs.AI"]


class TestArxivClientSearch:
    def test_search_returns_papers(self):
        """Search should return ArxivPaper objects (mocked)."""
        client = ArxivClient(delay_seconds=0.01)
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2106.00001"
        mock_result.title = "Test Paper"
        mock_result.summary = "An abstract."
        mock_result.published = None
        mock_result.updated = None
        mock_result.categories = ["cs.LG"]
        mock_result.primary_category = "cs.LG"
        mock_result.pdf_url = "http://arxiv.org/pdf/2106.00001"
        mock_result.doi = None
        mock_result.journal_ref = None
        mock_result.comment = None
        mock_result.authors = []
        mock_result.links = []

        with patch.object(client._client, "results", return_value=iter([mock_result])):
            papers = client.search("test query", max_results=1)
        assert len(papers) == 1
        assert papers[0].arxiv_id == "2106.00001"
        assert papers[0].title == "Test Paper"

    def test_search_with_category(self):
        """Search should include category in the query string."""
        client = ArxivClient(delay_seconds=0.01)
        with patch.object(client._client, "results", return_value=iter([])):
            papers = client.search("attention", primary_category="cs.LG")
        assert papers == []

    def test_get_paper(self):
        """get_paper should return a single paper or None."""
        client = ArxivClient(delay_seconds=0.01)
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2106.00001"
        mock_result.title = "Found"
        mock_result.summary = "abs"
        mock_result.published = None
        mock_result.updated = None
        mock_result.categories = []
        mock_result.primary_category = "cs.LG"
        mock_result.pdf_url = ""
        mock_result.doi = None
        mock_result.journal_ref = None
        mock_result.comment = None
        mock_result.authors = []
        mock_result.links = []

        with patch.object(client._client, "results", return_value=iter([mock_result])):
            paper = client.get_paper("2106.00001")
        assert paper is not None
        assert paper.title == "Found"

    def test_get_paper_not_found(self):
        client = ArxivClient(delay_seconds=0.01)
        with patch.object(client._client, "results", return_value=iter([])):
            paper = client.get_paper("9999.99999")
        assert paper is None
