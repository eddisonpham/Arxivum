"""Component tests for the LibraryService."""

import json
import pytest

from src.db.models import ActivityRow, PaperRow


class TestSearchAndImport:
    def test_search_imports_papers(self, app_context):
        results = app_context.library.search_and_import("attention", max_results=2)
        assert len(results) == 2
        assert all(r.imported for r in results)
        assert app_context.db.count_papers() == 2

    def test_search_logs_activity(self, app_context):
        app_context.library.search_and_import("transformer", max_results=1)
        activities = app_context.db.list_activity(limit=10, action_type="search")
        assert len(activities) >= 1
        assert activities[0].status == "completed"

    def test_search_import_logs_activity(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        imports = app_context.db.list_activity(limit=10, action_type="import")
        assert len(imports) >= 1

    def test_search_with_enrich(self, app_context):
        """auto_enrich=True should call S2 and store metrics."""
        results = app_context.library.search_and_import("bert", max_results=1, auto_enrich=True)
        assert len(results) >= 1
        # The mock S2 returns citation_count=1000
        metrics = app_context.db.get_metrics(results[0].arxiv_id)
        assert metrics is not None
        assert metrics.citation_count == 1000

    def test_search_imports_into_chroma(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        assert app_context.chroma.count() >= 2  # abstract + title chunks


class TestQueryLibrary:
    def test_query_returns_results(self, app_context):
        app_context.library.search_and_import("attention", max_results=2)
        results = app_context.library.query_library("attention mechanism", top_k=5)
        assert len(results) > 0
        assert all(hasattr(r, "arxiv_id") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_query_logs_activity(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        app_context.library.query_library("anything", top_k=1)
        queries = app_context.db.list_activity(limit=10, action_type="query")
        assert len(queries) >= 1
        assert queries[0].status == "completed"

    def test_query_empty_library(self, app_context):
        results = app_context.library.query_library("nonexistent", top_k=5)
        assert results == []

    def test_query_with_rerank(self, app_context):
        app_context.library.search_and_import("attention", max_results=2)
        results = app_context.library.query_library("transformer", top_k=2, rerank=True)
        assert len(results) <= 2

    def test_query_without_rerank(self, app_context):
        app_context.library.search_and_import("attention", max_results=2)
        results = app_context.library.query_library("transformer", top_k=2, rerank=False)
        assert len(results) <= 2

    def test_query_result_to_dict(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        results = app_context.library.query_library("test", top_k=1)
        if results:
            d = results[0].to_dict()
            assert "arxiv_id" in d
            assert "title" in d
            assert "score" in d


class TestRemovePaper:
    def test_remove_existing(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        assert app_context.library.remove_paper(arxiv_id) is True
        assert app_context.db.get_paper(arxiv_id) is None

    def test_remove_nonexistent(self, app_context):
        assert app_context.library.remove_paper("9999.99999") is False

    def test_remove_deletes_from_chroma(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        count_before = app_context.chroma.count()
        app_context.library.remove_paper(arxiv_id)
        count_after = app_context.chroma.count()
        assert count_after < count_before


class TestGetPaperDetail:
    def test_detail_returns_full_data(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        detail = app_context.library.get_paper_detail(arxiv_id)
        assert detail is not None
        assert detail["arxiv_id"] == arxiv_id
        assert "title" in detail
        assert "abstract" in detail
        assert "authors" in detail
        assert "summaries" in detail
        assert "ideas" in detail

    def test_detail_nonexistent(self, app_context):
        assert app_context.library.get_paper_detail("9999.99999") is None


class TestListLibrary:
    def test_list_returns_papers(self, app_context):
        app_context.library.search_and_import("test", max_results=2)
        result = app_context.library.list_library(limit=10)
        assert result["total"] == 2
        assert len(result["papers"]) == 2

    def test_list_pagination(self, app_context):
        app_context.library.search_and_import("test", max_results=2)
        page1 = app_context.library.list_library(limit=1, offset=0)
        page2 = app_context.library.list_library(limit=1, offset=1)
        assert len(page1["papers"]) == 1
        assert len(page2["papers"]) == 1


class TestIndexSummarySection:
    def test_index_summary_creates_chunk(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        count_before = app_context.chroma.count()
        app_context.library.index_summary_section(arxiv_id, "methodology", "Uses method X.")
        count_after = app_context.chroma.count()
        assert count_after == count_before + 1

    def test_index_summary_nonexistent_paper_raises(self, app_context):
        with pytest.raises(ValueError):
            app_context.library.index_summary_section("9999.99999", "overall", "text")
