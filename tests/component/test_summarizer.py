"""Component tests for the SummarizerService."""

import pytest


class TestSummarizer:
    def test_summarize_generates_all_sections(self, app_context):
        """summarize() should generate all 7 default sections."""
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        result = app_context.summarizer.summarize(arxiv_id)
        assert "problem_statement" in result
        assert "methodology" in result
        assert "findings" in result
        assert "ablations" in result
        assert "discussion" in result
        assert "limitations" in result
        assert "overall" in result

    def test_summarize_specific_sections(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        result = app_context.summarizer.summarize(arxiv_id, sections=["overall", "methodology"])
        assert "overall" in result
        assert "methodology" in result
        assert "findings" not in result

    def test_summarize_caches(self, app_context):
        """Second call without force should return cached."""
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.summarizer.summarize(arxiv_id)
        # Second call should not call the LLM
        llm = app_context.models.llm
        call_count_before = len(llm.calls)
        app_context.summarizer.summarize(arxiv_id)
        call_count_after = len(llm.calls)
        assert call_count_after == call_count_before  # no new LLM calls

    def test_summarize_force_regenerates(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.summarizer.summarize(arxiv_id)
        llm = app_context.models.llm
        call_count_before = len(llm.calls)
        app_context.summarizer.summarize(arxiv_id, force=True)
        call_count_after = len(llm.calls)
        assert call_count_after > call_count_before

    def test_summarize_nonexistent_raises(self, app_context):
        with pytest.raises(ValueError):
            app_context.summarizer.summarize("9999.99999")

    def test_summarize_logs_activity(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.summarizer.summarize(arxiv_id)
        activities = app_context.db.list_activity(limit=10, action_type="summarize")
        assert len(activities) >= 1
        assert activities[0].status == "completed"

    def test_summarize_indexes_summary_chunks(self, app_context):
        """Generated summary sections should be indexed in ChromaDB."""
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        count_before = app_context.chroma.count()
        app_context.summarizer.summarize(arxiv_id)
        count_after = app_context.chroma.count()
        # At least some summary sections should be indexed (those not "N/A")
        assert count_after > count_before

    def test_get_cached(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.summarizer.summarize(arxiv_id)
        cached = app_context.summarizer.get_cached(arxiv_id)
        assert len(cached) > 0
        assert "overall" in cached
