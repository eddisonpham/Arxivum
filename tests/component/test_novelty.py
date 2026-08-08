"""Component tests for the NoveltyService."""

import json
import pytest

from src.inference.llm import StubLLM


class TestNoveltyVerification:
    def test_verify_novelty_likely_novel(self, app_context):
        """With no similar papers in library or arXiv, verdict should be likely_novel."""
        # Use a custom LLM that returns the right shape per prompt type.
        def _respond(msgs):
            content = " ".join(m.get("content", "") for m in msgs)
            if "JSON array" in content and "ideas" in content:
                return json.dumps([{
                    "title": "Invert the attention pattern",
                    "summary": "Substitute the softmax attention with a contrary rule.",
                    "extension": "Inverts the prior assumption.",
                    "next_steps": ["ablate softmax"],
                    "search_queries": ["contrary attention"],
                }])
            if "novelty assessor" in content or "core claim" in content:
                return json.dumps({
                    "verdict": "likely_novel", "confidence": 0.95,
                    "reason": "No similar work found.",
                })
            # Constraints request falls through to a sensible default.
            return ("{\"assumptions\":[\"data is iid\"],"
                    "\"inductive_biases\":[\"softmax normalization\"],"
                    "\"limitations\":[\"single corpus\"],"
                    "\"domain\":\"NLP\",\"key_method\":\"attention\"}")
        custom_llm = StubLLM(responder=_respond)
        app_context.models.set_llm(custom_llm)

        # Import a paper, generate an idea, verify novelty
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id

        # Mock arXiv to return no results for novelty search
        app_context.arxiv_client.search.return_value = []

        ideas = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        idea_id = ideas[0]["id"]
        result = app_context.novelty.verify_novelty(idea_id)
        assert result["verdict"] == "likely_novel"
        assert "notes" in result
        assert "similar_arxiv_ids" in result

    def test_verify_novelty_similar_exists(self, app_context):
        """When arXiv returns a similar paper and LLM says similar_exists."""
        def _respond(msgs):
            content = " ".join(m.get("content", "") for m in msgs)
            if "JSON array" in content and "ideas" in content:
                return json.dumps([{
                    "title": "Invert the attention pattern",
                    "summary": "Substitute the softmax attention with a contrary rule.",
                    "extension": "Inverts the prior assumption.",
                    "next_steps": ["ablate softmax"],
                    "search_queries": ["contrary attention"],
                }])
            return json.dumps({
                "verdict": "similar_exists", "confidence": 0.95,
                "reason": "This paper already addresses the idea.",
            })
        custom_llm = StubLLM(responder=_respond)
        app_context.models.set_llm(custom_llm)

        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id

        # Mock arXiv to return a similar paper
        from src.clients.arxiv_client import ArxivPaper
        similar = ArxivPaper(
            arxiv_id="9999.99999",
            title="Very Similar Work",
            authors=["Someone"],
            abstract="This is very similar to the proposed idea.",
        )
        app_context.arxiv_client.search.return_value = [similar]

        ideas = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        idea_id = ideas[0]["id"]
        result = app_context.novelty.verify_novelty(idea_id)
        assert result["verdict"] == "similar_exists"
        assert "9999.99999" in result["similar_arxiv_ids"]




    def test_verify_novelty_logs_activity(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.arxiv_client.search.return_value = []
        ideas = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        result = app_context.novelty.verify_novelty(ideas[0]["id"])
        activities = app_context.db.list_activity(limit=10, action_type="novelty")
        assert len(activities) >= 1
        assert activities[0].status == "completed"

    def test_verify_novelty_nonexistent_idea_raises(self, app_context):
        with pytest.raises(ValueError):
            app_context.novelty.verify_novelty(999)

    def test_verify_novelty_stores_check(self, app_context):
        """The novelty check result should be stored in the DB."""
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.arxiv_client.search.return_value = []
        ideas = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        idea_id = ideas[0]["id"]
        result = app_context.novelty.verify_novelty(idea_id)
        checks = app_context.db.get_novelty_checks(idea_id)
        assert len(checks) >= 1
        assert checks[0].verdict == result["verdict"]

    def test_verify_novelty_with_search_query_override(self, app_context):
        """Passing a search_query should include it in query_terms."""
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.arxiv_client.search.return_value = []
        ideas = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        idea_id = ideas[0]["id"]
        result = app_context.novelty.verify_novelty(idea_id, search_query="custom query")
        assert "custom query" in result["query_terms"]

    def test_local_check_skips_source_paper(self, app_context):
        """The local RAG pre-check should not match the source paper itself."""
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.arxiv_client.search.return_value = []
        ideas = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        idea_id = ideas[0]["id"]
        result = app_context.novelty.verify_novelty(idea_id)
        # The source paper should NOT appear in similar_arxiv_ids from local check
        assert arxiv_id not in result["similar_arxiv_ids"]
