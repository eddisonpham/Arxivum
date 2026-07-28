"""Component tests for the IdeaService."""

import json
import pytest

from src.inference.llm import StubLLM


class TestIdeaGeneration:
    def test_generate_ideas_returns_list(self, app_context):
        """generate_ideas should return a list of idea dicts."""
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        result = app_context.ideas.generate_ideas(arxiv_id, num_ideas=2)
        assert isinstance(result, list)
        assert len(result) <= 2
        for idea in result:
            assert "id" in idea
            assert "idea_text" in idea
            assert "search_queries" in idea
            assert "status" in idea
            assert idea["status"] == "pending"

    def test_generate_ideas_stores_in_db(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.ideas.generate_ideas(arxiv_id, num_ideas=3)
        ideas = app_context.db.list_ideas(arxiv_id)
        assert len(ideas) == 3

    def test_generate_ideas_logs_activity(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        activities = app_context.db.list_activity(limit=10, action_type="idea")
        assert len(activities) >= 1
        assert activities[0].status == "completed"

    def test_generate_ideas_nonexistent_raises(self, app_context):
        with pytest.raises(ValueError):
            app_context.ideas.generate_ideas("9999.99999")

    def test_generate_ideas_with_custom_responder(self, app_context):
        """Test with a custom LLM responder that returns specific ideas."""
        custom_llm = StubLLM(responder=lambda msgs: json.dumps([
            {
                "title": "Novel approach A",
                "summary": "Use method A to solve X",
                "extension": "Extends by using A",
                "next_steps": ["step1", "step2"],
                "search_queries": ["method A for X", "novel approach A"],
            }
        ]))
        app_context.models.set_llm(custom_llm)

        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        result = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        assert len(result) == 1
        assert result[0]["title"] == "Novel approach A"
        assert result[0]["search_queries"] == ["method A for X", "novel approach A"]

    def test_update_status(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        ideas = app_context.ideas.generate_ideas(arxiv_id, num_ideas=1)
        idea_id = ideas[0]["id"]
        assert app_context.ideas.update_status(idea_id, "approved") is True
        idea = app_context.db.get_idea(idea_id)
        assert idea.status == "approved"

    def test_update_status_invalid(self, app_context):
        with pytest.raises(ValueError):
            app_context.ideas.update_status(1, "invalid_status")

    def test_list_ideas(self, app_context):
        app_context.library.search_and_import("test", max_results=1)
        papers = app_context.db.list_papers(limit=10)
        arxiv_id = papers[0].arxiv_id
        app_context.ideas.generate_ideas(arxiv_id, num_ideas=2)
        ideas = app_context.ideas.list_ideas(arxiv_id)
        assert len(ideas) == 2
