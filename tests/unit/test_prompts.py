"""Unit tests for prompt templates and JSON extraction."""

import json
import pytest

from src.services.prompts import (
    SUMMARY_SECTIONS, constraint_messages, extract_json, idea_messages,
    novelty_messages, summary_messages,
)


class TestSummaryMessages:
    def test_returns_two_messages(self):
        msgs = summary_messages("Title", "Abstract", ["overall"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_contains_title_and_abstract(self):
        msgs = summary_messages("My Title", "My Abstract", ["overall"])
        assert "My Title" in msgs[1]["content"]
        assert "My Abstract" in msgs[1]["content"]

    def test_contains_sections(self):
        msgs = summary_messages("T", "A", ["methodology", "findings"])
        assert "methodology" in msgs[1]["content"]
        assert "findings" in msgs[1]["content"]

    def test_default_sections_list(self):
        assert "problem_statement" in SUMMARY_SECTIONS
        assert "methodology" in SUMMARY_SECTIONS
        assert "findings" in SUMMARY_SECTIONS
        assert "ablations" in SUMMARY_SECTIONS
        assert "overall" in SUMMARY_SECTIONS


class TestConstraintMessages:
    def test_returns_two_messages(self):
        msgs = constraint_messages("T", "A")
        assert len(msgs) == 2

    def test_contains_required_keys(self):
        msgs = constraint_messages("T", "A")
        content = msgs[1]["content"]
        assert "assumptions" in content
        assert "inductive_biases" in content
        assert "limitations" in content


class TestIdeaMessages:
    def test_contains_num_ideas(self):
        msgs = idea_messages("T", "A", None, 3, "methodological")
        assert "3" in msgs[1]["content"]

    def test_contains_focus_area(self):
        msgs = idea_messages("T", "A", None, 2, "theoretical")
        assert "theoretical" in msgs[1]["content"]

    def test_contains_constraints(self):
        constraints = {"assumptions": ["x"], "domain": "NLP"}
        msgs = idea_messages("T", "A", constraints, 1, "applied")
        assert "assumptions" in msgs[1]["content"]

    def test_contains_search_queries_key(self):
        msgs = idea_messages("T", "A", None, 1, "applied")
        assert "search_queries" in msgs[1]["content"]


class TestNoveltyMessages:
    def test_contains_idea_and_candidate(self):
        msgs = novelty_messages("My idea", "Cand title", "Cand abstract")
        content = msgs[1]["content"]
        assert "My idea" in content
        assert "Cand title" in content
        assert "Cand abstract" in content

    def test_contains_verdict_options(self):
        msgs = novelty_messages("idea", "t", "a")
        content = msgs[1]["content"]
        assert "likely_novel" in content
        assert "needs_review" in content
        assert "similar_exists" in content


class TestExtractJson:
    def test_direct_json_object(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_direct_json_array(self):
        result = extract_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_markdown_fenced_json(self):
        result = extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_markdown_fenced_no_language(self):
        result = extract_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        result = extract_json('Here is the JSON:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_nested_json(self):
        raw = '{"outer": {"inner": [1, 2]}, "x": true}'
        result = extract_json(raw)
        assert result["outer"]["inner"] == [1, 2]

    def test_json_array_with_surrounding_text(self):
        result = extract_json('Result: [{"a": 1}, {"b": 2}]')
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            extract_json("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            extract_json("")

    def test_json_with_escaped_quotes(self):
        raw = '{"text": "He said \\"hello\\""}'
        result = extract_json(raw)
        assert result["text"] == 'He said "hello"'

    def test_json_with_braces_in_strings(self):
        raw = '{"code": "function() { return {}; }"}'
        result = extract_json(raw)
        assert "function" in result["code"]
