"""Unit tests for IdeaService post-filter (Tier 1.3)."""

from __future__ import annotations

import json

import pytest

from src.services.ideas import (
    IDEA_TITLE_JACCARD_LIMIT,
    IdeaService,
    _idea_contrarian_signal,
    _idea_text,
    _post_filter_ideas,
)
from src.services.prompts import CONTRARIAN_KEYWORDS


def test_contrarian_signal_positive():
    item = {
        "title": "Invert the routing entropy objective",
        "summary": "Replace the regulariser with an inverse.",
        "extension": "Substitute the loss with the contrary objective.",
        "next_steps": ["ablate at 100k tokens"],
    }
    assert _idea_contrarian_signal(item) is True


def test_contrarian_signal_negative_no_keyword():
    item = {
        "title": "Conduct a similar study across datasets",
        "summary": "Conduct a similar empirical study.",
        "extension": "Builds on the previous paper.",
        "next_steps": ["Collect more data"],
        "search_queries": ["paper follow-up"],
    }
    assert _idea_contrarian_signal(item) is False


def test_contrarian_signal_in_search_queries_only():
    item = {
        "title": "A follow-up study",
        "summary": "Re-runs the experiment with more seeds.",
        "extension": "Same model, more seeds.",
        "next_steps": ["increase seeds"],
        "search_queries": ["contrarian replication study"],
    }
    assert _idea_contrarian_signal(item) is True


def test_jaccard_limit_constant():
    assert IDEA_TITLE_JACCARD_LIMIT == 0.6


def test_post_filter_drops_parrot_phrases():
    source = (
        "Scaling laws for mixture-of-experts routing under sparse activation "
        "We study scaling behaviour of mixture-of-experts transformer variants"
    )
    candidates = [
        {  # parrot: shares most tokens
            "title": "Scaling laws for mixture-of-experts routing",
            "summary": "We study scaling behaviour of mixture-of-experts transformers",
            "extension": "Builds on the original.",
            "next_steps": ["evaluate at more scales"],
            "search_queries": ["mixture-of-experts scaling"],
        },
        {  # contrarian
            "title": "Remove the regulariser and retrain",
            "summary": "Replace the anti-collapse objective with a contrary one.",
            "extension": "Inverts the prior constraint.",
            "next_steps": ["train without the penalty"],
        },
        {  # no contrarian, not a parrot
            "title": "Apply the recipe to retrieval.",
            "summary": "Use the pattern in retrieval pipelines.",
            "extension": "Builds on the original.",
            "next_steps": ["port to retrieval"],
            "search_queries": ["apply to retrieval"],
        },
    ]
    out = _post_filter_ideas(candidates, source)
    titles = [c.get("title", "") for c in out]
    assert "Scaling laws for mixture-of-experts routing" not in titles
    assert "Remove the regulariser and retrain" in titles
    assert "Apply the recipe to retrieval." not in titles
    # exactly one kept
    assert len(out) == 1


def test_post_filter_preserves_order():
    candidates = [
        {"title": "Z: invert", "summary": "substitute objective", "extension": "swap"},
        {"title": "A: remove", "summary": "drop the penalty", "extension": "contrary"},
    ]
    out = _post_filter_ideas(candidates, "completely unrelated source text")
    assert [c["title"] for c in out] == ["Z: invert", "A: remove"]


def test_idea_text_helper():
    assert _idea_text({"summary": "s"}) == "s"
    assert _idea_text({"title": "t"}) == "t"
    assert _idea_text({"summary": "s", "title": "t"}) == "s"
    assert _idea_text({}) == ""


def test_end_to_end_filter_then_generate():
    """Inject a StubLLM that returns only parrot ideas, then confirm
    the post-filter rejects them and the retry path converges."""
    from src.inference.llm import StubLLM
    from src.db.models import Database, PaperRow
    from src.inference.manager import ModelManager

    db = Database(":memory:")
    db.upsert_paper(PaperRow(arxiv_id="2401.00001", title="T", authors=["a"],
                              abstract="abstract"))
    mm = ModelManager()
    parrot = json.dumps([{
        "title": "Conduct a similar study",
        "summary": "Conduct a similar empirical study.",
        "extension": "Builds on the original.",
        "next_steps": ["evaluate"],
        "search_queries": ["follow-up study"],
    }])
    mm.set_llm(StubLLM(default=parrot))

    svc = IdeaService(db, mm)
    # Pure parrot (no contrarian signal) -> filter drops -> 0 ideas
    ideas = svc.generate_ideas("2401.00001", num_ideas=2)
    assert ideas == [], (
        "pure-parrot LLM output should not produce any ideas after "
        "the post-filter"
    )
    # Now mix in a contrarian one
    mixed = json.dumps([
        {"title": "Conduct a similar study", "summary": "Replicate.",
         "extension": "Same approach.", "next_steps": ["replicate"]},
        {"title": "Invert the constraint", "summary": "Contrary approach.",
         "extension": "Substitute objective.", "next_steps": ["substitute"]},
    ])
    mm.set_llm(StubLLM(default=mixed))
    ideas = svc.generate_ideas("2401.00001", num_ideas=2)
    assert len(ideas) == 1
    assert "Invert" in ideas[0]["title"]


def test_contrarian_keyword_set_documented():
    """The keyword list must be a tuple to prevent accidental mutation
    and contain the most important signals from the rubric."""
    assert isinstance(CONTRARIAN_KEYWORDS, tuple)
    for required in ("invert", "remove", "without", "contrary", "substit"):
        assert required in CONTRARIAN_KEYWORDS, required
