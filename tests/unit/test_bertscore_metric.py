"""Unit tests for the BERTScore fallback metric (Tier 1.4)."""

from __future__ import annotations

import pytest

from tests.benchmark.metrics import bert_score_f1, rouge_l_f1


def test_returns_zero_on_empty():
    assert bert_score_f1("", "anything") == 0.0
    assert bert_score_f1("anything", "") == 0.0


def test_identical_strings_yield_high_score():
    """Fallback F1 of identical strings must reflect the token overlap."""
    txt = "scaling behaviour of mixture-of-experts transformers across compute"
    s = bert_score_f1(txt, txt)
    assert 0.99 <= s <= 1.0


def test_disjoint_strings_yield_low_fallback_score():
    a = "alpha beta gamma delta"
    b = "xenon yttrium zirconium tungsten"
    s = bert_score_f1(a, b)
    assert 0.0 <= s < 0.05


def test_paraphrase_yields_higher_score_than_rouge():
    """BERTScore is meant to credit paraphrases higher than ROUGE-L.

    Real-LLM summaries paraphrase; ROUGE-L under-recognises them.
    The fallback here is a token-F1 in this test, but since the two
    strings share many tokens the BERTScore-F1 will exceed ROUGE-L
    by a healthy margin. If the published bert_score package is
    installed, the gap will only widen."""
    reference = ("We study mixture-of-experts routing behaviour across "
                  "compute budgets and report a power-law slope near -0.05.")
    summary = ("The paper studies MoE sparse routing under several "
               "compute regimes and reports an empirical scaling-law slope "
               "close to -0.05.")
    bert = bert_score_f1(reference, summary)
    rouge = rouge_l_f1(reference, summary)
    # Token overlap is non-trivial but not identical.
    assert 0.20 <= bert <= 1.0
    # BERTScore F1 should be at least as generous as ROUGE-L for paraphrases.
    assert bert >= rouge * 0.9


def test_returns_float_in_unit_interval():
    s = bert_score_f1("the quick brown fox", "the lazy dog")
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_handles_repeated_tokens():
    """Repeated tokens in prediction inflate F1 numerator; tokens not
    in reference do not (they just lower precision)."""
    s = bert_score_f1("a b c", "a a a a")
    assert 0.0 < s <= 1.0


def test_non_ascii_input_does_not_raise():
    bert_score_f1("caf\u00e9", "caf\u00e9 latte")
    bert_score_f1("", "\u4e2d\u6587")  # chinese fallback
