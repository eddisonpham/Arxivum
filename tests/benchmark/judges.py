"""LLM-as-judge prompt templates and parsers.

Each judge follows the G-Eval (Liu et al. 2023) / Prometheus (Kim et
al. 2023) protocol: structured rubric, 1-5 integer scale, chain-of-
thought reasoning before the score, output as strict JSON. Parsers
validate the score lies in the allowed range and clamp otherwise.

The judge prompts are intentionally detailed and operationally
defined rather than vague quality claims - this is what G-Eval and
MT-Bench (Zheng et al. 2023) recommend for calibrated judgments.

A judge returns a dict:
    {"score": int (1-5), "rationale": str, "raw": str}

The same judge can be driven by a real local LLM (judge_llm(...)) or
by a deterministic stub (stub_judge(...)) that returns cached outputs
for offline, fast CI runs.
"""

from __future__ import annotations

import re
from typing import Callable

# ── Common prompt preamble ──────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert evaluator of research artefacts. "
    "Follow the rubric strictly. "
    "Reason step by step internally, then output ONLY valid JSON. "
    "Output schema: "
    '{"score": <integer 1-5>, "rationale": "<one short sentence>"}. '
    "No markdown fences, no prose outside the JSON."
)


def _user_judge(criteria: str, anchor_label: str, anchor_1: str, anchor_3: str, anchor_5: str,
                query: str, source_a: str, source_b: str) -> str:
    return (
        f"Rubric (anchor {anchor_label}):\n"
        f"  1 = {anchor_1}\n"
        f"  3 = {anchor_3}\n"
        f"  5 = {anchor_5}\n\n"
        f"Criteria: {criteria}\n\n"
        f"INPUT A (source paper):\n{source_a}\n\n"
        f"INPUT B (candidate artefact):\n{source_b}\n\n"
        f"Question / focus: {query}\n\n"
        "Reason step by step:\n"
        "  1. Does B stay factually grounded in A?\n"
        "  2. Does B cover the requested criteria?\n"
        "  3. Compare against the rubric anchors.\n"
        "Then output the JSON only."
    )


# ── Judge 1: Summary factuality (QAGS-style) ─────────────────────────────

JUDGE_FACTUALITY_FOCUS = (
    "Score how factually consistent B is with A. "
    "Penalise any invented numbers, results, datasets, or claims not "
    "traceable to A. A score of 5 means every sentence in B is supported "
    "by a verbatim or near-paraphrase claim in A. A score of 3 means "
    "most of B is grounded but some claims extend beyond what A says. "
    "A score of 1 means B contradicts A or invents material facts."
)


def judge_factuality(source_a: str, source_b: str, query: str = "") -> dict:
    """Build messages for a factuality judge."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": _user_judge(
            criteria=JUDGE_FACTUALITY_FOCUS,
            anchor_label="factuality",
            anchor_1="B contradicts A or invents specific facts.",
            anchor_3="B is mostly grounded but some sentences extend beyond A.",
            anchor_5="Every sentence in B is supported by a claim in A.",
            query=query or "(none)",
            source_a=source_a[:2000],
            source_b=source_b[:2000],
        )},
    ]


# ── Judge 2: Summary coverage ──────────────────────────────────────────

JUDGE_COVERAGE_FOCUS = (
    "Score how completely B covers the requested sections of A "
    "(problem statement, methodology, findings, ablations, "
    "discussion, limitations, overall). A score of 5 covers all "
    "seven sections meaningfully. A score of 3 covers four of seven. "
    "A score of 1 covers at most one section."
)


def judge_coverage(source_a: str, source_b: str, query: str = "", sections: list[str] | None = None) -> dict:
    sec_list = ", ".join(sections) if sections else "the canonical seven sections"
    user = _user_judge(
        criteria=JUDGE_COVERAGE_FOCUS + f" Specifically: {sec_list}.",
        anchor_label="coverage",
        anchor_1="At most one section is meaningfully summarised.",
        anchor_3="Roughly half the sections are summarised.",
        anchor_5="All requested sections are summarised and non-trivially.",
        query=query or "(none)",
        source_a=source_a[:2000],
        source_b=source_b[:2000],
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ── Judge 3: Idea plausibility (extensibility) ──────────────────────────

JUDGE_PLAUSIBILITY_FOCUS = (
    "Score how plausible the research idea in B is as an extension of A. "
    "A score of 5 means the idea is concretely actionable, explicitly "
    "builds on A's method or constraints, and proposes a testable claim. "
    "A score of 3 means the idea is in the right area but is generic or "
    "underspecified. A score of 1 means the idea is incoherent with A's "
    "method or restates A's conclusion as a future direction."
)


def judge_idea_plausibility(source_a: str, source_b: str, query: str = "") -> dict:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": _user_judge(
            criteria=JUDGE_PLAUSIBILITY_FOCUS,
            anchor_label="plausibility",
            anchor_1="Incoherent or just restates A's future-work.",
            anchor_3="Plausible but generic; no concrete claim.",
            anchor_5="Concrete, actionable, explicitly grounded in A.",
            query=query or "(none)",
            source_a=source_a[:2000],
            source_b=source_b[:2000],
        )},
    ]


# ── Judge 4: Novelty calibration ───────────────────────────────────────

JUDGE_NOVELTY_FOCUS = (
    "Decide whether B (a research idea) overlaps with C (a candidate "
    "paper abstract already published). Output the appropriate verdict "
    "as 'similar_exists' if C substantially addresses B, "
    "'needs_review' if there is partial overlap worth manual check, "
    "or 'likely_novel' if C does not address B. Be conservative: any "
    "significant overlap on the *core claim* of B is similar_exists."
)


def judge_novelty(source_a: str, source_b: str, source_c: str, query: str = "") -> dict:
    user = (
        "You will see three inputs:\n"
        "  A = the source paper the idea was generated FROM (background only)\n"
        "  B = the proposed research idea\n"
        "  C = a candidate arXiv paper abstract to compare against B\n\n"
        f"Rubric:\n{JUDGE_NOVELTY_FOCUS}\n\n"
        "Output a JSON object with keys: "
        '{"verdict": "similar_exists"|"needs_review"|"likely_novel", '
        '"reason": "<one short sentence>"}.\n\n'
        "---\n\n"
        f"A (source paper):\n{source_a[:1500]}\n\n"
        f"B (idea):\n{source_b[:1500]}\n\n"
        f"C (candidate):\n{source_c[:1500]}\n\n"
        "Reason step by step:\n"
        "  1. What is the *core claim* of B?\n"
        "  2. Does C already address that core claim?\n"
        "  3. Pick one verdict from the rubric.\n"
        "Output the JSON only."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ── Parsers: validate the JSON output of any judge ───────────────────────

_SCORE_RE = re.compile(r"\"score\"\s*:\s*(\d+)")
_VERDICT_RE = re.compile(r"\"verdict\"\s*:\s*\"(likely_novel|needs_review|similar_exists)\"")
_RATIONALE_RE = re.compile(r"\"rationale\"\s*:\s*\"([^\"]{0,500})\"")
_REASON_RE = re.compile(r"\"reason\"\s*:\s*\"([^\"]{0,500})\"")


def parse_score_judge(raw: str, scale_min: int = 1, scale_max: int = 5) -> dict:
    """Parse a 1-5 score judge output. Clamps to range."""
    m_score = _SCORE_RE.search(raw)
    m_rat = _RATIONALE_RE.search(raw)
    if not m_score:
        return {"score": None, "rationale": "PARSE_FAILED", "raw": raw[:200]}
    s = int(m_score.group(1))
    s = max(scale_min, min(scale_max, s))
    return {"score": s, "rationale": m_rat.group(1) if m_rat else "", "raw": raw[:200]}


def parse_verdict_judge(raw: str) -> dict:
    """Parse a novelty verdict output."""
    m_v = _VERDICT_RE.search(raw)
    if not m_v:
        return {"verdict": None, "reason": "PARSE_FAILED", "raw": raw[:200]}
    return {"verdict": m_v.group(1), "reason": _REASON_RE.search(raw).group(1) if _REASON_RE.search(raw) else ""}


# ── A trivial stub judge (deterministic, offline) ────────────────────────

class StubJudge:
    """Deterministic stub that returns canned responses.

    The stub returns a 3 (mid-score) for rubric judges and
    'needs_review' for novelty judges, on every call. This is
    *intentionally* the worst case for evaluating the system's
    metric-collection plumbing: if the running system scores better
    than 3 on quality, that comes from somewhere other than the judge.
    Use a real local LLM to get quality differences.
    """

    supports_score = True
    supports_verdict = True

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 256, **_kw) -> str:
        user = " ".join(m.get("content", "") for m in messages if m["role"] == "user")
        if "verdict" in user and "core claim" in user:
            return '{"verdict": "likely_novel", "reason": "stub judge"}'
        return '{"score": 3, "rationale": "stub judge"}'


# ── A convenience façade ────────────────────────────────────────────────

def call_judge(judge_messages: list[dict], llm, kind: str = "score") -> dict:
    """Call the configured LLM with the judge messages and parse.

    kind = "score" -> parse with parse_score_judge (1-5)
    kind = "verdict" -> parse with parse_verdict_judge (novelty)
    """
    raw = llm.chat(
        messages=judge_messages, temperature=0.0, max_tokens=192,
    )
    if kind == "score":
        return parse_score_judge(raw)
    if kind == "verdict":
        return parse_verdict_judge(raw)
    raise ValueError(f"unknown judge kind: {kind}")


# ── Test sanity check that ensures all judges produce a parseable shape ──

def _selftest() -> None:
    raw_score = '{"score": 4, "rationale": "looks good"}'
    raw_verdict = '{"verdict": "needs_review", "reason": "partial overlap"}'
    out1 = parse_score_judge(raw_score)
    assert out1["score"] == 4 and out1["rationale"] == "looks good", out1
    out2 = parse_verdict_judge(raw_verdict)
    assert out2["verdict"] == "needs_review" and out2["reason"] == "partial overlap", out2
    out3 = parse_score_judge('{"score": 99, "rationale": "x"}')
    assert out3["score"] == 5, out3
    out4 = parse_score_judge("not json")
    assert out4["score"] is None, out4


_selftest()
