"""Centralised LLM prompt templates.

Keeping prompts in one place makes them easy to tune and test.  Each
function returns a list of chat messages ready for ``LLM.chat()``.
"""

from __future__ import annotations

import json
from typing import Any

# ── Summarization ────────────────────────────────────────────────────────────

SUMMARY_SECTIONS = [
    "problem_statement",
    "methodology",
    "findings",
    "ablations",
    "discussion",
    "limitations",
    "overall",
]


def summary_messages(title: str, abstract: str, sections: list[str]) -> list[dict]:
    """Build chat messages for structured summarization.

    Asks the LLM to output a JSON object with keys from ``sections``.
    """
    requested = ", ".join(sections)
    return [
        {
            "role": "system",
            "content": (
                "You are a precise research assistant. Summarise the given "
                "paper concisely. Output ONLY valid JSON — no markdown, no "
                "commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Title: {title}\n"
                f"Abstract: {abstract}\n\n"
                f"Generate a JSON object with these keys: {requested}.\n"
                f"Each value must be a concise string under 100 words.\n"
                f"Use 'N/A' if a section cannot be determined from the abstract.\n\n"
                f"Output the JSON object only:"
            ),
        },
    ]


# ── Constraint extraction ────────────────────────────────────────────────────

def constraint_messages(title: str, abstract: str) -> list[dict]:
    """Extract assumptions, biases, limitations, domain, key method."""
    return [
        {
            "role": "system",
            "content": (
                "You are a research analyst. Extract the constraints of the "
                "given paper. Output ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Title: {title}\n"
                f"Abstract: {abstract}\n\n"
                f"Extract a JSON object with keys:\n"
                f'  "assumptions": [list of strings],\n'
                f'  "inductive_biases": [list of strings],\n'
                f'  "limitations": [list of strings],\n'
                f'  "domain": "string",\n'
                f'  "key_method": "string"\n\n'
                f"Output the JSON object only:"
            ),
        },
    ]


# ── Idea generation ─────────────────────────────────────────────────────────

def idea_messages(
    title: str,
    abstract: str,
    constraints: dict[str, Any] | None,
    num_ideas: int,
    focus_area: str,
) -> list[dict]:
    """Build chat messages for idea generation."""
    constraints_str = json.dumps(constraints, indent=2) if constraints else "{}"
    return [
        {
            "role": "system",
            "content": (
                "You are a creative research scientist. Given a paper, "
                "generate novel research ideas that build on or invert its "
                "constraints and biases. Output ONLY a valid JSON array."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
                f"Constraints: {constraints_str}\n\n"
                f"Generate {num_ideas} {focus_area} research ideas.\n"
                f"Output a JSON array where each element has keys:\n"
                f'  "title": "short idea title",\n'
                f'  "summary": "one-sentence summary",\n'
                f'  "extension": "how it extends or contradicts the source",\n'
                f'  "next_steps": ["2-3 concrete steps"],\n'
                f'  "search_queries": ["2-3 short arXiv queries for novelty check"]\n\n'
                f"Output the JSON array only:"
            ),
        },
    ]


# ── Novelty verification ─────────────────────────────────────────────────────

def novelty_messages(
    idea_text: str,
    candidate_title: str,
    candidate_abstract: str,
) -> list[dict]:
    """Build chat messages for novelty comparison."""
    return [
        {
            "role": "system",
            "content": (
                "You are a research novelty assessor. Compare a proposed "
                "research idea with a candidate paper and determine whether "
                "the candidate already addresses the idea. Output ONLY valid "
                "JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Proposed idea: {idea_text}\n\n"
                f"Candidate paper title: {candidate_title}\n"
                f"Candidate paper abstract: {candidate_abstract}\n\n"
                f'Output a JSON object with keys:\n'
                f'  "verdict": "likely_novel" | "needs_review" | "similar_exists",\n'
                f'  "reason": "short explanation"\n\n'
                f"Output the JSON object only:"
            ),
        },
    ]


# ── JSON extraction helper ───────────────────────────────────────────────────

def extract_json(raw: str) -> Any:
    """Best-effort extraction of JSON from an LLM response.

    LLMs sometimes wrap JSON in markdown fences or add trailing text.
    This function tries, in order:
      1. Direct ``json.loads``.
      2. Extract the first ``{...}`` or ``[...]`` block.
    Returns the parsed object/array, or raises ``ValueError`` on failure.
    """
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first balanced JSON object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not extract JSON from LLM output: {raw[:200]!r}")
