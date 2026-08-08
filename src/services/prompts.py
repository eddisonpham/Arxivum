"""Centralised LLM prompt templates.

Keeping prompts in one place makes them easy to tune and test.  Each
function returns a list of chat messages ready for ``LLM.chat()``.
"""

from __future__ import annotations

import json
from typing import Any

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


def novelty_messages(
    idea_text: str,
    candidate_title: str,
    candidate_abstract: str,
) -> list[dict]:
    """Build chat messages for novelty comparison.

    Asks for an explicit, calibrated ``confidence`` (0.0 .. 1.0) so
    that low-confidence binary verdicts can be downgraded to
    ``needs_review`` by the service layer. See
    :data:`src.services.novelty.LOW_CONFIDENCE_THRESHOLD`.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a research novelty assessor. Compare a proposed "
                "research idea with a candidate paper and determine whether "
                "the candidate already addresses the idea. "
                "Output ONLY valid JSON. "
                "After your verdict, also output a JSON field 'confidence' "
                "(0.0..1.0) that reflects how strongly the candidate paper's "
                "abstract overlaps with the proposed idea. Be calibrated: "
                "0.9+ only when the candidate is unambiguously equivalent; "
                "0.5-0.7 when overlap is partial; <0.4 when the abstracts "
                "merely share vocabulary."
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
                f'  "confidence": <float 0.0..1.0>,\n'
                f'  "reason": "short explanation"\n\n'
                f"Output the JSON object only:"
            ),
        },
    ]


def extract_messages(title: str, abstract: str, arxiv_id: str) -> list[dict]:
    """Build chat messages for structured extraction (SPECTER/OSLO-style).

    Asks the LLM to emit a single JSON object conforming to the
    canonical arXivum schema. The schema is *operationally defined*
    so that the output is comparable across papers and round-trippable
    to BibTeX/LaTeX.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a scientific paper parser. Extract a structured "
                "bibliographic schema for the given paper. Output ONLY "
                "valid JSON — no markdown, no commentary, no extra keys."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
                f"arxiv_id: {arxiv_id}\n\n"
                "Emit a JSON object with EXACTLY these keys:\n"
                '  "method": "<one concise sentence naming the method>",\n'
                '  "datasets": [<list of dataset names, may be empty>],\n'
                '  "baselines": [<list of baseline system names, may be empty>],\n'
                '  "headline_metric": {"name": "<metric name>", "value": "<value>", "split": "<eval split/benchmark name>"},\n'
                '  "contribution": "<one-sentence contribution claim>",\n'
                '  "limitations": [<list of admitted limitations, may be empty>],\n'
                '  "domain": "<sub-field, e.g. deep-learning scaling, retrieval augmentation, summarisation>",\n'
                '  "bibcode": "arXiv:<id>"\n\n'
                "If a field is not stated or cannot be determined, use empty "
                "string, empty list, or empty dict as appropriate.\n"
                "Output the JSON object only:"
            ),
        },
    ]


def extract_json(raw: str) -> Any:
    """Best-effort extraction of JSON from an LLM response.

    LLMs sometimes wrap JSON in markdown fences or add trailing text.
    This function tries, in order:
      1. Direct ``json.loads``.
      2. Extract the first ``{...}`` or ``[...]`` block.
    Returns the parsed object/array, or raises ``ValueError`` on failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    bracket_start = text.find("[")
    candidates: list[tuple[int, str, str]] = []
    if brace_start != -1:
        candidates.append((brace_start, "{", "}"))
    if bracket_start != -1:
        candidates.append((bracket_start, "[", "]"))
    candidates.sort(key=lambda c: c[0])

    for start, opener, closer in candidates:
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
