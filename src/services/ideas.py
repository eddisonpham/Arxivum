"""Idea generation service.

Extracts constraints (assumptions, biases, limitations) from a paper and
generates novel research ideas grounded in those constraints via the local
LLM.  Each idea is stored with its ``search_queries`` for later novelty
verification.

Tier-1.3 post-filter (IMPROVEMENT_PLAN): generated idea arrays are
juried through :func:`_post_filter_ideas` before they are persisted so
that:

  1. ideas whose Jaccard vs the source's title + abstract exceeds
     ``IDEA_TITLE_JACCARD_LIMIT`` are dropped as paraphrases;
  2. ideas that contain no contrarian keyword in
     (title, extension, next_steps, search_queries) are dropped as
     generic. The keyword list lives in
     :data:`src.services.prompts.CONTRARIAN_KEYWORDS`.

If the juried set is shorter than ``num_ideas``, we re-issue a single
prompt asking for ``remaining`` *distinct, contrarian* ideas and
append them until we hit the requested count or exhaust retries.
"""

from __future__ import annotations

import logging
from typing import Any

from src.db.models import Database, IdeaRow
from src.inference.manager import ModelManager
from src.services.prompts import (
    CONTRARIAN_KEYWORDS,
    constraint_messages,
    extract_json,
    idea_messages,
)
from src.utils import track_activity

logger = logging.getLogger(__name__)

IDEA_TITLE_JACCARD_LIMIT = 0.6
IDEA_GENERATE_RETRY_ROUNDS = 1


def _idea_contrarian_signal(item: dict[str, Any]) -> bool:
    """Return True if any of (title, summary, extension, next_steps,
    search_queries) contains a contrarian keyword."""
    fields = [
        str(item.get("title", "")),
        str(item.get("summary", "")),
        str(item.get("extension", "")),
    ]
    next_steps = item.get("next_steps") or []
    if isinstance(next_steps, list):
        fields.extend(str(s) for s in next_steps)
    queries = item.get("search_queries") or []
    if isinstance(queries, list):
        fields.extend(str(q) for q in queries)
    haystack = " ".join(fields).lower()
    return any(kw in haystack for kw in CONTRARIAN_KEYWORDS)


def _idea_text(item: dict[str, Any]) -> str:
    return str(item.get("summary") or item.get("title") or item.get("idea_text") or "")


def _post_filter_ideas(
    parsed: list[dict[str, Any]],
    source_text: str,
) -> list[dict[str, Any]]:
    """Drop ideas that are parroting-paraphrases of the source paper.

    A candidate is dropped if either:
      - its idea_text shares >= ``IDEA_TITLE_JACCARD_LIMIT`` Jaccard
        similarity on word level with the source's title+abstract, OR
      - it contains no contrarian keyword in any of its fields.

    Order is preserved so the highest-signal ideas surface first.
    """
    src_tokens = set(source_text.lower().split())
    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text_tokens = set(_idea_text(item).lower().split())
        if not text_tokens or not src_tokens:
            jacc = 0.0
        else:
            jacc = len(text_tokens & src_tokens) / len(text_tokens | src_tokens)
        if jacc >= IDEA_TITLE_JACCARD_LIMIT:
            logger.debug(
                "dropping parrot-ish idea (jacc %.2f): %s",
                jacc, _idea_text(item)[:60])
            continue
        if not _idea_contrarian_signal(item):
            logger.debug(
                "dropping idea without contrarian signal: %s",
                _idea_text(item)[:60])
            continue
        out.append(item)
    return out


class IdeaService:
    """Generate novel research ideas from a paper."""

    def __init__(self, db: Database, models: ModelManager) -> None:
        self.db = db
        self.models = models

    def _generate_and_jury(
        self,
        title: str,
        abstract: str,
        constraints: dict[str, Any] | None,
        num_ideas: int,
        focus_area: str,
        source_text: str,
    ) -> list[dict[str, Any]]:
        """Generate ``num_ideas`` candidate ideas, jury them, retry if
        short of the quota.  ``source_text`` is the joined
        ``title + abstract`` used for the parrot-Jaccard check."""
        juried: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        rounds = max(1, IDEA_GENERATE_RETRY_ROUNDS + 1)
        for round_idx in range(rounds):
            remaining = num_ideas - len(juried)
            if remaining <= 0:
                break
            n = remaining if round_idx == 0 else max(remaining * 2, 2)
            messages = idea_messages(
                title, abstract, constraints, n, focus_area,
            )
            try:
                raw = self.models.llm.chat(
                    messages=messages, temperature=0.7, max_tokens=1024,
                )
                parsed = extract_json(raw)
            except Exception as exc:
                logger.warning("idea generation round %d failed: %s",
                                round_idx, exc)
                parsed = []
            if not isinstance(parsed, list):
                parsed = [parsed] if isinstance(parsed, dict) else []
            filtered = _post_filter_ideas(parsed, source_text)
            for item in filtered:
                text = _idea_text(item).strip().lower()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                juried.append(item)
                if len(juried) >= num_ideas:
                    break
        return juried[:num_ideas]

    def _extract_constraints(self, title: str, abstract: str) -> dict[str, Any]:
        """Ask the LLM to extract assumptions, biases, limitations."""
        messages = constraint_messages(title, abstract)
        raw = self.models.llm.chat(
            messages=messages, temperature=0.2, max_tokens=512,
        )
        try:
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            logger.warning("Constraint extraction parse failed: %s", exc)
        return {
            "assumptions": [],
            "inductive_biases": [],
            "limitations": [],
            "domain": "unknown",
            "key_method": "unknown",
        }

    def generate_ideas(
        self,
        arxiv_id: str,
        num_ideas: int = 3,
        focus_area: str = "methodological",
    ) -> list[dict]:
        """Generate novel ideas for a paper.

        Args:
            arxiv_id: Source paper's arXiv ID.
            num_ideas: How many ideas to generate (max 5).
            focus_area: ``theoretical``, ``applied``, ``methodological``,
                or ``hybrid``.

        Returns:
            A list of idea dicts, each with ``id``, ``idea_text``,
            ``constraints_used``, ``search_queries``, ``status``.
        """
        num_ideas = min(max(num_ideas, 1), 5)
        paper = self.db.get_paper(arxiv_id)
        if not paper:
            raise ValueError(f"Paper {arxiv_id} not found in library")

        with track_activity(
            self.db, "idea", arxiv_id=arxiv_id,
            metadata_json={"num_ideas": num_ideas, "focus_area": focus_area},
        ):
            constraints = self._extract_constraints(paper.title, paper.abstract)
            source_text = f"{paper.title} {paper.abstract}"

            juried = self._generate_and_jury(
                paper.title, paper.abstract, constraints,
                num_ideas=num_ideas, focus_area=focus_area,
                source_text=source_text,
            )

            ideas: list[dict] = []
            model_name = getattr(self.models.llm, "model_path", "stub")
            for item in juried:
                if not isinstance(item, dict):
                    continue
                idea_text = item.get("summary") or item.get("title") or "(untitled idea)"
                search_queries = item.get("search_queries", [])
                if not isinstance(search_queries, list):
                    search_queries = [str(search_queries)]
                idea_id = self.db.add_idea(IdeaRow(
                    id=None,
                    arxiv_id=arxiv_id,
                    idea_text=idea_text,
                    constraints_used=constraints,
                    generated_with_model=model_name,
                    status="pending",
                    search_queries=[str(q) for q in search_queries],
                ))
                ideas.append({
                    "id": idea_id,
                    "arxiv_id": arxiv_id,
                    "idea_text": idea_text,
                    "title": item.get("title", ""),
                    "extension": item.get("extension", ""),
                    "next_steps": item.get("next_steps", []),
                    "search_queries": [str(q) for q in search_queries],
                    "status": "pending",
                })

            return ideas

    def update_status(self, idea_id: int, status: str) -> bool:
        """Approve or reject an idea (human-in-the-loop)."""
        if status not in ("pending", "approved", "rejected"):
            raise ValueError(f"Invalid status: {status}")
        return self.db.update_idea_status(idea_id, status)

    def list_ideas(self, arxiv_id: str) -> list[dict]:
        """List all ideas for a paper."""
        rows = self.db.list_ideas(arxiv_id)
        return [
            {
                "id": r.id,
                "arxiv_id": r.arxiv_id,
                "idea_text": r.idea_text,
                "constraints_used": r.constraints_used,
                "status": r.status,
                "search_queries": r.search_queries,
                "created_at": r.created_at,
            }
            for r in rows
        ]
