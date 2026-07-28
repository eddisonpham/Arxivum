"""Idea generation service.

Extracts constraints (assumptions, biases, limitations) from a paper and
generates novel research ideas grounded in those constraints via the local
LLM.  Each idea is stored with its ``search_queries`` for later novelty
verification.
"""

from __future__ import annotations

import logging
from typing import Any

from src.db.models import ActivityRow, Database, IdeaRow
from src.inference.manager import ModelManager
from src.services.prompts import constraint_messages, extract_json, idea_messages

logger = logging.getLogger(__name__)

class IdeaService:
    """Generate novel research ideas from a paper."""

    def __init__(self, db: Database, models: ModelManager) -> None:
        self.db = db
        self.models = models

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

        log_id = self.db.log_activity(ActivityRow(
            id=None, action_type="idea", arxiv_id=arxiv_id, status="started",
            metadata_json={"num_ideas": num_ideas, "focus_area": focus_area},
        ))
        try:
            constraints = self._extract_constraints(paper.title, paper.abstract)

            messages = idea_messages(
                paper.title, paper.abstract, constraints, num_ideas, focus_area,
            )
            raw = self.models.llm.chat(
                messages=messages, temperature=0.7, max_tokens=1024,
            )
            parsed = extract_json(raw)
            if not isinstance(parsed, list):
                parsed = [parsed] if isinstance(parsed, dict) else []
            parsed = parsed[:num_ideas]

            model_name = getattr(self.models.llm, "model_path", "stub")
            ideas: list[dict] = []
            for item in parsed:
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

            self.db.update_activity_status(log_id, "completed")
            return ideas
        except Exception:
            self.db.update_activity_status(log_id, "failed")
            raise

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
