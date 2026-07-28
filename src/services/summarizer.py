"""Summarizer service.

Generates structured paper summaries (problem statement, methodology,
findings, ablations, discussion, limitations, overall) via the local LLM
and caches them in the ``summaries`` table.  Generated sections are also
indexed as ChromaDB chunks for RAG retrieval.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.db.models import ActivityRow, Database, SummaryRow
from src.inference.manager import ModelManager
from src.services.library import LibraryService
from src.services.prompts import SUMMARY_SECTIONS, extract_json, summary_messages

logger = logging.getLogger(__name__)


class SummarizerService:
    """Generate and cache structured paper summaries."""

    def __init__(self, db: Database, models: ModelManager, library: LibraryService) -> None:
        self.db = db
        self.models = models
        self.library = library

    def summarize(
        self,
        arxiv_id: str,
        sections: Sequence[str] | None = None,
        force: bool = False,
    ) -> dict[str, str]:
        """Generate or retrieve a structured summary for a paper.

        Args:
            arxiv_id: The paper's arXiv ID.
            sections: Which sections to generate.  Defaults to all.
            force: If True, regenerate even if cached.

        Returns:
            A mapping ``{section: content}``.
        """
        requested = list(sections) if sections else list(SUMMARY_SECTIONS)

        # Return cached if available and not forced.
        if not force:
            cached: dict[str, str] = {}
            for section in requested:
                row = self.db.get_summary(arxiv_id, section)
                if row:
                    cached[section] = row.content
            if len(cached) == len(requested):
                return cached

        paper = self.db.get_paper(arxiv_id)
        if not paper:
            raise ValueError(f"Paper {arxiv_id} not found in library")

        log_id = self.db.log_activity(ActivityRow(
            id=None, action_type="summarize", arxiv_id=arxiv_id,
            status="started", metadata_json={"sections": requested, "force": force},
        ))
        try:
            messages = summary_messages(paper.title, paper.abstract, requested)
            raw = self.models.llm.chat(
                messages=messages, temperature=0.3, max_tokens=1024,
            )
            parsed = extract_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"LLM returned non-object JSON: {type(parsed)}")

            model_name = getattr(self.models.llm, "model_path", "stub")
            result: dict[str, str] = {}
            for section in requested:
                content = str(parsed.get(section, "N/A")).strip()
                if not content or content == "N/A":
                    content = "N/A"
                self.db.upsert_summary(SummaryRow(
                    id=None, arxiv_id=arxiv_id, section=section,
                    content=content, model_used=model_name,
                ))
                # Index the summary section as a ChromaDB chunk for RAG.
                if content != "N/A":
                    try:
                        self.library.index_summary_section(arxiv_id, section, content)
                    except Exception:
                        logger.debug("Failed to index summary chunk for %s#%s",
                                     arxiv_id, section, exc_info=True)
                result[section] = content

            self.db.update_activity_status(log_id, "completed")
            return result
        except Exception:
            self.db.update_activity_status(log_id, "failed")
            raise

    def get_cached(self, arxiv_id: str) -> dict[str, str]:
        """Return all cached summary sections for a paper (no LLM call)."""
        rows = self.db.get_summaries(arxiv_id)
        return {r.section: r.content for r in rows}
