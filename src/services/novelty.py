"""Novelty re-verification service.

For each generated idea, checks whether similar work already exists:

  1. **Local RAG pre-check**: query the local library for similar papers.
  2. **External arXiv check**: search arXiv using the idea's
     ``search_queries``.
  3. **LLM judgment**: for each candidate, the LLM decides if the idea
     is already addressed.

Verdict: ``likely_novel``, ``needs_review``, or ``similar_exists``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.clients.arxiv_client import ArxivClient
from src.db.models import Database, NoveltyCheckRow
from src.inference.manager import ModelManager
from src.services.prompts import extract_json, novelty_messages
from src.utils import track_activity

logger = logging.getLogger(__name__)

LOCAL_SIMILARITY_THRESHOLD = 0.85

class NoveltyService:
    """Verify the novelty of a generated research idea."""

    def __init__(
        self,
        db: Database,
        models: ModelManager,
        arxiv_client: ArxivClient,
        library_query_fn: Callable[..., Any] | None = None,
    ) -> None:
        """
        Args:
            db: Database handle.
            models: Model manager (for embedding + LLM).
            arxiv_client: arXiv client for external checks.
            library_query_fn: Optional callable(query, top_k) -> list of
                QueryResultItem for the local RAG pre-check.  If None,
                the local check is skipped.
        """
        self.db = db
        self.models = models
        self.arxiv = arxiv_client
        self._library_query = library_query_fn

    def verify_novelty(
        self,
        idea_id: int,
        search_query: str | None = None,
    ) -> dict:
        """Run novelty re-verification on an idea.

        Returns a dict with ``idea_id``, ``verdict``, ``notes``,
        ``similar_arxiv_ids``, ``query_terms``.
        """
        idea = self.db.get_idea(idea_id)
        if not idea:
            raise ValueError(f"Idea {idea_id} not found")

        with track_activity(
            self.db, "novelty", arxiv_id=idea.arxiv_id,
            metadata_json={"idea_id": idea_id},
        ):
            query_terms = list(idea.search_queries) if idea.search_queries else []
            if search_query:
                query_terms.insert(0, search_query)
            if not query_terms:
                query_terms = [idea.idea_text[:100]]

            similar_ids: list[str] = []
            verdict = "likely_novel"
            notes_parts: list[str] = []

            local_matches = self._local_check(idea.idea_text, idea.arxiv_id)
            if local_matches:
                similar_ids.extend(m["arxiv_id"] for m in local_matches)
                verdict = "needs_review"
                notes_parts.append(
                    f"Local library has {len(local_matches)} similar paper(s)."
                )

            external_candidates = self._external_check(query_terms, idea.arxiv_id)
            if external_candidates:
                for cand in external_candidates:
                    similar_ids.append(cand["arxiv_id"])
                    llm_verdict = self._llm_judge(
                        idea.idea_text, cand["title"], cand["abstract"],
                    )
                    if llm_verdict == "similar_exists":
                        verdict = "similar_exists"
                        notes_parts.append(
                            f"LLM found similar existing work: {cand['title'][:80]}"
                        )
                        break
                    elif llm_verdict == "needs_review" and verdict != "similar_exists":
                        verdict = "needs_review"
                        notes_parts.append(
                            f"LLM flagged possible overlap: {cand['title'][:80]}"
                        )

            if not similar_ids and verdict == "likely_novel":
                notes_parts.append("No similar work found in local library or arXiv.")

            unique_similar = list(dict.fromkeys(similar_ids))
            notes = " ".join(notes_parts) if notes_parts else "No notes."
            check_id = self.db.add_novelty_check(NoveltyCheckRow(
                id=None, idea_id=idea_id,
                query_terms=query_terms,
                similar_arxiv_ids=unique_similar,
                verdict=verdict, notes=notes,
            ))

            return {
                "check_id": check_id,
                "idea_id": idea_id,
                "verdict": verdict,
                "notes": notes,
                "similar_arxiv_ids": unique_similar,
                "query_terms": query_terms,
            }

    def _local_check(
        self, idea_text: str, source_arxiv_id: str
    ) -> list[dict]:
        """Query the local library for papers similar to the idea."""
        if self._library_query is None:
            return []
        try:
            results = self._library_query(idea_text, top_k=5)
        except Exception as exc:
            logger.warning("Local novelty check failed: %s", exc)
            return []
        matches: list[dict] = []
        for r in results:
            if r.arxiv_id == source_arxiv_id:
                continue
            if r.score >= LOCAL_SIMILARITY_THRESHOLD:
                matches.append({
                    "arxiv_id": r.arxiv_id,
                    "title": r.title,
                    "score": r.score,
                })
        return matches

    def _external_check(
        self, query_terms: list[str], source_arxiv_id: str
    ) -> list[dict]:
        """Search arXiv with the idea's query terms."""
        candidates: list[dict] = []
        seen_ids: set[str] = set()
        for term in query_terms[:3]:
            try:
                papers = self.arxiv.search(
                    query=term, max_results=5, sort_by="relevance",
                )
            except Exception as exc:
                logger.warning("arXiv novelty search failed for %r: %s", term, exc)
                continue
            for p in papers:
                if p.arxiv_id == source_arxiv_id:
                    continue
                if p.arxiv_id in seen_ids:
                    continue
                seen_ids.add(p.arxiv_id)
                candidates.append({
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "abstract": p.abstract,
                })
        return candidates

    def _llm_judge(
        self, idea_text: str, cand_title: str, cand_abstract: str
    ) -> str:
        """Ask the LLM to compare the idea with a candidate paper."""
        messages = novelty_messages(idea_text, cand_title, cand_abstract)
        try:
            raw = self.models.llm.chat(
                messages=messages, temperature=0.2, max_tokens=256,
            )
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                v = parsed.get("verdict", "needs_review")
                if v in ("likely_novel", "needs_review", "similar_exists"):
                    return v
        except Exception as exc:
            logger.warning("LLM novelty judgment failed: %s", exc)
        return "needs_review"
