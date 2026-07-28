"""arXiv API client.

Wraps the ``arxiv`` PyPI package with a clean dataclass output,
arXiv-ID normalisation, and a hard 1-request-per-3-seconds rate limit.
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import arxiv

logger = logging.getLogger(__name__)

# arXiv hard rate limit.
_RATE_LIMIT_SECONDS = 3.0

# Matches arXiv IDs in URLs or bare forms: 2106.00001, arXiv:2106.00001, old-style hep-th/9901001
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/abs/|arxiv\.org/pdf/|arXiv:)?"
    r"([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z\-]+/[0-9]{7}(?:v[0-9]+)?)",
    re.IGNORECASE,
)


def normalize_arxiv_id(raw: str) -> str:
    """Extract and normalise an arXiv ID from a URL or raw string.

    >>> normalize_arxiv_id("http://arxiv.org/abs/2106.00001v2")
    '2106.00001v2'
    >>> normalize_arxiv_id("arXiv:2106.00001")
    '2106.00001'
    >>> normalize_arxiv_id("cs.LG/0701001")
    'cs.LG/0701001'
    """
    match = _ARXIV_ID_RE.search(raw.strip())
    if not match:
        raise ValueError(f"Could not extract arXiv ID from: {raw!r}")
    return match.group(1)


def strip_version(arxiv_id: str) -> str:
    """Remove the ``vN`` suffix from an arXiv ID."""
    return re.sub(r"v[0-9]+$", "", arxiv_id)


@dataclass
class ArxivPaper:
    """Normalised representation of an arXiv search result."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str | None = None  # ISO date
    updated: str | None = None  # ISO datetime
    categories: list[str] = field(default_factory=list)
    primary_category: str | None = None
    pdf_url: str | None = None
    abs_url: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None

    @classmethod
    def from_result(cls, result: arxiv.Result) -> "ArxivPaper":
        """Convert an ``arxiv.Result`` into an :class:`ArxivPaper`."""
        arxiv_id = normalize_arxiv_id(result.entry_id)
        authors = [a.name for a in result.authors]
        published = result.published.isoformat() if result.published else None
        updated = result.updated.isoformat() if result.updated else None
        return cls(
            arxiv_id=arxiv_id,
            title=result.title,
            authors=authors,
            abstract=result.summary,
            published=published,
            updated=updated,
            categories=list(result.categories),
            primary_category=result.primary_category,
            pdf_url=result.pdf_url,
            abs_url=result.entry_id,
            doi=result.doi,
            journal_ref=result.journal_ref,
            comment=result.comment,
        )

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "pdf_url": self.pdf_url,
            "abs_url": self.abs_url,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comment": self.comment,
        }


class ArxivClient:
    """Rate-limited arXiv search client.

    The ``arxiv`` package is synchronous, so this client is also synchronous.
    Callers should run it in a thread executor when inside an async context.
    """

    def __init__(
        self,
        page_size: int = 100,
        delay_seconds: float = _RATE_LIMIT_SECONDS,
        num_retries: int = 5,
    ) -> None:
        self._client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )
        self._last_request: float = 0.0

    def _throttle(self) -> None:
        """Ensure at least ``delay_seconds`` between requests."""
        elapsed = time.monotonic() - self._last_request
        wait = _RATE_LIMIT_SECONDS - elapsed
        if wait > 0:
            logger.debug("arXiv rate-limit: sleeping %.1fs", wait)
            time.sleep(wait)
        self._last_request = time.monotonic()

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        primary_category: str | None = None,
    ) -> list[ArxivPaper]:
        """Search arXiv and return a list of :class:`ArxivPaper`.

        ``sort_by`` is one of ``relevance``, ``last_updated``,
        ``submitted_date``.
        """
        # Build the arXiv query string, optionally constraining to a category.
        full_query = query
        if primary_category:
            full_query = f"cat:{primary_category} AND ({query})"

        criterion = {
            "relevance": arxiv.SortCriterion.Relevance,
            "last_updated": arxiv.SortCriterion.LastUpdatedDate,
            "submitted_date": arxiv.SortCriterion.SubmittedDate,
        }.get(sort_by, arxiv.SortCriterion.Relevance)

        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        self._throttle()
        results: list[ArxivPaper] = []
        for r in self._client.results(search):
            results.append(ArxivPaper.from_result(r))
        return results

    def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        """Fetch a single paper by arXiv ID."""
        search = arxiv.Search(id_list=[strip_version(arxiv_id)], max_results=1)
        self._throttle()
        for r in self._client.results(search):
            return ArxivPaper.from_result(r)
        return None
