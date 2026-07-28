"""Semantic Scholar Graph API client (async).

arXiv does not provide citation counts, venue, or author affiliations, so we
enrich every paper via the free Semantic Scholar (S2) Graph API.

This client is fully async (``httpx.AsyncClient``) and exposes:

* :meth:`fetch_paper`   – single-paper lookup
* :meth:`fetch_batch`   – explicit batch lookup (≤ 500 IDs)
* :meth:`enqueue` / :meth:`flush` – an async batch *queue* that accumulates
  arXiv IDs and flushes them on a timer (every ``flush_interval`` seconds)
  to amortise rate-limit costs.

Unauthenticated S2 limit is ~1 request / 1 s (with retries on 429).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "paperId,title,citationCount,influentialCitationCount,"
    "venue,publicationVenue,year,authors,externalIds"
)
S2_USER_AGENT = "research-library-mcp/0.1 (local research tool)"
S2_MAX_BATCH = 500
S2_RATE_LIMIT_SECONDS = 1.0
MISSING_CITATION = -1

@dataclass
class S2Metrics:
    """Enrichment metrics for a single paper."""

    arxiv_id: str
    citation_count: int = MISSING_CITATION
    influential_citation_count: int = MISSING_CITATION
    venue: str | None = None
    s2_paper_id: str | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "venue": self.venue,
            "s2_paper_id": self.s2_paper_id,
        }

def _parse_paper(arxiv_id: str, data: dict[str, Any] | None) -> S2Metrics:
    """Convert a single S2 paper JSON object into :class:`S2Metrics`.

    ``data`` may be ``None`` (paper not found) or a dict.
    """
    if not data:
        return S2Metrics(arxiv_id=arxiv_id)
    venue = data.get("venue") or ""
    pub_venue = data.get("publicationVenue")
    if not venue and pub_venue and isinstance(pub_venue, dict):
        venue = pub_venue.get("name") or ""
    venue = venue or None
    return S2Metrics(
        arxiv_id=arxiv_id,
        citation_count=int(data.get("citationCount") or MISSING_CITATION),
        influential_citation_count=int(
            data.get("influentialCitationCount") or MISSING_CITATION
        ),
        venue=venue,
        s2_paper_id=data.get("paperId"),
        raw=data,
    )

class SemanticScholarClient:
    """Async S2 Graph API client with a batch queue."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 4,
        flush_interval: float = 5.0,
    ) -> None:
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout
        self._max_retries = max_retries
        self._flush_interval = flush_interval
        self._queue: dict[str, asyncio.Future[S2Metrics]] = {}
        self._queue_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._last_request: float = 0.0

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": S2_USER_AGENT},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None

    async def _request(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        client = await self._ensure_client()
        backoff = 2.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            now = asyncio.get_event_loop().time()
            wait = S2_RATE_LIMIT_SECONDS - (now - self._last_request)
            if wait > 0 and attempt == 0:
                await asyncio.sleep(wait)
            self._last_request = asyncio.get_event_loop().time()
            try:
                resp = await client.request(method, url, **kwargs)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff))
                    logger.warning("S2 429 – backing off %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 404:
                    raise
                if attempt < self._max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    async def fetch_paper(self, arxiv_id: str) -> S2Metrics:
        """Fetch metrics for one arXiv ID (``ARXIV:`` prefix added)."""
        url = f"{S2_BASE}/paper/ARXIV:{arxiv_id}"
        try:
            resp = await self._request("GET", url, params={"fields": S2_FIELDS})
            return _parse_paper(arxiv_id, resp.json())
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("S2: paper %s not found", arxiv_id)
                return S2Metrics(arxiv_id=arxiv_id)
            raise
        except httpx.HTTPError as exc:
            logger.warning("S2: fetch_paper %s failed: %s", arxiv_id, exc)
            return S2Metrics(arxiv_id=arxiv_id)

    async def fetch_batch(self, arxiv_ids: list[str]) -> dict[str, S2Metrics]:
        """Fetch metrics for up to ``S2_MAX_BATCH`` arXiv IDs at once.

        Returns a mapping ``{arxiv_id: S2Metrics}``.
        """
        if not arxiv_ids:
            return {}
        ids_chunk = arxiv_ids[:S2_MAX_BATCH]
        s2_ids = [f"ARXIV:{aid}" for aid in ids_chunk]
        url = f"{S2_BASE}/paper/batch"
        try:
            resp = await self._request(
                "POST", url, json={"ids": s2_ids}, params={"fields": S2_FIELDS}
            )
            data_list = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("S2: fetch_batch failed: %s", exc)
            return {aid: S2Metrics(arxiv_id=aid) for aid in ids_chunk}

        out: dict[str, S2Metrics] = {}
        for aid, data in zip(ids_chunk, data_list):
            out[aid] = _parse_paper(aid, data)
        return out

    async def enqueue(self, arxiv_id: str) -> S2Metrics:
        """Queue an arXiv ID for batch enrichment.

        Returns a Future that resolves when the next flush completes.
        Starts the flush timer if not already running.
        """
        async with self._queue_lock:
            if arxiv_id in self._queue:
                return await asyncio.shield(self._queue[arxiv_id])
            fut: asyncio.Future[S2Metrics] = asyncio.get_event_loop().create_future()
            self._queue[arxiv_id] = fut
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_loop())
        return await asyncio.shield(fut)

    async def _flush_loop(self) -> None:
        """Wait for the flush interval, then flush the queue."""
        try:
            await asyncio.sleep(self._flush_interval)
            await self.flush()
        except asyncio.CancelledError:
            await self.flush()
            raise

    async def flush(self) -> None:
        """Flush all queued IDs in a single batch request."""
        async with self._queue_lock:
            if not self._queue:
                return
            pending = dict(self._queue)
            self._queue.clear()
        ids = list(pending.keys())
        logger.debug("S2 flush: %d ids", len(ids))
        results = await self.fetch_batch(ids)
        for aid, fut in pending.items():
            if not fut.done():
                metrics = results.get(aid, S2Metrics(arxiv_id=aid))
                fut.set_result(metrics)
