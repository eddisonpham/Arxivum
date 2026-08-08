"""Structured extraction service.

Extracts a structured bibliographic schema from a paper's title +
abstract. Schema follows common scientific-paper IE conventions
(SPECTER, OpenAlex, OSLO ontology).

Output schema::

    {
      "method": "<one-sentence method>",
      "datasets": ["<dataset 1>", "<dataset 2>"],
      "baselines": ["<baseline 1>", "<baseline 2>"],
      "headline_metric": {"name": "<metric>", "value": "<value>", "split": "<split>"},
      "contribution": "<one-sentence contribution claim>",
      "limitations": ["<limitation 1>", ...],
      "domain": "<sub-field>",
      "bibcode": "arXiv:<id>"
    }

The extractor caches its output in the ``summaries`` table under the
synthetic section name ``extraction_v1`` to avoid re-running the LLM
when the same paper is requested twice.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.db.models import Database, SummaryRow
from src.inference.manager import ModelManager
from src.services.prompts import extract_messages, extract_json
from src.utils import track_activity

logger = logging.getLogger(__name__)

EXTRACTION_SECTION = "extraction_v1"

_EXTRACTION_SCHEMA_KEYS = (
    "method", "datasets", "baselines", "headline_metric",
    "contribution", "limitations", "domain", "bibcode",
)


class StructuredExtractor:
    """Extract a structured schema from a paper's abstract."""

    def __init__(self, db: Database, models: ModelManager) -> None:
        self.db = db
        self.models = models

    def extract(self, arxiv_id: str, force: bool = False) -> dict[str, Any]:
        """Extract a structured schema for ``arxiv_id``.

        Returns a dict in the canonical schema. Cached results under
        ``section == 'extraction_v1'`` are returned unless ``force=True``.
        """
        if not force:
            cached = self.db.get_summary(arxiv_id, EXTRACTION_SECTION)
            if cached and cached.content:
                try:
                    parsed = json.loads(cached.content)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    logger.warning("Cached extraction is not JSON; regenerating")

        paper = self.db.get_paper(arxiv_id)
        if not paper:
            raise ValueError(f"Paper {arxiv_id} not found in library")

        with track_activity(
            self.db, "extract", arxiv_id=arxiv_id,
        ):
            messages = extract_messages(paper.title, paper.abstract, paper.arxiv_id)
            raw = self.models.llm.chat(
                messages=messages, temperature=0.2, max_tokens=512,
            )
            try:
                parsed = extract_json(raw)
            except ValueError as exc:
                logger.warning("Extraction parse failed: %s", exc)
                parsed = {"method": "", "datasets": [], "baselines": [],
                          "headline_metric": {}, "contribution": "",
                          "limitations": [], "domain": "unknown",
                          "bibcode": f"arXiv:{arxiv_id}"}
            if isinstance(parsed, dict):
                schema = self._normalise(parsed, arxiv_id)
            else:
                schema = self._empty_schema(arxiv_id)
            self._store(arxiv_id, schema)
            return schema

    @staticmethod
    def _normalise(parsed: dict, arxiv_id: str) -> dict:
        out = {k: parsed.get(k, "" if k in ("method", "contribution", "domain") else [] if k not in ("headline_metric",) else {}) for k in _EXTRACTION_SCHEMA_KEYS}
        for list_field in ("datasets", "baselines", "limitations"):
            v = out[list_field]
            if v is None:
                out[list_field] = []
            elif not isinstance(v, list):
                out[list_field] = [str(v)]
        hm = out["headline_metric"]
        if not isinstance(hm, dict) or hm is None:
            hm_name = str(hm) if hm is not None else ""
            out["headline_metric"] = {"name": hm_name, "value": "", "split": ""}
        out["bibcode"] = out.get("bibcode") or f"arXiv:{arxiv_id}"
        return out

    @staticmethod
    def _empty_schema(arxiv_id: str) -> dict:
        return {
            "method": "",
            "datasets": [],
            "baselines": [],
            "headline_metric": {"name": "", "value": "", "split": ""},
            "contribution": "",
            "limitations": [],
            "domain": "unknown",
            "bibcode": f"arXiv:{arxiv_id}",
        }

    def _store(self, arxiv_id: str, schema: dict) -> None:
        model_name = getattr(self.models.llm, "model_path", "stub")
        self.db.upsert_summary(SummaryRow(
            id=None, arxiv_id=arxiv_id, section=EXTRACTION_SECTION,
            content=json.dumps(schema, ensure_ascii=False),
            model_used=model_name,
        ))

    def get_cached(self, arxiv_id: str) -> dict[str, Any] | None:
        row = self.db.get_summary(arxiv_id, EXTRACTION_SECTION)
        if not row or not row.content:
            return None
        try:
            parsed = json.loads(row.content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
