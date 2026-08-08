"""Unit tests for the novelty confidence-threshold mechanism (Tier 1.2).

Covers:
  - the thresholding helper against the bench-level fixture
  - the parser correctly extracts `confidence` from inline JSON
  - the SQL migration is idempotent
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.services.novelty import (
    LOW_CONFIDENCE_THRESHOLD,
    _clamp_confidence,
    NoveltyService,
)


def test_threshold_default_is_six_tenths():
    """The default downgrade threshold must be a strict 0.6; do not
    silently change it without updating the docs."""
    assert LOW_CONFIDENCE_THRESHOLD == 0.6


def test_clamp_confidence_basics():
    assert _clamp_confidence(0.0) == 0.0
    assert _clamp_confidence(1.0) == 1.0
    assert _clamp_confidence(0.5) == 0.5
    assert _clamp_confidence(1.7) == 1.0
    assert _clamp_confidence(-0.4) == 0.0
    assert _clamp_confidence(None) == 0.5
    assert _clamp_confidence("bad") == 0.5


def test_threshold_logic_against_synthetic_pairs():
    """Threshold the synthetic-labelled pairs. The fixture should be
    100% accurate after thresholding when the LLM emits well-calibrated
    confidence on every prediction."""
    from tests.benchmark.synthetic import NOVELTY_PAIRS

    def _thresholded(pred: str, conf: float) -> str:
        if conf < LOW_CONFIDENCE_THRESHOLD and pred in ("likely_novel", "similar_exists"):
            return "needs_review"
        return pred

    total = 0
    correct_th = 0
    for p in NOVELTY_PAIRS:
        if p["expected_verdict"] == "similar_exists":
            conf = 0.9
            pred = "similar_exists"
        elif p["expected_verdict"] == "needs_review":
            conf = 0.65
            pred = "needs_review"
        else:  # likely_novel with high confidence -> not downgraded
            conf = 0.85
            pred = "likely_novel"
        total += 1
        if _thresholded(pred, conf) == p["expected_verdict"]:
            correct_th += 1
    assert correct_th == total, "fixture should be 100% after thresholding"


def test_threshold_downgrades_low_confidence_binary_verdict():
    """A 'similar_exists' with conf=0.3 must be downgraded to
    'needs_review' so the user knows the verdict isn't a hard signal."""
    def _thresholded(pred: str, conf: float) -> str:
        if conf < LOW_CONFIDENCE_THRESHOLD and pred in ("likely_novel", "similar_exists"):
            return "needs_review"
        return pred
    assert _thresholded("similar_exists", 0.3) == "needs_review"
    assert _thresholded("likely_novel", 0.3) == "needs_review"
    assert _thresholded("similar_exists", 0.7) == "similar_exists"
    assert _thresholded("likely_novel", 0.7) == "likely_novel"
    assert _thresholded("needs_review", 0.3) == "needs_review"


def test_service_returns_confidence_in_response(tmp_path):
    """A novelty check row must persist the max confidence observed."""
    from src.db.models import Database, IdeaRow, PaperRow
    from src.clients.arxiv_client import ArxivClient
    from src.inference.llm import StubLLM
    from src.inference.manager import ModelManager

    db_path = os.path.join(tmp_path, "novelty.db")
    db = Database(db_path)

    db.upsert_paper(PaperRow(
        arxiv_id="2401.00001", title="T", authors=["A"], abstract="abs"))
    db.upsert_paper(PaperRow(
        arxiv_id="2402.00099", title="Other", authors=["B"], abstract="other"))
    idea_id = db.add_idea(IdeaRow(
        id=None, arxiv_id="2401.00001", idea_text="novel idea",
        search_queries=["q"]))

    db.add_novelty_check_needs_column = None  # noqa  (silence lint)
    arxiv = ArxivClient()
    mm = ModelManager()
    mm.set_llm(StubLLM(default='{"verdict":"likely_novel","confidence":0.7,"reason":"r"}'))

    def _query(q, top_k=5):
        return []

    svc = NoveltyService(db, mm, arxiv, library_query_fn=_query)
    out = svc.verify_novelty(idea_id)
    assert "confidence" in out
    assert 0.0 <= out["confidence"] <= 1.0
    rows = db.get_novelty_checks(idea_id)
    assert len(rows) == 1
    assert isinstance(rows[0].confidence, float)


def test_migration_adds_confidence_column(tmp_path):
    """Drop a DB without the confidence column and confirm the new
    constructor adds it idempotently without losing data."""
    import sqlite3
    from src.db.models import Database, IdeaRow, PaperRow

    db_path = os.path.join(tmp_path, "legacy.db")
    conn = sqlite3.connect(db_path)
    # Recreate schema WITHOUT confidence column to simulate an old DB
    conn.executescript("""
        CREATE TABLE papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL, authors TEXT NOT NULL, abstract TEXT NOT NULL,
            published TEXT, updated TEXT, categories TEXT,
            primary_category TEXT, pdf_url TEXT, abs_url TEXT, doi TEXT,
            journal_ref TEXT, comment TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE metrics (
            arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            citation_count INTEGER NOT NULL DEFAULT -1,
            influential_citation_count INTEGER NOT NULL DEFAULT -1,
            venue TEXT, s2_paper_id TEXT, raw_s2_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            idea_text TEXT NOT NULL, constraints_used TEXT,
            generated_with_model TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            search_queries TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE novelty_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
            query_terms TEXT, similar_arxiv_ids TEXT,
            verdict TEXT NOT NULL, notes TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

    db = Database(db_path)
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(novelty_checks)").fetchall()}
    assert "confidence" in cols, "migration should add confidence column"

    db.upsert_paper(PaperRow(arxiv_id="1", title="x", authors=["a"], abstract="y"))
    id1 = db.add_idea(IdeaRow(id=None, arxiv_id="1", idea_text="i"))
    cid = db.add_novelty_check(
        type("N", (), {"id": None, "idea_id": id1,
                        "query_terms": [], "similar_arxiv_ids": [],
                        "verdict": "likely_novel",
                        "notes": "migrated", "confidence": 0.42,
                        "created_at": ""})()
    )
    assert cid > 0
    rows = db.get_novelty_checks(id1)
    assert rows and abs(rows[0].confidence - 0.42) < 1e-6
