"""SQLite data-access layer.

A single :class:`Database` wraps a SQLite connection and exposes typed
CRUD helpers for every table described in the data model.  All JSON
columns (``authors``, ``categories``, ``constraints_used`` …) are stored as
JSON text and decoded/encoded with :mod:`json`.

The class accepts an optional ``path`` so tests can pass ``":memory:"`` for
an ephemeral in-memory database.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    authors           TEXT NOT NULL,        -- JSON array of names
    abstract          TEXT NOT NULL,
    published         TEXT,                 -- ISO date
    updated           TEXT,                 -- ISO datetime
    categories        TEXT,                 -- JSON array
    primary_category  TEXT,
    pdf_url           TEXT,
    abs_url           TEXT,
    doi               TEXT,
    journal_ref       TEXT,
    comment           TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    arxiv_id                   TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    citation_count             INTEGER NOT NULL DEFAULT -1,
    influential_citation_count INTEGER NOT NULL DEFAULT -1,
    venue                      TEXT,
    s2_paper_id                TEXT,
    raw_s2_json                TEXT,
    created_at                 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id     TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    section      TEXT NOT NULL,
    content      TEXT NOT NULL,
    model_used   TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE(arxiv_id, section)
);

CREATE TABLE IF NOT EXISTS ideas (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id              TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    idea_text             TEXT NOT NULL,
    constraints_used      TEXT,             -- JSON
    generated_with_model  TEXT,
    status                TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected
    search_queries        TEXT,             -- JSON array
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS novelty_checks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id           INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    query_terms       TEXT,                -- JSON array
    similar_arxiv_ids TEXT,                -- JSON array
    verdict           TEXT NOT NULL,       -- likely_novel|needs_review|similar_exists
    notes             TEXT,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type    TEXT NOT NULL,          -- search|import|summarize|idea|novelty|query|remove
    arxiv_id       TEXT,
    query          TEXT,
    status         TEXT NOT NULL,          -- started|completed|failed
    metadata_json  TEXT,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_citation_count ON metrics(citation_count);
CREATE INDEX IF NOT EXISTS idx_metrics_venue          ON metrics(venue);
CREATE INDEX IF NOT EXISTS idx_papers_primary_cat     ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_activity_created       ON activity_log(created_at);
CREATE INDEX IF NOT EXISTS idx_summaries_arxiv        ON summaries(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_ideas_arxiv            ON ideas(arxiv_id);
"""

@dataclass
class PaperRow:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str | None = None
    updated: str | None = None
    categories: list[str] = field(default_factory=list)
    primary_category: str | None = None
    pdf_url: str | None = None
    abs_url: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None
    created_at: str = ""
    updated_at: str = ""

@dataclass
class MetricsRow:
    arxiv_id: str
    citation_count: int = -1
    influential_citation_count: int = -1
    venue: str | None = None
    s2_paper_id: str | None = None
    raw_s2_json: str | None = None
    created_at: str = ""

@dataclass
class SummaryRow:
    id: int | None
    arxiv_id: str
    section: str
    content: str
    model_used: str | None = None
    created_at: str = ""

@dataclass
class IdeaRow:
    id: int | None
    arxiv_id: str
    idea_text: str
    constraints_used: dict | None = None
    generated_with_model: str | None = None
    status: str = "pending"
    search_queries: list[str] = field(default_factory=list)
    created_at: str = ""

@dataclass
class NoveltyCheckRow:
    id: int | None
    idea_id: int
    query_terms: list[str] = field(default_factory=list)
    similar_arxiv_ids: list[str] = field(default_factory=list)
    verdict: str = ""
    notes: str | None = None
    created_at: str = ""

@dataclass
class ActivityRow:
    id: int | None
    action_type: str
    arxiv_id: str | None = None
    query: str | None = None
    status: str = "started"
    metadata_json: dict | None = None
    created_at: str = ""

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

class Database:
    """Thin SQLite wrapper with typed helpers."""

    def __init__(self, path: str | Path = "") -> None:
        self.path: str = str(path) if path else ""
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        path = self.path or ":memory:"
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager that commits on success, rolls back on error."""
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def upsert_paper(self, p: PaperRow) -> None:
        now = _now_iso()
        p.created_at = p.created_at or now
        p.updated_at = now
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO papers (arxiv_id, title, authors, abstract, published,
                    updated, categories, primary_category, pdf_url, abs_url, doi,
                    journal_ref, comment, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    title=excluded.title, authors=excluded.authors,
                    abstract=excluded.abstract, published=excluded.published,
                    updated=excluded.updated, categories=excluded.categories,
                    primary_category=excluded.primary_category,
                    pdf_url=excluded.pdf_url, abs_url=excluded.abs_url,
                    doi=excluded.doi, journal_ref=excluded.journal_ref,
                    comment=excluded.comment, updated_at=excluded.updated_at
                """,
                (
                    p.arxiv_id, p.title, json.dumps(p.authors), p.abstract,
                    p.published, p.updated, json.dumps(p.categories),
                    p.primary_category, p.pdf_url, p.abs_url, p.doi,
                    p.journal_ref, p.comment, p.created_at, p.updated_at,
                ),
            )

    def get_paper(self, arxiv_id: str) -> PaperRow | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        return self._row_to_paper(row) if row else None

    def list_papers(
        self,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        primary_category: str | None = None,
        min_citations: int | None = None,
        venue: str | None = None,
    ) -> list[PaperRow]:
        sort_col = {
            "citation_count": "m.citation_count",
            "published": "p.published",
            "created_at": "p.created_at",
        }.get(sort_by, "p.created_at")
        desc = " DESC" if sort_by != "published" else " DESC"
        sql = (
            "SELECT p.* FROM papers p "
            "LEFT JOIN metrics m ON p.arxiv_id = m.arxiv_id WHERE 1=1"
        )
        params: list[Any] = []
        if primary_category:
            sql += " AND p.primary_category = ?"
            params.append(primary_category)
        if min_citations is not None:
            sql += " AND m.citation_count >= ?"
            params.append(min_citations)
        if venue:
            sql += " AND m.venue LIKE ?"
            params.append(f"%{venue}%")
        sql += f" ORDER BY {sort_col}{desc} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def count_papers(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])

    def delete_paper(self, arxiv_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM papers WHERE arxiv_id=?", (arxiv_id,))
        self.conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> PaperRow:
        return PaperRow(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            authors=json.loads(row["authors"]) if row["authors"] else [],
            abstract=row["abstract"],
            published=row["published"],
            updated=row["updated"],
            categories=json.loads(row["categories"]) if row["categories"] else [],
            primary_category=row["primary_category"],
            pdf_url=row["pdf_url"],
            abs_url=row["abs_url"],
            doi=row["doi"],
            journal_ref=row["journal_ref"],
            comment=row["comment"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert_metrics(self, m: MetricsRow) -> None:
        m.created_at = m.created_at or _now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO metrics (arxiv_id, citation_count,
                    influential_citation_count, venue, s2_paper_id,
                    raw_s2_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    citation_count=excluded.citation_count,
                    influential_citation_count=excluded.influential_citation_count,
                    venue=excluded.venue, s2_paper_id=excluded.s2_paper_id,
                    raw_s2_json=excluded.raw_s2_json
                """,
                (
                    m.arxiv_id, m.citation_count, m.influential_citation_count,
                    m.venue, m.s2_paper_id, m.raw_s2_json, m.created_at,
                ),
            )

    def get_metrics(self, arxiv_id: str) -> MetricsRow | None:
        row = self.conn.execute(
            "SELECT * FROM metrics WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        if not row:
            return None
        return MetricsRow(
            arxiv_id=row["arxiv_id"],
            citation_count=row["citation_count"],
            influential_citation_count=row["influential_citation_count"],
            venue=row["venue"],
            s2_paper_id=row["s2_paper_id"],
            raw_s2_json=row["raw_s2_json"],
            created_at=row["created_at"],
        )

    def upsert_summary(self, s: SummaryRow) -> int:
        s.created_at = s.created_at or _now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO summaries (arxiv_id, section, content, model_used, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id, section) DO UPDATE SET
                    content=excluded.content, model_used=excluded.model_used,
                    created_at=excluded.created_at
                """,
                (s.arxiv_id, s.section, s.content, s.model_used, s.created_at),
            )
            return int(cur.lastrowid or self._summary_id(s.arxiv_id, s.section))

    def _summary_id(self, arxiv_id: str, section: str) -> int:
        row = self.conn.execute(
            "SELECT id FROM summaries WHERE arxiv_id=? AND section=?",
            (arxiv_id, section),
        ).fetchone()
        return int(row[0]) if row else 0

    def get_summaries(self, arxiv_id: str) -> list[SummaryRow]:
        rows = self.conn.execute(
            "SELECT * FROM summaries WHERE arxiv_id=? ORDER BY section",
            (arxiv_id,),
        ).fetchall()
        return [
            SummaryRow(
                id=r["id"], arxiv_id=r["arxiv_id"], section=r["section"],
                content=r["content"], model_used=r["model_used"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_summary(self, arxiv_id: str, section: str) -> SummaryRow | None:
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE arxiv_id=? AND section=?",
            (arxiv_id, section),
        ).fetchone()
        if not row:
            return None
        return SummaryRow(
            id=row["id"], arxiv_id=row["arxiv_id"], section=row["section"],
            content=row["content"], model_used=row["model_used"],
            created_at=row["created_at"],
        )

    def add_idea(self, i: IdeaRow) -> int:
        i.created_at = i.created_at or _now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO ideas (arxiv_id, idea_text, constraints_used,
                    generated_with_model, status, search_queries, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    i.arxiv_id, i.idea_text,
                    json.dumps(i.constraints_used) if i.constraints_used else None,
                    i.generated_with_model, i.status,
                    json.dumps(i.search_queries) if i.search_queries else None,
                    i.created_at,
                ),
            )
            return int(cur.lastrowid)

    def get_idea(self, idea_id: int) -> IdeaRow | None:
        row = self.conn.execute("SELECT * FROM ideas WHERE id=?", (idea_id,)).fetchone()
        return self._row_to_idea(row) if row else None

    def list_ideas(self, arxiv_id: str) -> list[IdeaRow]:
        rows = self.conn.execute(
            "SELECT * FROM ideas WHERE arxiv_id=? ORDER BY id", (arxiv_id,)
        ).fetchall()
        return [self._row_to_idea(r) for r in rows]

    def update_idea_status(self, idea_id: int, status: str) -> bool:
        cur = self.conn.execute(
            "UPDATE ideas SET status=? WHERE id=?", (status, idea_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_idea(row: sqlite3.Row) -> IdeaRow:
        return IdeaRow(
            id=row["id"],
            arxiv_id=row["arxiv_id"],
            idea_text=row["idea_text"],
            constraints_used=json.loads(row["constraints_used"]) if row["constraints_used"] else None,
            generated_with_model=row["generated_with_model"],
            status=row["status"],
            search_queries=json.loads(row["search_queries"]) if row["search_queries"] else [],
            created_at=row["created_at"],
        )

    def add_novelty_check(self, n: NoveltyCheckRow) -> int:
        n.created_at = n.created_at or _now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO novelty_checks (idea_id, query_terms, similar_arxiv_ids,
                    verdict, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    n.idea_id,
                    json.dumps(n.query_terms) if n.query_terms else None,
                    json.dumps(n.similar_arxiv_ids) if n.similar_arxiv_ids else None,
                    n.verdict, n.notes, n.created_at,
                ),
            )
            return int(cur.lastrowid)

    def get_novelty_checks(self, idea_id: int) -> list[NoveltyCheckRow]:
        rows = self.conn.execute(
            "SELECT * FROM novelty_checks WHERE idea_id=? ORDER BY id DESC",
            (idea_id,),
        ).fetchall()
        return [
            NoveltyCheckRow(
                id=r["id"], idea_id=r["idea_id"],
                query_terms=json.loads(r["query_terms"]) if r["query_terms"] else [],
                similar_arxiv_ids=json.loads(r["similar_arxiv_ids"]) if r["similar_arxiv_ids"] else [],
                verdict=r["verdict"], notes=r["notes"], created_at=r["created_at"],
            )
            for r in rows
        ]

    def log_activity(self, a: ActivityRow) -> int:
        a.created_at = a.created_at or _now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO activity_log (action_type, arxiv_id, query, status,
                    metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    a.action_type, a.arxiv_id, a.query, a.status,
                    json.dumps(a.metadata_json) if a.metadata_json else None,
                    a.created_at,
                ),
            )
            return int(cur.lastrowid)

    def update_activity_status(self, log_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE activity_log SET status=? WHERE id=?", (status, log_id)
        )
        self.conn.commit()

    def list_activity(self, limit: int = 50, action_type: str | None = None) -> list[ActivityRow]:
        sql = "SELECT * FROM activity_log"
        params: list[Any] = []
        if action_type:
            sql += " WHERE action_type=?"
            params.append(action_type)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [
            ActivityRow(
                id=r["id"], action_type=r["action_type"], arxiv_id=r["arxiv_id"],
                query=r["query"], status=r["status"],
                metadata_json=json.loads(r["metadata_json"]) if r["metadata_json"] else None,
                created_at=r["created_at"],
            )
            for r in rows
        ]
