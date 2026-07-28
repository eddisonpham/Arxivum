"""ChromaDB vector store for paper chunks.

Stores embeddings of abstracts, titles, and generated summary sections.
Uses ``BAAI/bge-small-en-v1.5`` (384-d) embeddings provided by the caller
so the embedding logic stays testable and decoupled from the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

CHUNK_ABSTRACT = "abstract"
CHUNK_TITLE = "title"
CHUNK_SUMMARY = "summary"

@dataclass
class Chunk:
    """A single document chunk stored in ChromaDB."""

    id: str
    document: str
    arxiv_id: str
    chunk_type: str
    title: str = ""
    venue: str | None = None
    citation_count: int = -1
    primary_category: str | None = None
    section: str | None = None
    embedding: list[float] = field(default_factory=list)

@dataclass
class QueryResult:
    """A single retrieval result."""

    arxiv_id: str
    chunk_id: str
    document: str
    score: float
    chunk_type: str
    title: str = ""
    venue: str | None = None
    citation_count: int = -1
    primary_category: str | None = None
    section: str | None = None

COLLECTION_NAME = "paper_chunks"

class ChromaStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        path: str = "",
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.path = path
        if path:
            self._client = chromadb.PersistentClient(
                path=path, settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def chunk_id(arxiv_id: str, chunk_type: str, section: str | None = None) -> str:
        """Build a deterministic chunk ID.

        Scheme: ``{arxiv_id}#{chunk_type}#{section}`` (section optional).
        """
        if section:
            return f"{arxiv_id}#{chunk_type}#{section}"
        return f"{arxiv_id}#{chunk_type}#0"

    @staticmethod
    def _meta(chunk: Chunk) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "arxiv_id": chunk.arxiv_id,
            "chunk_type": chunk.chunk_type,
            "title": chunk.title,
        }
        if chunk.venue is not None:
            meta["venue"] = chunk.venue
        if chunk.citation_count is not None:
            meta["citation_count"] = chunk.citation_count
        if chunk.primary_category is not None:
            meta["primary_category"] = chunk.primary_category
        if chunk.section is not None:
            meta["section"] = chunk.section
        return meta

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Insert or update a batch of pre-embedded chunks."""
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.document for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = [self._meta(c) for c in chunks]
        self._collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    def upsert_chunk(self, chunk: Chunk) -> None:
        self.upsert_chunks([chunk])

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """Vector similarity search with optional metadata filter."""
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        results: list[QueryResult] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = max(0.0, 1.0 - float(dist))
            results.append(
                QueryResult(
                    arxiv_id=meta.get("arxiv_id", ""),
                    chunk_id=cid,
                    document=doc,
                    score=score,
                    chunk_type=meta.get("chunk_type", ""),
                    title=meta.get("title", ""),
                    venue=meta.get("venue"),
                    citation_count=meta.get("citation_count", -1),
                    primary_category=meta.get("primary_category"),
                    section=meta.get("section"),
                )
            )
        return results

    def get_by_arxiv_id(self, arxiv_id: str) -> list[Chunk]:
        """Return all chunks belonging to a paper."""
        res = self._collection.get(where={"arxiv_id": arxiv_id})
        chunks: list[Chunk] = []
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        embeddings = res.get("embeddings", [])
        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            chunks.append(
                Chunk(
                    id=cid,
                    document=docs[i] if i < len(docs) else "",
                    arxiv_id=meta.get("arxiv_id", arxiv_id),
                    chunk_type=meta.get("chunk_type", ""),
                    title=meta.get("title", ""),
                    venue=meta.get("venue"),
                    citation_count=meta.get("citation_count", -1),
                    primary_category=meta.get("primary_category"),
                    section=meta.get("section"),
                    embedding=list(embeddings[i]) if embeddings is not None and i < len(embeddings) else [],
                )
            )
        return chunks

    def count(self) -> int:
        return self._collection.count()

    def delete_by_arxiv_id(self, arxiv_id: str) -> None:
        """Remove all chunks for a paper."""
        self._collection.delete(where={"arxiv_id": arxiv_id})

    def reset(self) -> None:
        """Drop and recreate the collection (used by tests / migrate)."""
        try:
            self._client.delete_collection(self._collection.name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
