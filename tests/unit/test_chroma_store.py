"""Unit tests for the ChromaDB vector store."""

import pytest

from src.db.chroma_store import (
    CHUNK_ABSTRACT, CHUNK_TITLE, Chunk, ChromaStore, QueryResult,
)


class TestChromaStore:
    def test_chunk_id_scheme(self):
        assert ChromaStore.chunk_id("2106.00001", CHUNK_ABSTRACT) == "2106.00001#abstract#0"
        assert ChromaStore.chunk_id("2106.00001", CHUNK_TITLE) == "2106.00001#title#0"
        assert ChromaStore.chunk_id("2106.00001", "summary", "methodology") == "2106.00001#summary#methodology"

    def test_upsert_and_count(self, chroma_store):
        chunk = Chunk(
            id=ChromaStore.chunk_id("2106.00001", CHUNK_ABSTRACT),
            document="Test abstract text",
            arxiv_id="2106.00001",
            chunk_type=CHUNK_ABSTRACT,
            title="Test Paper",
            citation_count=10,
            primary_category="cs.LG",
            embedding=[0.1] * 384,
        )
        chroma_store.upsert_chunk(chunk)
        assert chroma_store.count() == 1

    def test_upsert_multiple(self, chroma_store):
        chunks = [
            Chunk(
                id=ChromaStore.chunk_id(f"2106.{i:05d}", CHUNK_ABSTRACT),
                document=f"abstract {i}", arxiv_id=f"2106.{i:05d}",
                chunk_type=CHUNK_ABSTRACT, title=f"Paper {i}",
                embedding=[0.1 * i] * 384,
            )
            for i in range(3)
        ]
        chroma_store.upsert_chunks(chunks)
        assert chroma_store.count() == 3

    def test_query_returns_results(self, chroma_store):
        chunk = Chunk(
            id=ChromaStore.chunk_id("2106.00001", CHUNK_ABSTRACT),
            document="attention mechanism transformer",
            arxiv_id="2106.00001",
            chunk_type=CHUNK_ABSTRACT, title="Attention Paper",
            citation_count=100, primary_category="cs.CL",
            embedding=[0.5] * 384,
        )
        chroma_store.upsert_chunk(chunk)
        results = chroma_store.query(
            query_embedding=[0.5] * 384, top_k=5,
        )
        assert len(results) >= 1
        r = results[0]
        assert r.arxiv_id == "2106.00001"
        assert r.chunk_type == CHUNK_ABSTRACT
        assert r.title == "Attention Paper"

    def test_query_with_metadata_filter(self, chroma_store):
        for i in range(3):
            chroma_store.upsert_chunk(Chunk(
                id=ChromaStore.chunk_id(f"2106.{i:05d}", CHUNK_ABSTRACT),
                document=f"abstract {i}", arxiv_id=f"2106.{i:05d}",
                chunk_type=CHUNK_ABSTRACT, title=f"Paper {i}",
                primary_category="cs.CL" if i < 2 else "cs.LG",
                embedding=[0.1 * i] * 384,
            ))
        results = chroma_store.query(
            query_embedding=[0.1] * 384, top_k=10,
            where={"primary_category": "cs.CL"},
        )
        assert all(r.primary_category == "cs.CL" for r in results)
        assert len(results) == 2

    def test_get_by_arxiv_id(self, chroma_store):
        for ct in [CHUNK_ABSTRACT, CHUNK_TITLE]:
            chroma_store.upsert_chunk(Chunk(
                id=ChromaStore.chunk_id("2106.00001", ct),
                document="text", arxiv_id="2106.00001",
                chunk_type=ct, title="T",
                embedding=[0.1] * 384,
            ))
        chunks = chroma_store.get_by_arxiv_id("2106.00001")
        assert len(chunks) == 2
        types = {c.chunk_type for c in chunks}
        assert types == {CHUNK_ABSTRACT, CHUNK_TITLE}

    def test_delete_by_arxiv_id(self, chroma_store):
        chroma_store.upsert_chunk(Chunk(
            id=ChromaStore.chunk_id("2106.00001", CHUNK_ABSTRACT),
            document="text", arxiv_id="2106.00001",
            chunk_type=CHUNK_ABSTRACT, title="T",
            embedding=[0.1] * 384,
        ))
        assert chroma_store.count() == 1
        chroma_store.delete_by_arxiv_id("2106.00001")
        assert chroma_store.count() == 0

    def test_reset(self, chroma_store):
        chroma_store.upsert_chunk(Chunk(
            id=ChromaStore.chunk_id("2106.00001", CHUNK_ABSTRACT),
            document="text", arxiv_id="2106.00001",
            chunk_type=CHUNK_ABSTRACT, title="T",
            embedding=[0.1] * 384,
        ))
        assert chroma_store.count() == 1
        chroma_store.reset()
        assert chroma_store.count() == 0

    def test_upsert_empty_batch(self, chroma_store):
        chroma_store.upsert_chunks([])
        assert chroma_store.count() == 0

    def test_upsert_updates_existing(self, chroma_store):
        cid = ChromaStore.chunk_id("2106.00001", CHUNK_ABSTRACT)
        chroma_store.upsert_chunk(Chunk(
            id=cid, document="v1", arxiv_id="2106.00001",
            chunk_type=CHUNK_ABSTRACT, title="T", embedding=[0.1] * 384,
        ))
        chroma_store.upsert_chunk(Chunk(
            id=cid, document="v2", arxiv_id="2106.00001",
            chunk_type=CHUNK_ABSTRACT, title="T", embedding=[0.1] * 384,
        ))
        assert chroma_store.count() == 1
        chunks = chroma_store.get_by_arxiv_id("2106.00001")
        assert chunks[0].document == "v2"
