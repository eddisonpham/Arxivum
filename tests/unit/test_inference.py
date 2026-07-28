"""Unit tests for the inference layer (embedder, reranker, LLM, manager)."""

import pytest

from src.inference.llm import LocalLLM, StubLLM


class TestStubLLM:
    def test_chat_returns_default(self):
        llm = StubLLM(default='{"overall": "test"}')
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert result == '{"overall": "test"}'

    def test_chat_records_calls(self):
        llm = StubLLM()
        llm.chat([{"role": "user", "content": "first"}])
        llm.chat([{"role": "user", "content": "second"}])
        assert len(llm.calls) == 2
        assert llm.calls[0][0]["content"] == "first"
        assert llm.calls[1][0]["content"] == "second"

    def test_chat_with_responder(self):
        def responder(msgs):
            return f"Response to: {msgs[0]['content']}"

        llm = StubLLM(responder=responder)
        result = llm.chat([{"role": "user", "content": "hello"}])
        assert result == "Response to: hello"

    def test_is_loaded(self):
        llm = StubLLM()
        assert llm.is_loaded is True

    def test_ensure_loaded_noop(self):
        llm = StubLLM()
        llm._ensure_loaded()  # should not raise

    def test_unload_noop(self):
        llm = StubLLM()
        llm.unload()  # should not raise


class TestStubEmbedder:
    def test_embed_returns_384d(self, stub_embedder):
        vecs = stub_embedder.embed(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 384

    def test_embed_one(self, stub_embedder):
        vec = stub_embedder.embed_one("test")
        assert len(vec) == 384

    def test_embed_deterministic(self, stub_embedder):
        v1 = stub_embedder.embed_one("same text")
        v2 = stub_embedder.embed_one("same text")
        assert v1 == v2

    def test_embed_different_texts_differ(self, stub_embedder):
        v1 = stub_embedder.embed_one("text A")
        v2 = stub_embedder.embed_one("text B")
        assert v1 != v2

    def test_dim(self, stub_embedder):
        assert stub_embedder.dim == 384

    def test_is_loaded_after_embed(self, stub_embedder):
        assert stub_embedder.is_loaded is False
        stub_embedder.embed_one("x")
        assert stub_embedder.is_loaded is True

    def test_unload(self, stub_embedder):
        stub_embedder.embed_one("x")
        assert stub_embedder.is_loaded is True
        stub_embedder.unload()
        assert stub_embedder.is_loaded is False


class TestStubReranker:
    def test_rerank_returns_indices(self, stub_reranker):
        ranked = stub_reranker.rerank("query", ["a", "b", "c"])
        assert len(ranked) == 3
        assert all(isinstance(idx, int) for idx, _ in ranked)

    def test_rerank_top_k(self, stub_reranker):
        ranked = stub_reranker.rerank("query", ["a", "b", "c"], top_k=2)
        assert len(ranked) == 2

    def test_rerank_caps_at_10(self, stub_reranker):
        candidates = [f"c{i}" for i in range(20)]
        ranked = stub_reranker.rerank("query", candidates)
        assert len(ranked) == 10

    def test_rerank_empty(self, stub_reranker):
        ranked = stub_reranker.rerank("query", [])
        assert ranked == []

    def test_is_loaded_after_rerank(self, stub_reranker):
        assert stub_reranker.is_loaded is False
        stub_reranker.rerank("q", ["a"])
        assert stub_reranker.is_loaded is True


class TestModelManager:
    def test_model_state_initial(self, model_manager):
        state = model_manager.model_state
        assert "embedder_loaded" in state
        assert "reranker_loaded" in state
        assert "llm_loaded" in state
        assert "resident" in state
        assert "constrained_memory" in state

    def test_embedder_access(self, model_manager):
        e = model_manager.embedder
        assert e is not None
        vec = e.embed_one("test")
        assert len(vec) == 384

    def test_reranker_access(self, model_manager):
        r = model_manager.reranker
        assert r is not None
        ranked = r.rerank("q", ["a", "b"])
        assert len(ranked) == 2

    def test_llm_access(self, model_manager):
        llm = model_manager.llm
        assert llm is not None
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)

    def test_set_llm(self, model_manager):
        new_stub = StubLLM(default='{"custom": true}')
        model_manager.set_llm(new_stub)
        result = model_manager.llm.chat([{"role": "user", "content": "x"}])
        assert result == '{"custom": true}'

    def test_shutdown(self, model_manager):
        model_manager.embedder.embed_one("x")
        model_manager.shutdown()
        state = model_manager.model_state
        assert state["embedder_loaded"] is False
