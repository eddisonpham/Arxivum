"""Tests for the StructuredExtractor service."""

from __future__ import annotations

import json

from src.services.extractor import EXTRACTION_SECTION, StructuredExtractor

EXTRACT_STUB_JSON = json.dumps({
    "method": "sparse mixture-of-experts transformer",
    "datasets": ["language modelling corpus", "code benchmark suite"],
    "baselines": ["dense transformer at matched FLOPS"],
    "headline_metric": {"name": "loss", "value": "slope -0.05", "split": "language + code"},
    "contribution": "Empirical scaling-law analysis of MoE routing with anti-collapse regulariser.",
    "limitations": ["language + code only", "single compute regime per model"],
    "domain": "deep learning scaling",
    "bibcode": "arXiv:2106.00001",
})


def _make_extractor(llm_response: str, db):
    from src.inference.llm import StubLLM

    class _OneShotLLM(StubLLM):
        def __init__(self, resp):
            super().__init__(default=resp)
        def chat(self, messages, **kw):
            return llm_response
    llm = _OneShotLLM(EXTRACT_STUB_JSON)
    llm._ensure_loaded = lambda: None
    from src.inference.manager import ModelManager
    mgr = ModelManager(constrained_memory=False)
    mgr.set_llm(llm)
    return StructuredExtractor(db, mgr)


def test_extract_emits_full_schema(app_context):
    extractor = _make_extractor(EXTRACT_STUB_JSON, app_context.db)
    results = app_context.library.search_and_import("attention", max_results=1)
    arxiv_id = results[0].arxiv_id
    schema = extractor.extract(arxiv_id)
    assert schema["method"]
    assert isinstance(schema["datasets"], list)
    assert "datasets" in schema and len(schema["datasets"]) == 2
    assert schema["headline_metric"]["name"] == "loss"
    assert schema["bibcode"] == f"arXiv:{arxiv_id}"


def test_extract_caches_in_summaries_table(app_context):
    extractor = _make_extractor(EXTRACT_STUB_JSON, app_context.db)
    results = app_context.library.search_and_import("attention", max_results=1)
    arxiv_id = results[0].arxiv_id
    schema_first = extractor.extract(arxiv_id)
    cached = app_context.db.get_summary(arxiv_id, EXTRACTION_SECTION)
    assert cached is not None
    assert cached.content
    parsed = json.loads(cached.content)
    assert parsed == schema_first


def test_extract_force_regenerates(app_context):
    extractor = _make_extractor(EXTRACT_STUB_JSON, app_context.db)
    results = app_context.library.search_and_import("attention", max_results=1)
    arxiv_id = results[0].arxiv_id
    extractor.extract(arxiv_id)
    extractor.extract(arxiv_id, force=True)
    cached = app_context.db.get_summary(arxiv_id, EXTRACTION_SECTION)
    assert cached is not None


def test_extract_unknown_paper_raises(app_context):
    extractor = _make_extractor(EXTRACT_STUB_JSON, app_context.db)
    try:
        extractor.extract("9999.99999")
    except ValueError as exc:
        assert "not found" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for unknown paper")


def test_normalise_handles_non_list_fields():
    from src.services.extractor import StructuredExtractor as E
    parsed = {"datasets": "CIFAR-10", "baselines": None, "limitations": "small data",
              "headline_metric": "acc=90", "method": "x", "domain": "y",
              "contribution": "z", "bibcode": ""}
    out = E._normalise(parsed, "2401.00001")
    assert isinstance(out["datasets"], list) and out["datasets"] == ["CIFAR-10"]
    assert isinstance(out["baselines"], list) and out["baselines"] == []
    assert isinstance(out["limitations"], list) and out["limitations"] == ["small data"]
    assert isinstance(out["headline_metric"], dict)
    assert out["bibcode"] == "arXiv:2401.00001"
