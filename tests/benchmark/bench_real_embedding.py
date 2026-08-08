"""Real-embedding ranking benchmark.

Hash pseudo-embeddings used in the offline mocked suite cap NDCG
quality. This benchmark probes the real fallback chain:

  specter2_base (768-d)  -> bge-large-en-v1.5 (1024-d)  ->
  bge-small-en-v1.5 (384-d) -> hash-pseudo (384-d, baseline)

For each embedder that successfully loads, encode the synthetic
library, encode each graded query, compute cosine NDCG@5 / NDCG@10 /
P@5 / R@5 / MRR, and compare to published reference numbers
(BEIR / SciFact).

Embedders that fail to load (missing weights, OOM, network blocked)
are skipped with a status row in the report — they are NOT treated
as failures, because the benchmark is meant to work in any
deployment. The hash embedder is always present as a baseline so
the table has a non-empty floor.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from typing import Sequence

from tests.benchmark.metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from tests.benchmark.synthetic import GRADED_QUERIES, SYNTHETIC_LIBRARY


@dataclass(frozen=True)
class EmbedderSpec:
    """A real embedder candidate plus its published BEIR reference."""

    name: str
    hf_id: str
    dim: int
    approx_size_mb: int
    cpu_estimate_s: float
    ref_scifact_ndcg10: float
    ref_trec_covid_ndcg10: float
    notes: str


EMBEDDER_FALLBACK_CHAIN: list[EmbedderSpec] = [
    EmbedderSpec(
        name="SPECTER2 base",
        hf_id="allenai/specter2_base",
        dim=768,
        approx_size_mb=440,
        cpu_estimate_s=8.0,
        ref_scifact_ndcg10=0.708,
        ref_trec_covid_ndcg10=0.654,
        notes="Triplet-loss fine-tune of SciBERT on citation triplets; title+abstract concatenation.",
    ),
    EmbedderSpec(
        name="BGE-large",
        hf_id="BAAI/bge-large-en-v1.5",
        dim=1024,
        approx_size_mb=1300,
        cpu_estimate_s=15.0,
        ref_scifact_ndcg10=0.708,
        ref_trec_covid_ndcg10=0.781,
        notes="BAAI flagship English embedder; 1024-d version of BGE.",
    ),
    EmbedderSpec(
        name="BGE-small",
        hf_id="BAAI/bge-small-en-v1.5",
        dim=384,
        approx_size_mb=130,
        cpu_estimate_s=4.0,
        ref_scifact_ndcg10=0.626,
        ref_trec_covid_ndcg10=0.632,
        notes="Current default. Fast on CPU; rank drops 4-8 NDCG@10 vs BGE-large.",
    ),
]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na * nb else 0.0


def _hash_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic hash pseudo-embedder (always available)."""
    h = hashlib.sha256(text.encode()).digest()
    data = (h * 12)[:dim]
    vec = [(b / 255.0 - 0.5) * 2 for b in data]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _try_load_embedder(spec: EmbedderSpec) -> object | None:
    """Load a real embedder; return None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "models"))
        st = SentenceTransformer(spec.hf_id)
    except Exception:
        return None
    return st


def _st_embed(st, texts: Sequence[str]) -> list[list[float]]:
    vecs = st.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


@dataclass
class EmbedderRun:
    spec: EmbedderSpec
    loaded: bool
    error: str | None
    encode_time_ms: float
    ndcg_5: float
    ndcg_10: float
    precision_5: float
    recall_5: float
    mrr_value: float
    ref_scifact_ndcg10: float
    ref_trec_covid_ndcg10: float


def _doc_text(p: dict) -> str:
    title = p.get("title", "")
    abstract = p.get("abstract", "")
    return f"{title} [SEP] {abstract}".strip()


def _evaluate(
    doc_vecs: list[list[float]],
    queries: list[dict],
) -> dict:
    ndcg_5_total = 0.0
    ndcg_10_total = 0.0
    p_5_total = 0.0
    r_5_total = 0.0
    mrr_total = 0.0
    n = len(queries)
    for q in queries:
        query = q["query"]
        gold_rel = q.get("gold_relevance") or {aid: 1 for aid in q.get("gold_arxiv_ids", [])}
        if not gold_rel:
            continue
        qvec = _hash_embed(query)
        scored = [
            (_cosine(qvec, dv), idx) for idx, dv in enumerate(doc_vecs)
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        retrieved_ids = [SYNTHETIC_LIBRARY[idx]["arxiv_id"] for _, idx in scored]
        gold_set = set(gold_rel.keys())
        ndcg_5_total += ndcg_at_k(retrieved_ids, gold_rel, 5)
        ndcg_10_total += ndcg_at_k(retrieved_ids, gold_rel, 10)
        p_5_total += precision_at_k(retrieved_ids, gold_set, 5)
        r_5_total += recall_at_k(retrieved_ids, gold_set, 5)
        mrr_total += mrr(retrieved_ids, gold_set)
    return {
        "ndcg@5": ndcg_5_total / max(1, n),
        "ndcg@10": ndcg_10_total / max(1, n),
        "p@5": p_5_total / max(1, n),
        "r@5": r_5_total / max(1, n),
        "mrr": mrr_total / max(1, n),
        "queries": n,
    }


def bench_real_embedding(
    include_chain: bool = True,
    force_hash_only: bool = False,
) -> dict:
    """Run the embedding fallback-chain benchmark.

    Returns a dict with per-embedder ``runs`` and aggregate
    ``summary`` records ready for markdown rendering.
    """
    doc_texts = [_doc_text(p) for p in SYNTHETIC_LIBRARY]
    runs: list[EmbedderRun] = []

    hash_vecs = [_hash_embed(t) for t in doc_texts]
    eval_metrics = _evaluate(hash_vecs, GRADED_QUERIES)
    runs.append(
        EmbedderRun(
            spec=EmbedderSpec(
                name="Hash (pseudo)",
                hf_id="hash-pseudo",
                dim=384,
                approx_size_mb=0,
                cpu_estimate_s=0.0,
                ref_scifact_ndcg10=0.0,
                ref_trec_covid_ndcg10=0.0,
                notes="Deterministic baseline; not a real embedding.",
            ),
            loaded=True,
            error=None,
            encode_time_ms=0.0,
            ndcg_5=eval_metrics["ndcg@5"],
            ndcg_10=eval_metrics["ndcg@10"],
            precision_5=eval_metrics["p@5"],
            recall_5=eval_metrics["r@5"],
            mrr_value=eval_metrics["mrr"],
            ref_scifact_ndcg10=0.0,
            ref_trec_covid_ndcg10=0.0,
        )
    )

    if not force_hash_only and include_chain:
        for spec in EMBEDDER_FALLBACK_CHAIN:
            t0 = time.perf_counter()
            st = _try_load_embedder(spec)
            encode_ms = 0.0
            if st is None:
                runs.append(
                    EmbedderRun(
                        spec=spec,
                        loaded=False,
                        error="weights unavailable or load failed",
                        encode_time_ms=0.0,
                        ndcg_5=0.0,
                        ndcg_10=0.0,
                        precision_5=0.0,
                        recall_5=0.0,
                        mrr_value=0.0,
                        ref_scifact_ndcg10=spec.ref_scifact_ndcg10,
                        ref_trec_covid_ndcg10=spec.ref_trec_covid_ndcg10,
                    )
                )
                continue
            try:
                doc_vecs = _st_embed(st, doc_texts)
                encode_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                runs.append(
                    EmbedderRun(
                        spec=spec,
                        loaded=False,
                        error=f"encode failed: {type(exc).__name__}: {exc}",
                        encode_time_ms=(time.perf_counter() - t0) * 1000.0,
                        ndcg_5=0.0,
                        ndcg_10=0.0,
                        precision_5=0.0,
                        recall_5=0.0,
                        mrr_value=0.0,
                        ref_scifact_ndcg10=spec.ref_scifact_ndcg10,
                        ref_trec_covid_ndcg10=spec.ref_trec_covid_ndcg10,
                    )
                )
                continue

            ndcg_5 = 0.0
            ndcg_10 = 0.0
            p_5 = 0.0
            r_5 = 0.0
            mrr_total = 0.0
            for q in GRADED_QUERIES:
                gold_rel = q.get("gold_relevance") or {aid: 1 for aid in q.get("gold_arxiv_ids", [])}
                if not gold_rel:
                    continue
                qvec = _st_embed(st, [q["query"]])[0]
                scored = [
                    (_cosine(qvec, dv), idx) for idx, dv in enumerate(doc_vecs)
                ]
                scored.sort(key=lambda t: t[0], reverse=True)
                retrieved_ids = [SYNTHETIC_LIBRARY[idx]["arxiv_id"] for _, idx in scored]
                gold_set = set(gold_rel.keys())
                ndcg_5 += ndcg_at_k(retrieved_ids, gold_rel, 5)
                ndcg_10 += ndcg_at_k(retrieved_ids, gold_rel, 10)
                p_5 += precision_at_k(retrieved_ids, gold_set, 5)
                r_5 += recall_at_k(retrieved_ids, gold_set, 5)
                mrr_total += mrr(retrieved_ids, gold_set)
            n_queries = max(1, len(GRADED_QUERIES))
            runs.append(
                EmbedderRun(
                    spec=spec,
                    loaded=True,
                    error=None,
                    encode_time_ms=encode_ms,
                    ndcg_5=ndcg_5 / n_queries,
                    ndcg_10=ndcg_10 / n_queries,
                    precision_5=p_5 / n_queries,
                    recall_5=r_5 / n_queries,
                    mrr_value=mrr_total / n_queries,
                    ref_scifact_ndcg10=spec.ref_scifact_ndcg10,
                    ref_trec_covid_ndcg10=spec.ref_trec_covid_ndcg10,
                )
            )
            del st

    summary = {
        "runs": [
            {
                "name": r.spec.name,
                "hf_id": r.spec.hf_id,
                "dim": r.spec.dim,
                "size_mb": r.spec.approx_size_mb,
                "loaded": r.loaded,
                "error": r.error,
                "encode_ms": round(r.encode_time_ms, 1),
                "ndcg@5": round(r.ndcg_5, 4),
                "ndcg@10": round(r.ndcg_10, 4),
                "p@5": round(r.precision_5, 4),
                "r@5": round(r.recall_5, 4),
                "mrr": round(r.mrr_value, 4),
                "ref_scifact_ndcg10": r.ref_scifact_ndcg10,
                "ref_trec_covid_ndcg10": r.ref_trec_covid_ndcg10,
                "notes": r.spec.notes,
            }
            for r in runs
        ],
        "queries_evaluated": len(GRADED_QUERIES),
        "corpus_size": len(SYNTHETIC_LIBRARY),
    }
    return summary
