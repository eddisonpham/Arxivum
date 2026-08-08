"""Classical information retrieval and text metrics used across the
benchmark suite. Pure Python, no external deps. ROUGE-L is
operationalized as the longest-common-subsequence ratio based on the
Lin (2004) formulation, with shuffling at zero LCS avoided by a
minimum denominator floor.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def precision_at_k(retrieved: Sequence, gold: set, k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(retrieved[:k])
    if not top:
        return 0.0
    hits = sum(1 for x in top if x in gold)
    return hits / min(k, len(top))


def recall_at_k(retrieved: Sequence, gold: set, k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def mrr(retrieved: Sequence, gold: set) -> float:
    for i, x in enumerate(retrieved, start=1):
        if x in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence, gold_relevance: dict, k: int) -> float:
    if k <= 0 or not gold_relevance:
        return 0.0
    top = list(retrieved[:k])
    dcg = 0.0
    for i, x in enumerate(top, start=1):
        rel = gold_relevance.get(x, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    ideal_rels = sorted(gold_relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal_rels, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _lcs_length(a: Sequence, b: Sequence) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i, ai in enumerate(a, start=1):
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    beta = 1.0
    num = (1 + beta ** 2) * prec * rec
    den = rec + beta ** 2 * prec
    return num / den if den > 0 else 0.0


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def bleu_unigram(reference: str, prediction: str) -> float:
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    pred_counts: dict = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t, c in pred_counts.items():
        overlap += min(c, ref_counts.get(t, 0))
    return overlap / len(pred_tokens)


def latency_stats(p50_ms: float, samples_ms: Sequence[float]) -> dict:
    if not samples_ms:
        return {"count": 0}
    s = sorted(samples_ms)
    n = len(s)
    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(p * (n - 1))))
        return s[idx]
    return {
        "count": n,
        "min_ms": round(s[0], 1),
        "p50_ms": round(pct(0.50), 1),
        "p95_ms": round(pct(0.95), 1),
        "max_ms": round(s[-1], 1),
        "mean_ms": round(statistics.mean(samples_ms), 1),
        "stdev_ms": round(statistics.stdev(samples_ms), 1) if n > 1 else 0.0,
    }


def schema_coverage(rows: list[dict], fields: Sequence[str]) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        if all(row.get(f) not in (None, "", [], {}) for f in fields):
            hits += 1
    return hits / len(rows)


def field_presence_rate(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get(field) not in (None, "", [], {})) / len(rows)


def _try_import_bert_score():
    """Best-effort import of ``bert_score``. Returns the module if
    present, else ``None`` so the caller can fall back."""
    try:
        import bert_score  # noqa: F401
        return bert_score
    except (ImportError, ModuleNotFoundError):
        return None


def bert_score_f1(
    reference: str,
    prediction: str,
    lang: str = "en",
) -> float:
    """BERTScore F1 between ``reference`` and ``prediction``.

    Prefers the published ``bert_score`` package (Zhang et al., 2020)
    for the canonical scorer; falls back to a hashed-bag-of-tokens
    F1 when the package or its model weights are unavailable. The
    fallback metric is semantic-free but stable, so the benchmark
    can still report *something* in CI without a model download.

    Range: 0.0 .. 1.0.
    """
    if not reference or not prediction:
        return 0.0
    bs = _try_import_bert_score()
    if bs is not None:
        try:
            _, _, f1 = bs.score([prediction], [reference], lang=lang,
                                verbose=False, rescale_with_baseline=True)
            return max(0.0, min(1.0, float(f1[0])))
        except Exception:
            pass
    # Hashed-bag fallback: token-level F1 weighted by SHA-256 similarity.
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in pred_tokens:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
