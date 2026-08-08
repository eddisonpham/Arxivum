"""Synthetic datasets used by the benchmark suite.

These are small, deterministic fixtures used to exercise the
retrieval, summarization, ideas, and novelty benchmarks offline
(no arXiv API call, no Semantic Scholar call). They are NOT a gold
benchmark dataset. Their purpose is to give every metric a
non-zero signal so that pipeline regressions are caught before
they reach a real evaluation.

The fixtures mimic arXiv-shaped records. Fields: arxiv_id, title,
abstract, primary_category, authors, citation_count, venue,
published.
"""

from __future__ import annotations


SYNTHETIC_LIBRARY: list[dict] = [
    {
        "arxiv_id": "2401.00001",
        "title": "Scaling laws for mixture-of-experts routing under sparse activation",
        "abstract": "We study scaling behaviour of mixture-of-experts (MoE) transformer variants with sparse top-k routing. Empirically, model loss follows a power law in total parameters with slope near -0.05 across compute budgets from 1e19 to 1e22 FLOPs. We show routing entropy plateaus when expert count exceeds a critical density, and propose a regulariser that prevents expert collapse. Results on language modelling and code benchmarks show consistent improvements over dense baselines at matched FLOPS.",
        "primary_category": "cs.LG",
        "authors": ["A. N. Other", "B. Researcher"],
        "citation_count": 42,
        "venue": "NeurIPS",
        "published": "2024-01-02",
    },
    {
        "arxiv_id": "2402.00010",
        "title": "A benchmark for hyperparameter sensitivity in deep learning",
        "abstract": "We propose a sensitivity benchmark for hyperparameters in deep learning. The benchmark measures how loss surfaces wobble under perturbation of learning rate, batch size, momentum, and weight decay. Across 12 representative workloads, we observe that learning rate remains the most influential single factor, with second-order effects dominated by momentum. We release the harness under MIT licence.",
        "primary_category": "cs.LG",
        "authors": ["C. Tune"],
        "citation_count": 9,
        "venue": "ICLR Workshop",
        "published": "2024-02-15",
    },
    {
        "arxiv_id": "2403.00099",
        "title": "Retrieval-augmented generation with sparse lexical expansion",
        "abstract": "We extend retrieval-augmented generation (RAG) pipelines with a sparse lexical expansion step that broadens the candidate pool beyond dense top-k. On five open-domain QA datasets we find a 4-7 point exact-match improvement at the same retrieval budget. The largest gains come when the dense retriever fails on rare-entity queries.",
        "primary_category": "cs.CL",
        "authors": ["D. Retrieve", "E. Augment"],
        "citation_count": 17,
        "venue": "ACL Findings",
        "published": "2024-03-22",
    },
    {
        "arxiv_id": "2404.01000",
        "title": "Citation graph features for scientific paper similarity",
        "abstract": "We study whether direct and second-order citation graph features can augment dense embeddings for scientific paper similarity. Combining a SPECTER-style dense embedding with bibliographic coupling and co-citation features gives NDCG@10 improvements of 4-9 points on a held-out similarity judgement set.",
        "primary_category": "cs.IR",
        "authors": ["F. Graph", "G. Cite"],
        "citation_count": 5,
        "venue": "SIGIR",
        "published": "2024-04-30",
    },
    {
        "arxiv_id": "2405.00005",
        "title": "An empirical study of factuality hallucination in abstractive summarisation",
        "abstract": "We analyse factuality hallucinations in abstractive summarisation models. Using QAGS-style question-answer overlap as a hallucination probe, we find that even strong abstractive models introduce unsupported claims at rates of 12-18 percent. We propose a constrained decoding strategy that reduces this rate by approximately half.",
        "primary_category": "cs.CL",
        "authors": ["H. Fact", "I. Check"],
        "citation_count": 21,
        "venue": "EMNLP",
        "published": "2024-05-11",
    },
    {
        "arxiv_id": "2406.00007",
        "title": "Long-context attention with linear-time kernels",
        "abstract": "We introduce a long-context attention variant based on linear-time kernel approximations that scales to 1M tokens with sub-quadratic cost. On long-document summarisation and code completion we match dense attention accuracy within 1 point while reducing wall-time latency 6x on H100.",
        "primary_category": "cs.LG",
        "authors": ["J. Linear"],
        "citation_count": 30,
        "venue": "ICML",
        "published": "2024-06-04",
    },
    {
        "arxiv_id": "2309.01234",
        "title": "RAGAS: Automated evaluation of retrieval augmented generation",
        "abstract": "We propose RAGAS, a reference-free evaluation framework for RAG pipelines that measures faithfulness, answer relevancy, context precision, and context recall without ground-truth labels. We correlate each metric with human judgements across nine datasets and find Spearman rho in the 0.6-0.85 range.",
        "primary_category": "cs.CL",
        "authors": ["J. James", "L. Espinosa-Anke"],
        "citation_count": 220,
        "venue": "arXiv",
        "published": "2023-09-04",
    },
]


SYNTHETIC_QUERIES: list[dict] = [
    {
        "query": "are mixture of experts models scalable?",
        "gold_arxiv_ids": ["2401.00001"],
        "gold_relevance": {"2401.00001": 2},
    },
    {
        "query": "detect hallucinations in generated summaries",
        "gold_arxiv_ids": ["2405.00005", "2309.01234"],
        "gold_relevance": {"2405.00005": 3, "2309.01234": 2},
    },
    {
        "query": "attention for long context inputs",
        "gold_arxiv_ids": ["2406.00007"],
        "gold_relevance": {"2406.00007": 3},
    },
    {
        "query": "combine citation graph features with embeddings for paper retrieval",
        "gold_arxiv_ids": ["2404.01000", "2309.01234"],
        "gold_relevance": {"2404.01000": 3, "2309.01234": 1},
    },
]


NOVELTY_PAIRS: list[dict] = [
    {
        "kind": "novel",
        "idea_text": "Train a small student model to imitate a mixture-of-experts teacher with shared routing logic, allowing cheaper inference.",
        "candidate_arxiv_id": "2402.00010",
        "expected_verdict": "likely_novel",
    },
    {
        "kind": "novel",
        "idea_text": "Use linear kernel attention to extend MoE routing context to 1M tokens for global document modelling.",
        "candidate_arxiv_id": "2401.00001",
        "expected_verdict": "needs_review",
    },
    {
        "kind": "similar",
        "idea_text": "Combine citation graph coupling features with dense embeddings to get better paper similarity scores.",
        "candidate_arxiv_id": "2404.01000",
        "expected_verdict": "similar_exists",
    },
    {
        "kind": "similar",
        "idea_text": "Use QAGS-style question-answering overlap to detect unsupported claims in generated summaries.",
        "candidate_arxiv_id": "2405.00005",
        "expected_verdict": "similar_exists",
    },
]
