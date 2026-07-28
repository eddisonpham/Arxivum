"""Full pipeline demo — exercises every research tool end-to-end.

Run the visual panel first in a separate terminal:

    python -m src.api.main

Then run this script:

    python scripts/demo.py

Every action is logged to the activity log, so you can watch it happen
in real time at http://localhost:8000/activity.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app import create_app, shutdown_app
from src.config import get_settings


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main() -> int:
    settings = get_settings()
    settings.ensure_dirs()

    banner("Starting ArXivum pipeline demo")
    print(f"Database: {settings.db_path}")
    print(f"Models:   {settings.models_dir}")
    print(f"LLM:      {settings.llm_model_file}")

    ctx = create_app()
    lib = ctx.library
    summarizer = ctx.summarizer
    ideas_svc = ctx.ideas
    novelty_svc = ctx.novelty

    try:
        # ── Step 1: Search arXiv ──────────────────────────────────────
        banner("Step 1: Search arXiv")
        topic = "graph neural networks molecule property prediction"
        print(f"Query: {topic}")
        print("Searching arXiv (rate-limited, ~3s)...")
        results = lib.search_and_import(
            query=topic,
            max_results=3,
            auto_enrich=True,
        )
        print(f"\nImported {len(results)} papers:")
        for r in results:
            print(f"  - [{r.arxiv_id}] {r.title}")
            print(f"    Citations: {r.citation_count}  Venue: {r.venue or 'N/A'}")

        if not results:
            print("No results. Exiting.")
            return 1

        target_id = results[0].arxiv_id
        print(f"\nTarget paper for deep analysis: {target_id}")

        # ── Step 2: Summarize ─────────────────────────────────────────
        banner("Step 2: Generate structured summary")
        print(f"Summarizing {target_id}...")
        print("(Loading local LLM on first call, may take 10-30s)")
        t0 = time.time()
        summaries = summarizer.summarize(target_id)
        elapsed = time.time() - t0
        print(f"\nGenerated {len(summaries)} sections in {elapsed:.1f}s:")
        for section, content in summaries.items():
            preview = content[:120].replace('\n', ' ')
            print(f"  [{section}]: {preview}...")

        # ── Step 3: Generate ideas ────────────────────────────────────
        banner("Step 3: Generate research ideas")
        print(f"Generating ideas from {target_id}...")
        t0 = time.time()
        ideas = ideas_svc.generate_ideas(target_id, num_ideas=2, focus_area="methodological")
        elapsed = time.time() - t0
        print(f"\nGenerated {len(ideas)} ideas in {elapsed:.1f}s:")
        for idea in ideas:
            print(f"  [Idea #{idea['id']}] {idea.get('title', '(untitled)')}")
            print(f"    {idea['idea_text'][:150]}")
            print(f"    Search queries: {idea['search_queries']}")
            print()

        # ── Step 4: Verify novelty ────────────────────────────────────
        if ideas:
            banner("Step 4: Verify novelty")
            idea_id = ideas[0]["id"]
            print(f"Verifying novelty of idea #{idea_id}...")
            print("(Searches local library + arXiv, then LLM judges overlap)")
            t0 = time.time()
            verdict = novelty_svc.verify_novelty(idea_id)
            elapsed = time.time() - t0
            print(f"\nVerdict: {verdict['verdict'].upper()}")
            print(f"Notes: {verdict['notes']}")
            print(f"Similar papers found: {verdict['similar_arxiv_ids']}")
            print(f"Query terms used: {verdict['query_terms']}")
            print(f"Time: {elapsed:.1f}s")

        # ── Step 5: Query the library ─────────────────────────────────
        banner("Step 5: Query the library")
        query = "molecular property prediction with attention"
        print(f"Query: {query}")
        print("(Hybrid vector search + cross-encoder reranking)")
        t0 = time.time()
        query_results = lib.query_library(query, top_k=5, rerank=True)
        elapsed = time.time() - t0
        print(f"\nTop {len(query_results)} results in {elapsed:.1f}s:")
        for i, r in enumerate(query_results, 1):
            print(f"  {i}. [{r.arxiv_id}] {r.title}")
            print(f"     Score: {r.score:.4f}  Citations: {r.citation_count}")
            print(f"     Snippet: {r.abstract_snippet[:100]}...")

        # ── Step 6: Paper details ─────────────────────────────────────
        banner("Step 6: Get paper details")
        detail = lib.get_paper_detail(target_id)
        if detail:
            print(f"Title: {detail['title']}")
            print(f"Authors: {', '.join(detail['authors'][:3])}{'...' if len(detail['authors']) > 3 else ''}")
            print(f"Categories: {detail['categories']}")
            metrics = detail.get('metrics')
            if metrics:
                print(f"Citations: {metrics.get('citation_count', 'N/A')}")
                print(f"Venue: {metrics.get('venue', 'N/A')}")
            print(f"Summaries: {len(detail['summaries'])} sections cached")
            print(f"Ideas: {len(detail['ideas'])} generated")

        # ── Step 7: Activity log ──────────────────────────────────────
        banner("Step 7: Activity log")
        activities = ctx.db.list_activity(limit=20)
        print(f"Last {len(activities)} actions:")
        for a in activities:
            print(f"  [{a.created_at[:19]}] {a.action_type:10s} {a.status:10s} {a.arxiv_id or a.query or ''}")

        # ── Done ──────────────────────────────────────────────────────
        banner("Demo complete")
        print(f"Total papers in library: {ctx.db.count_papers()}")
        print(f"Total ChromaDB chunks:   {ctx.chroma.count()}")
        print(f"\nOpen http://localhost:8000 to inspect the visual panel.")
        print(f"Open http://localhost:8000/activity for the activity log.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        shutdown_app(ctx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
