"""Capture raw Qwen2.5 outputs from the real benchmark path.

Run after `python -m tests.benchmark.run_all --real --no-bb` to
piggy-back on the live LLM. Picks one paper per family and writes
the raw LLM strings (summary JSON, ideas JSON, novelty verdict,
extraction schema) to disk so the generation *quality* can be
inspected by hand, not just the metrics.

Usage:
  python tools/capture_real_llm_samples.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.bench_runner import build_env
from tests.benchmark.synthetic import NOVELTY_PAIRS, SYNTHETIC_LIBRARY

SAMPLES_PATH = PROJECT_ROOT / "tests" / "benchmark" / "real_llm_samples.json"
DOCS_PATH = PROJECT_ROOT / "docs" / "REAL_LLM_BENCHMARK_SAMPLES.md"


def _capture_summary_direct(ctx, paper) -> dict:
    """Capture summary by routing through the prompts builder directly
    to retain the raw LLM string."""
    from src.services.prompts import SUMMARY_SECTIONS, extract_json, summary_messages

    msgs = summary_messages(paper["title"], paper["abstract"], SUMMARY_SECTIONS)
    t0 = time.perf_counter()
    raw = ctx.models.llm.chat(msgs, temperature=0.0, max_tokens=600)
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = None
    return {"raw": raw, "parsed": parsed, "wall_ms": ms}


def _capture_ideas_direct(ctx, paper) -> dict:
    from src.services.prompts import extract_json, idea_messages

    msgs = idea_messages(paper["title"], paper["abstract"], None, num_ideas=3,
                          focus_area="diverse")
    t0 = time.perf_counter()
    raw = ctx.models.llm.chat(msgs, temperature=0.0, max_tokens=800)
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = None
    return {"raw": raw, "parsed": parsed, "wall_ms": ms}


def _capture_novelty_direct(ctx, pair) -> dict:
    from src.services.prompts import extract_json, novelty_messages
    candidate = next(sp for sp in SYNTHETIC_LIBRARY
                     if sp["arxiv_id"] == pair["candidate_arxiv_id"])
    msgs = novelty_messages(pair["idea_text"], candidate["title"], candidate["abstract"])
    t0 = time.perf_counter()
    raw = ctx.models.llm.chat(msgs, temperature=0.0, max_tokens=192)
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = None
    return {"raw": raw, "parsed": parsed, "wall_ms": ms,
            "expected_verdict": pair["expected_verdict"]}


def _capture_extraction_direct(ctx, paper) -> dict:
    from src.services.prompts import extract_json, extract_messages

    msgs = extract_messages(paper["title"], paper["abstract"], paper["arxiv_id"])
    t0 = time.perf_counter()
    raw = ctx.models.llm.chat(msgs, temperature=0.0, max_tokens=512)
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = None
    return {"raw": raw, "parsed": parsed, "wall_ms": ms}


def main() -> int:
    print("Building env with real local LLM (cold load)...", flush=True)
    from src.inference.llm import LocalLLM
    from src.config import get_settings

    s = get_settings()
    llm = LocalLLM(model_path=s.llm_model_path, n_ctx=s.llm_n_ctx,
                    n_threads=s.llm_n_threads, n_gpu_layers=s.llm_n_gpu_layers)
    t_load = time.perf_counter()
    llm._ensure_loaded()
    print(f"  model loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    env = build_env(stub_llm=llm)

    papers_by_id = {p["arxiv_id"]: p for p in SYNTHETIC_LIBRARY}
    picked = [
        "2401.00001",
        "2405.00005",
        "2309.01234",
    ]

    samples: dict = {"metadata": {
        "model": s.llm_model_file,
        "n_ctx": s.llm_n_ctx,
        "n_threads": s.llm_n_threads,
        "n_gpu_layers": s.llm_n_gpu_layers,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    print("Capturing summary samples...", flush=True)
    samples["summaries"] = {}
    for arxiv_id in picked:
        s_obj = _capture_summary_direct(env.ctx, papers_by_id[arxiv_id])
        samples["summaries"][arxiv_id] = s_obj
        print(f"  {arxiv_id} -> {s_obj['wall_ms']}ms "
              f"(parse={'ok' if s_obj['parsed'] else 'FAIL'})", flush=True)

    print("Capturing idea samples...", flush=True)
    samples["ideas"] = {}
    for arxiv_id in picked:
        s_obj = _capture_ideas_direct(env.ctx, papers_by_id[arxiv_id])
        samples["ideas"][arxiv_id] = s_obj
        print(f"  {arxiv_id} -> {s_obj['wall_ms']}ms "
              f"(parse={'ok' if s_obj['parsed'] else 'FAIL'})", flush=True)

    print("Capturing novelty samples...", flush=True)
    samples["novelty"] = []
    for pair in NOVELTY_PAIRS:
        s_obj = _capture_novelty_direct(env.ctx, pair)
        samples["novelty"].append({"pair": pair, "output": s_obj})
        verdict = s_obj["parsed"]["verdict"] if s_obj["parsed"] else "PARSE_FAIL"
        print(f"  {pair['kind']} -> {verdict} "
              f"(expected={pair['expected_verdict']})", flush=True)

    print("Capturing extraction samples...", flush=True)
    samples["extraction"] = {}
    for arxiv_id in picked:
        s_obj = _capture_extraction_direct(env.ctx, papers_by_id[arxiv_id])
        samples["extraction"][arxiv_id] = s_obj
        print(f"  {arxiv_id} -> {s_obj['wall_ms']}ms "
              f"(parse={'ok' if s_obj['parsed'] else 'FAIL'})", flush=True)

    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES_PATH.write_text(json.dumps(samples, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {SAMPLES_PATH}")

    md_lines = _render_markdown(samples)
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {DOCS_PATH}")

    env.close()
    return 0


def _render_markdown(samples: dict) -> list[str]:
    md: list[str] = []
    m = samples["metadata"]
    md.append("# Real-LLM Benchmark Samples")
    md.append("")
    md.append(f"Captured from `{m['model']}` (n_ctx={m['n_ctx']}, "
              f"n_threads={m['n_threads']}, n_gpu_layers={m['n_gpu_layers']}) "
              f"on {m['generated_at']}.")
    md.append("")
    md.append("Each panel shows the raw LLM string (verbatim) plus the JSON "
              "parser's view. Use these to inspect what the 1.5B model "
              "actually produces end-to-end before you tune prompts.")
    md.append("")

    md.append("## Summary samples")
    md.append("")
    for arxiv_id, s_obj in samples["summaries"].items():
        md.append(f"### `{arxiv_id}` (wall {s_obj['wall_ms']} ms)")
        md.append("")
        md.append("**Raw:**")
        md.append("")
        md.append("```text")
        md.append(s_obj["raw"].strip())
        md.append("```")
        md.append("")
        md.append("**Parsed:**")
        md.append("")
        md.append("```json")
        md.append(json.dumps(s_obj["parsed"], indent=2, ensure_ascii=False))
        md.append("```")
        md.append("")

    md.append("## Idea samples")
    md.append("")
    for arxiv_id, s_obj in samples["ideas"].items():
        md.append(f"### `{arxiv_id}` (wall {s_obj['wall_ms']} ms)")
        md.append("")
        md.append("**Raw:**")
        md.append("")
        md.append("```text")
        md.append(s_obj["raw"].strip())
        md.append("```")
        md.append("")
        md.append("**Parsed:**")
        md.append("")
        md.append("```json")
        md.append(json.dumps(s_obj["parsed"], indent=2, ensure_ascii=False))
        md.append("```")
        md.append("")

    md.append("## Novelty samples")
    md.append("")
    md.append("| Kind | Expected | Predicted | OK | Wall (ms) |")
    md.append("|---|---|---|---|---|")
    for row in samples["novelty"]:
        pair = row["pair"]
        out = row["output"]
        pred = out["parsed"]["verdict"] if out["parsed"] else "PARSE_FAIL"
        ok = pred == pair["expected_verdict"]
        md.append(f"| {pair['kind']} | {pair['expected_verdict']} | {pred} | "
                  f"{'✅' if ok else '❌'} | {out['wall_ms']} |")
    md.append("")
    for row in samples["novelty"]:
        pair = row["pair"]
        out = row["output"]
        md.append(f"### Idea '{pair['idea_text'][:60]}...' vs `{pair['candidate_arxiv_id']}`")
        md.append("")
        md.append("**Raw:**")
        md.append("")
        md.append("```text")
        md.append(out["raw"].strip())
        md.append("```")
        md.append("")

    md.append("## Extraction samples")
    md.append("")
    for arxiv_id, s_obj in samples["extraction"].items():
        md.append(f"### `{arxiv_id}` (wall {s_obj['wall_ms']} ms)")
        md.append("")
        md.append("**Raw:**")
        md.append("")
        md.append("```text")
        md.append(s_obj["raw"].strip())
        md.append("```")
        md.append("")
        md.append("**Parsed:**")
        md.append("")
        md.append("```json")
        md.append(json.dumps(s_obj["parsed"], indent=2, ensure_ascii=False))
        md.append("```")
        md.append("")

    return md


if __name__ == "__main__":
    raise SystemExit(main())
