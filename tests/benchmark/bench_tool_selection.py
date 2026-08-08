"""Tool-selection reliability benchmark.

Quantifies the critique 5.1 claim that the 9-tool MCP surface confused
coding-agent tool selection. The benchmark constructs a synthetic set
of natural-language task descriptions, pairs each with the canonical
tool it should resolve to, and measures whether a deterministic
``StubRouter`` (mocked mode) or the real local LLM (--real mode)
picks the right one.

Output metrics:

  * ``accuracy``        — fraction of tasks with correct first-choice
  * ``confusion``       — list of mis-routed prompts (capped at 10)
  * ``surface_old``     — static inventory of the pre-refactor 9-tool
                          surface (for before/after reporting)
  * ``surface_new``     — live introspection of the current MCP server
                          (tool count, arg count, schema token estimate)
  * ``reduction``       — string form "9 -> 3" tools and "32 -> 26"
                          arguments
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass


# ── Synthetic task set (42 prompts) ─────────────────────────────────────

@dataclass
class Task:
    prompt: str
    expected_tool: str


TASKS: list[Task] = [
    # ── library_search (12) ──
    Task("Find the top 10 papers on transformer attention.", "library_search"),
    Task("Search arXiv for MoE scaling laws and import them into the library.", "library_search"),
    Task("Pull all papers from arXiv about hypercomplex networks this year.", "library_search"),
    Task("Look up the latest arXiv submission on retrieval augmentation.", "library_search"),
    Task("Import these papers from arXiv into the library, then summarise.", "library_search"),
    Task("Add a paper to my library with auto-enrich from Semantic Scholar.", "library_search"),
    Task("Get the most-cited papers on long-context attention.", "library_search"),
    Task("Bring in the recent RAG papers and extract structured metadata.", "library_search"),
    Task("Search arXiv for graph neural networks in chemistry.", "library_search"),
    Task("Find papers on hyperspectral imaging under cs.CV.", "library_search"),
    Task("Import and summarise the top three Mixture-of-Experts pre-prints.", "library_search"),
    Task("Pull any new papers on factuality hallucination since 2024.", "library_search"),

    # ── library_paper (17) ──
    Task("What does paper 2401.00001 say overall?", "library_paper"),
    Task("Summarise paper 2205.13273 in the seven canonical sections.", "library_paper"),
    Task("Generate a structured extraction schema for 2309.01234.", "library_paper"),
    Task("Propose three novel ideas grounded in 2401.00001.", "library_paper"),
    Task("Remove paper 2211.12792 from the library.", "library_paper"),
    Task("Verify the novelty of idea 42 against arXiv.", "library_paper"),
    Task("Approve idea 17 as a research direction.", "library_paper"),
    Task("Reject idea 9 — too generic.", "library_paper"),
    Task("Give me the methods section of paper 2106.00001.", "library_paper"),
    Task("Pull the abstract-extracted schema for 2406.00007.", "library_paper"),
    Task("Regenerate the summary for 2403.00099 from scratch.", "library_paper"),
    Task("Did we already extract the metadata for 2405.00005?", "library_paper"),
    Task("Mark idea 17 as approved.", "library_paper"),
    Task("Show everything we know about paper 2401.00001.", "library_paper"),
    Task("Re-extract the bibliographic schema for 2401.00001.", "library_paper"),
    Task("Generate five methodological ideas about 2401.00001.", "library_paper"),
    Task("Delete paper 2401.00001.", "library_paper"),

    # ── library_find (13) ──
    Task("Find papers in my library that mention hallucination detection.", "library_find"),
    Task("List every paper I have imported, sorted by citation count.", "library_find"),
    Task("Show me the recent activity log entries.", "library_find"),
    Task("What about papers in my library about attention — top 5?", "library_find"),
    Task("List the first ten arXiv cs.LG papers I imported.", "library_find"),
    Task("Find papers in my library with min 100 citations about scaling.", "library_find"),
    Task("Show me only search actions from the activity log.", "library_find"),
    Task("List the recently added papers to my library.", "library_find"),
    Task("What summarise actions have I run in the past hour?", "library_find"),
    Task("Pull library papers ranked by published date.", "library_find"),
    Task("Find similar papers to my 2401.00001 inside my library.", "library_find"),
    Task("List my library papers by category cs.CL.", "library_find"),
    Task("Activity log filtered to extract actions.", "library_find"),
]


def stub_route(prompt: str) -> tuple[str, list[str]]:
    """Spec compatibility: defer to the production router.

    Kept here because the benchmark has historically owned the rules.
    Newly delegated to :func:`src.mcp_server.router.route` so the
    production router and the benchmark can never drift.
    """
    from src.mcp_server.router import route as _prod_route
    return _prod_route(prompt)


# Re-export the production router directly so external callers can
# `from tests.benchmark.bench_tool_selection import stub_route` and
# still get the canonical routing logic without an extra hop.

__all__ = [
    "TASKS",
    "stub_route",
    "real_llm_route",
    "schema_token_estimate",
    "get_old_tool_inventory",
    "get_new_tool_inventory",
    "bench_tool_selection",
]



# ── Real-LLM router (--real mode) ───────────────────────────────────────

def real_llm_route(prompt: str, llm) -> str:
    """Use the local LLM to pick a tool from the tool list."""
    tool_list = "\n".join(
        f"- {name}" for name in ("library_search", "library_paper", "library_find")
    )
    messages = [
        {"role": "system", "content":
         "You are a strict tool-routing classifier. Output only the chosen "
         "tool name, nothing else."},
        {"role": "user", "content":
         f"Available tools:\n{tool_list}\n\n"
         f"Task: {prompt}\n\n"
         "Reply with exactly one of: library_search, library_paper, "
         "library_find."},
    ]
    raw = llm.chat(messages, temperature=0.0, max_tokens=24, stop=["\n"])
    raw = raw.strip().strip('"').strip("'").lower()
    for name in ("library_search", "library_paper", "library_find"):
        if name in raw:
            return name
    return "library_find"


# ── Tool surface introspection ──────────────────────────────────────────

def schema_token_estimate(tool_schemas: list[dict]) -> int:
    return max(1, len(json.dumps(tool_schemas)) // 4)


def get_old_tool_inventory() -> dict:
    """Static representation of the pre-refactor 9-tool surface (for
    before/after reporting; the actual 9-tool surface has been removed
    from the codebase but the inventory is documented here)."""
    old_tools = [
        {"name": "research_search_papers", "params": ["query", "max_results",
            "primary_category", "auto_enrich", "summarize"]},
        {"name": "research_query_library", "params": ["query", "top_k",
            "min_citations", "venue", "primary_category", "rerank"]},
        {"name": "research_get_paper_details", "params": ["arxiv_id"]},
        {"name": "research_remove_paper", "params": ["arxiv_id", "delete_files"]},
        {"name": "research_generate_summary", "params": ["arxiv_id",
            "sections", "force"]},
        {"name": "research_extract_paper", "params": ["arxiv_id", "force"]},
        {"name": "research_generate_ideas", "params": ["arxiv_id",
            "num_ideas", "focus_area"]},
        {"name": "research_verify_novelty", "params": ["idea_id", "search_query"]},
        {"name": "research_list_library", "params": ["limit", "offset",
            "sort_by", "primary_category", "min_citations", "venue"]},
        {"name": "research_get_activity_log", "params": ["limit", "action_type"]},
    ]
    total_params = sum(len(t["params"]) for t in old_tools)
    return {
        "tool_count": len(old_tools),
        "arg_count": total_params,
        "tool_names": [t["name"] for t in old_tools],
    }


def get_new_tool_inventory(mcp_instance) -> dict:
    """Introspect the live MCP server."""
    if mcp_instance is None:
        return {"tool_count": 0, "arg_count": 0, "tool_names": [],
                "schema_token_est": 0}
    tools = asyncio.run(mcp_instance.list_tools())
    schemas = []
    total_params = 0
    for t in tools:
        schema = t.input_schema or {}
        props = schema.get("properties", {})
        schemas.append({"name": t.name, "params": list(props.keys()),
                        "description": t.description or ""})
        total_params += len(props)
    return {
        "tool_count": len(tools),
        "arg_count": total_params,
        "tool_names": [t.name for t in tools],
        "schema_token_est": schema_token_estimate(schemas),
    }


# ── Runner ──────────────────────────────────────────────────────────────

def bench_tool_selection(mcp_instance=None, llm=None) -> dict:
    """Run the reliability test and report accuracy + surface deltas."""
    inv_old = get_old_tool_inventory()
    inv_new = get_new_tool_inventory(mcp_instance)
    correct = 0
    confusions: list[dict] = []
    per_tool_hits: dict[str, dict[str, int]] = {}
    for t in TASKS:
        if llm is not None:
            pick = real_llm_route(t.prompt, llm)
        else:
            pick, _ = stub_route(t.prompt)
        ok = pick == t.expected_tool
        if ok:
            correct += 1
        per_tool_hits.setdefault(t.expected_tool,
                                   {"correct": 0, "total": 0})
        per_tool_hits[t.expected_tool]["total"] += 1
        if ok:
            per_tool_hits[t.expected_tool]["correct"] += 1
        else:
            confusions.append({
                "expected": t.expected_tool,
                "predicted": pick,
                "prompt": t.prompt,
            })
    summary = {
        "accuracy": round(correct / len(TASKS), 4) if TASKS else 0.0,
        "n_tasks": len(TASKS),
        "per_tool_accuracy": {
            k: round(v["correct"] / v["total"], 4) if v["total"] else 0.0
            for k, v in per_tool_hits.items()
        },
        "confusion_count": len(confusions),
        "confusions": confusions,
        "surface_old": inv_old,
        "surface_new": inv_new,
        "tool_count_reduction": f"{inv_old['tool_count']} -> {inv_new['tool_count']}",
        "arg_count_reduction": f"{inv_old['arg_count']} -> {inv_new['arg_count']}",
    }
    return summary
