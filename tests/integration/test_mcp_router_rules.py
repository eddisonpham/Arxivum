"""Integration test for the rule-based MCP router.

The router is ``src.mcp_server.router.route``; the benchmark shares
the same rules in ``tests.benchmark.bench_tool_selection.stub_route``.
This test pins the *production* router to the same fixture so that
the two can never drift apart. Acceptance: ≥ 95 % accuracy on the
42-task synthetic set, with ≥ 80 % accuracy per tool.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.mcp_server.router import LIBRARY_FIND, LIBRARY_PAPER, LIBRARY_SEARCH, route


@dataclass(frozen=True)
class _Task:
    prompt: str
    expected: str


# Subset of the bench_tool_selection tasks, kept here so this test has
# no fixture import dependency. The router rules are shared so the
# answers are guaranteed to match.
TASKS: list[_Task] = [
    # ── library_search ──
    _Task("Find the top 10 papers on transformer attention.", LIBRARY_SEARCH),
    _Task("Search arXiv for MoE scaling laws and import them.", LIBRARY_SEARCH),
    _Task("Look up the latest arXiv submission on retrieval augmentation.", LIBRARY_SEARCH),
    _Task("Import these papers from arXiv into the library.", LIBRARY_SEARCH),
    _Task("Add a paper to my library with auto-enrich from S2.", LIBRARY_SEARCH),
    _Task("Bring in the recent RAG papers.", LIBRARY_SEARCH),
    _Task("Find the most-cited papers on long-context attention.", LIBRARY_SEARCH),
    _Task("Pull new papers on factuality hallucination since 2024.", LIBRARY_SEARCH),
    _Task("Find papers on hyperspectral imaging under cs.CV.", LIBRARY_SEARCH),
    _Task("Import and summarise the top three Mixture-of-Experts pre-prints.", LIBRARY_SEARCH),

    # ── library_paper ──
    _Task("What does paper 2401.00001 say overall?", LIBRARY_PAPER),
    _Task("Summarise paper 2205.13273 in the seven canonical sections.", LIBRARY_PAPER),
    _Task("Generate a structured extraction schema for 2309.01234.", LIBRARY_PAPER),
    _Task("Propose three novel ideas grounded in 2401.00001.", LIBRARY_PAPER),
    _Task("Remove paper 2211.12792 from the library.", LIBRARY_PAPER),
    _Task("Verify the novelty of idea 42 against arXiv.", LIBRARY_PAPER),
    _Task("Approve idea 17 as a research direction.", LIBRARY_PAPER),
    _Task("Reject idea 9 — too generic.", LIBRARY_PAPER),
    _Task("Pull the abstract-extracted schema for 2406.00007.", LIBRARY_PAPER),
    _Task("Re-extract the bibliographic schema for 2401.00001.", LIBRARY_PAPER),
    _Task("Did we already extract the metadata for 2405.00005?", LIBRARY_PAPER),
    _Task("Show everything we know about paper 2401.00001.", LIBRARY_PAPER),
    _Task("Generate five methodological ideas about 2401.00001.", LIBRARY_PAPER),
    _Task("Delete paper 2401.00001.", LIBRARY_PAPER),
    _Task("Mark idea 17 as approved.", LIBRARY_PAPER),

    # ── library_find ──
    _Task("Find papers in my library that mention hallucination detection.", LIBRARY_FIND),
    _Task("List every paper I have imported, sorted by citation count.", LIBRARY_FIND),
    _Task("Show me the recent activity log entries.", LIBRARY_FIND),
    _Task("List the first ten arXiv cs.LG papers I imported.", LIBRARY_FIND),
    _Task("Find papers in my library with min 100 citations about scaling.", LIBRARY_FIND),
    _Task("Show me only search actions from the activity log.", LIBRARY_FIND),
    _Task("What summarise actions have I run in the past hour?", LIBRARY_FIND),
    _Task("Find similar papers to my 2401.00001 inside my library.", LIBRARY_FIND),
    _Task("List my library papers by category cs.CL.", LIBRARY_FIND),
    _Task("Activity log filtered to extract actions.", LIBRARY_FIND),
]


def test_router_constant_set():
    assert {LIBRARY_SEARCH, LIBRARY_PAPER, LIBRARY_FIND} == {
        "library_search", "library_paper", "library_find",
    }


def test_router_accuracy_overall_at_least_95pct():
    correct = sum(1 for t in TASKS if route(t.prompt)[0] == t.expected)
    acc = correct / len(TASKS)
    assert acc >= 0.95, (
        f"Router accuracy regressed to {acc:.2%}; "
        f"{len(TASKS) - correct} of {len(TASKS)} tasks mis-routed. "
        f"Worst offenders: {sorted([(t.prompt, route(t.prompt)[0]) for t in TASKS if route(t.prompt)[0] != t.expected], key=lambda x: x[0])[:5]}"
    )


@pytest.mark.parametrize("tool", [LIBRARY_SEARCH, LIBRARY_PAPER, LIBRARY_FIND])
def test_router_per_tool_accuracy_at_least_80pct(tool):
    subset = [t for t in TASKS if t.expected == tool]
    if not subset:
        pytest.skip(f"no tasks for {tool}")
    correct = sum(1 for t in subset if route(t.prompt)[0] == tool)
    acc = correct / len(subset)
    assert acc >= 0.80, f"{tool}: {acc:.2%} ({correct}/{len(subset)})"


def test_router_always_returns_a_valid_tool():
    """No prompt should ever return a name outside the 3-tool surface."""
    bad: list[str] = []
    for t in TASKS:
        pick, _ = route(t.prompt)
        if pick not in {LIBRARY_SEARCH, LIBRARY_PAPER, LIBRARY_FIND}:
            bad.append(f"{t.prompt!r} -> {pick!r}")
    assert not bad, "; ".join(bad)


def test_router_fallback_for_unmatched():
    """A prompt about nothing matches should default to library_find."""
    pick, ranked = route("asdfghjkl")
    assert pick == LIBRARY_FIND
    assert set(ranked) <= {LIBRARY_SEARCH, LIBRARY_PAPER, LIBRARY_FIND}


def test_router_known_examples():
    spot = {
        # search-side
        "Search arXiv for transformers and import them": LIBRARY_SEARCH,
        "Pull new papers on long-context attention since 2024": LIBRARY_SEARCH,
        # paper-side
        "Generate a structured extraction schema for 2309.01234": LIBRARY_PAPER,
        "Verify the novelty of idea 42 against arXiv": LIBRARY_PAPER,
        # find-side
        "List every paper I have imported, sorted by citation count": LIBRARY_FIND,
        "What summarise actions have I run in the past hour?": LIBRARY_FIND,
    }
    for prompt, expected in spot.items():
        assert route(prompt)[0] == expected, (
            f"spot-check failed: {prompt!r} -> expected {expected}, "
            f"got {route(prompt)[0]}"
        )
