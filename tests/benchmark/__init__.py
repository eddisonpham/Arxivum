"""Benchmark and evaluation suite for ArXivum.

Operationalises the metrics in docs/AI_RESEARCHER_CRITIQUE.md so that
every criticism has a number attached and every refactor passes the
same test. Run via `python -m tests.benchmark.run_all [--real]`.
The `--real` flag invokes the actual local LLM as the judge; without
it the suite runs on stub/mocked models for fast, deterministic CI.
"""
