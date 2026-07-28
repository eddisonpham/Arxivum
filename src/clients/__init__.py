"""External API clients: arXiv + Semantic Scholar."""

from src.clients.arxiv_client import ArxivClient, ArxivPaper
from src.clients.s2_client import SemanticScholarClient, S2Metrics

__all__ = [
    "ArxivClient",
    "ArxivPaper",
    "SemanticScholarClient",
    "S2Metrics",
]
