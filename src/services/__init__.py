"""Service layer: business logic for library, summarization, ideas, novelty."""

from src.services.library import LibraryService, QueryResultItem
from src.services.summarizer import SummarizerService
from src.services.ideas import IdeaService
from src.services.novelty import NoveltyService

__all__ = [
    "LibraryService",
    "QueryResultItem",
    "SummarizerService",
    "IdeaService",
    "NoveltyService",
]
