"""Application factory — wires all components together.

Both the MCP server and the FastAPI service use :func:`create_app` to get
a fully-wired :class:`AppContext` containing the database, ChromaDB,
clients, model manager, and all services.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.clients.arxiv_client import ArxivClient
from src.utils import run_async
from src.clients.s2_client import SemanticScholarClient
from src.config import get_settings
from src.db.chroma_store import ChromaStore
from src.db.models import Database
from src.inference.manager import ModelManager
from src.services.ideas import IdeaService
from src.services.library import LibraryService
from src.services.novelty import NoveltyService
from src.services.summarizer import SummarizerService

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Container holding all wired application components."""

    db: Database
    chroma: ChromaStore
    arxiv_client: ArxivClient
    s2_client: SemanticScholarClient
    models: ModelManager
    library: LibraryService
    summarizer: SummarizerService
    ideas: IdeaService
    novelty: NoveltyService


def create_app(
    db_path: str = "",
    chroma_path: str = "",
    constrained_memory: bool = True,
) -> AppContext:
    """Create and wire all application components.

    Args:
        db_path: Path to SQLite DB.  Empty string = in-memory (for tests).
        chroma_path: Path to ChromaDB.  Empty string = ephemeral (for tests).
        constrained_memory: If True, only one heavy model resident at a time.
    """
    settings = get_settings()
    if not db_path:
        db_path = str(settings.db_path) if settings.app_env != "test" else ""
    if not chroma_path:
        chroma_path = str(settings.chroma_path) if settings.app_env != "test" else ""

    if db_path:
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    db = Database(db_path)
    chroma = ChromaStore(path=chroma_path)
    arxiv_client = ArxivClient()
    s2_client = SemanticScholarClient()
    models = ModelManager(constrained_memory=constrained_memory)

    library = LibraryService(db, chroma, arxiv_client, s2_client, models)
    summarizer = SummarizerService(db, models, library)
    ideas = IdeaService(db, models)
    novelty = NoveltyService(
        db, models, arxiv_client,
        library_query_fn=library.query_library,
    )

    ctx = AppContext(
        db=db, chroma=chroma, arxiv_client=arxiv_client, s2_client=s2_client,
        models=models, library=library, summarizer=summarizer,
        ideas=ideas, novelty=novelty,
    )
    logger.info("App context created (db=%s, chroma=%s)", db_path or ":memory:",
                chroma_path or "ephemeral")
    return ctx


def shutdown_app(ctx: AppContext) -> None:
    """Gracefully close all resources."""
    ctx.models.shutdown()
    try:
        run_async(ctx.s2_client.aclose())
    except Exception:
        logger.debug("s2_client.aclose() failed during shutdown", exc_info=True)
    ctx.db.close()
    logger.info("App context shut down.")
