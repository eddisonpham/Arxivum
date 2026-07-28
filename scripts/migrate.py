"""Database migration / initialisation script.

Creates the SQLite database (with all tables) and an empty ChromaDB
collection under the configured data directory.

Usage::

    python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script (``python scripts/migrate.py``) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.db.models import Database
from src.db.chroma_store import ChromaStore


def main() -> int:
    settings = get_settings()
    settings.ensure_dirs()

    print(f"[migrate] SQLite → {settings.db_path}")
    db = Database(settings.db_path)
    paper_count = db.count_papers()
    print(f"[migrate] SQLite ready ({paper_count} papers currently stored).")
    db.close()

    print(f"[migrate] ChromaDB → {settings.chroma_path}")
    store = ChromaStore(path=str(settings.chroma_path))
    print(f"[migrate] ChromaDB ready ({store.count()} chunks currently stored).")

    print("[migrate] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
