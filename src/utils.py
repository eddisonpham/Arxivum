"""Shared utility helpers."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Awaitable, Iterator, TypeVar

from src.db.models import ActivityRow, Database

T = TypeVar("T")


@contextmanager
def track_activity(
    db: Database, action_type: str, **kwargs: Any
) -> Iterator[None]:
    """Log an activity and auto-update its status on success or failure.

    Usage::

        with track_activity(db, "search", query=q):
            results = do_work()
            return results

    The activity is marked ``completed`` on normal exit and ``failed`` on
    any exception (which is re-raised).
    """
    log_id = db.log_activity(ActivityRow(
        id=None, action_type=action_type, status="started", **kwargs,
    ))
    try:
        yield
        db.update_activity_status(log_id, "completed")
    except Exception:
        db.update_activity_status(log_id, "failed")
        raise

def run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion, safe in both sync and async contexts.

    * If there is **no** running event loop (e.g. MCP stdio server), this
      uses ``asyncio.run()``.
    * If there **is** a running loop (e.g. inside a FastAPI async
      endpoint), we cannot call ``asyncio.run()`` — instead we schedule
      the coroutine on the running loop and block until it completes via
      ``asyncio.run_coroutine_threadsafe`` + ``concurrent.futures``.

    For the POC, callers in async FastAPI endpoints should prefer
    ``await asyncio.to_thread(sync_fn, ...)`` to avoid blocking the loop.
    This helper is the safety net for synchronous call paths.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures
    import threading

    loop = asyncio.get_event_loop()
    future: concurrent.futures.Future[T] = asyncio.run_coroutine_threadsafe(
        coro, loop  # type: ignore[arg-type]
    )
    return future.result()
