"""Shared utility helpers."""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")

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
