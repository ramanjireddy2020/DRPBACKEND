"""Helpers for calling async coroutines from sync call sites (agent tools).

The txkg_test.py service functions are async (they use asyncio.to_thread for
blocking LLM/HTTP calls), but BaseTool.run() and the LangGraph node wrappers
are synchronous. run_sync() bridges the two, working whether or not the
caller happens to already be inside a running event loop.
"""
import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine to completion from synchronous code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread — safe to drive directly.
        return asyncio.run(coro)

    # A loop is already running on this thread (e.g. called from within an
    # async FastAPI handler) — asyncio.run() would raise, so execute the
    # coroutine on a fresh loop in a separate thread instead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
