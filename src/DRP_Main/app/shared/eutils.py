"""
One process-wide rate limiter for NCBI E-utilities.

NCBI's limit is per *IP*, not per caller: 3 requests/second without an API key,
10 with one. This process had three independent limiters — `txkg/discovery_service`,
`literature/mining_agent`, and nothing at all in `drug_curation/pubmed_service` —
so each stayed under the limit on its own while the process as a whole did not.
Running the full pipeline made that obvious: CurateX issued one unthrottled query
per candidate, saturated the quota, and LitMineX (sharing the same egress IP)
then retrieved zero articles for queries that work fine in isolation.

Everything that talks to eutils should go through here, so the budget is shared.
Both a sync and an async gate are provided because the callers are mixed: the
drug-curation service uses `requests`, the literature agent uses `httpx`.

On Databricks Apps the egress IP is shared with other tenants, so an API key is
worth more than the raw numbers suggest — it moves the quota from per-IP to
per-key. Set `NCBI_API_KEY` and both the pacing and the key itself are picked up
automatically.
"""
from __future__ import annotations

import asyncio
import os
import random
import threading
import time
from typing import Any, Dict, Optional

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


def api_key() -> str:
    """The NCBI API key, if one is configured."""
    return os.getenv("NCBI_API_KEY", "")


def _min_interval() -> float:
    # 10 req/s with a key, 3 without — stay just inside either, leaving headroom
    # for anything else sharing the egress IP.
    return 0.11 if api_key() else 0.4


def with_api_key(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add `api_key` to eutils query params when one is configured."""
    key = api_key()
    if key:
        params = {**params, "api_key": key}
    return params


# ── sync gate (requests-based callers) ────────────────────────────────────────
_sync_lock = threading.Lock()
_sync_last = [0.0]


def throttle_sync() -> None:
    """Block until the next eutils request may be sent."""
    with _sync_lock:
        wait = _min_interval() - (time.monotonic() - _sync_last[0])
        if wait > 0:
            time.sleep(wait)
        _sync_last[0] = time.monotonic()


# ── async gate (httpx-based callers) ──────────────────────────────────────────
#: Keyed by event loop: an asyncio primitive is bound to the loop that created
#: it, and tests call `asyncio.run` repeatedly while the server has one loop.
_async_gates: Dict[int, Any] = {}


def _async_gate():
    try:
        loop_key = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_key = 0
    gate = _async_gates.get(loop_key)
    if gate is None:
        gate = (asyncio.Lock(), [0.0])
        if len(_async_gates) > 32:
            _async_gates.clear()
        _async_gates[loop_key] = gate
    return gate


async def throttle_async() -> None:
    """Await until the next eutils request may be sent."""
    lock, last = _async_gate()
    async with lock:
        wait = _min_interval() - (time.monotonic() - last[0])
        if wait > 0:
            await asyncio.sleep(wait)
        last[0] = time.monotonic()


# ── retry on 429 ──────────────────────────────────────────────────────────────
MAX_RETRIES = 3


def backoff_seconds(attempt: int) -> float:
    """
    Exponential backoff with jitter: ~0.5s, ~1s, ~2s.

    The jitter matters more than the delay here — without it, every concurrent
    caller that got a 429 retries at the same instant and trips the limit again.
    """
    return (0.5 * (2 ** attempt)) * (0.75 + random.random() * 0.5)


def is_rate_limited(status_code: Optional[int]) -> bool:
    return status_code == 429
