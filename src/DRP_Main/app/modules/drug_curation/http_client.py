"""
Shared HTTP plumbing for the CurateX source clients.

Every §2 source is a public REST/GraphQL endpoint with its own rate limits and
its own habit of going down. One retrying session, one timeout policy and one
"never raise, return None" contract keeps a single flaky source from failing a
whole profile build — the coverage indicator (§4.3) is what surfaces the gap.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import requests

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

USER_AGENT = "InnoDD-CurateX/1.0 (https://innominds.com; pharmaceutical research)"

_session_lock = threading.Lock()
_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """A process-wide pooled session. Thread-safe: requests.Session is."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                session = requests.Session()
                session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
                adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=32)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                _session = session
    return _session


def _timeout() -> int:
    return int(getattr(settings, "CURATEX_REQUEST_TIMEOUT", 20) or 20)


def _retries() -> int:
    return int(getattr(settings, "CURATEX_MAX_RETRIES", 2) or 2)


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
) -> Optional[Any]:
    """GET a JSON document. Returns None on any failure, never raises."""
    attempts = (retries if retries is not None else _retries()) + 1
    for attempt in range(attempts):
        try:
            response = get_session().get(
                url, params=params, headers=headers, timeout=timeout or _timeout()
            )
            if response.status_code == 404:
                return None
            if response.status_code == 429 or 500 <= response.status_code < 600:
                # Transient: back off and retry rather than reporting no data.
                if attempt + 1 < attempts:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                logger.warning("GET %s returned %s after retries", url, response.status_code)
                return None
            response.raise_for_status()
            return response.json()
        except ValueError:
            logger.warning("GET %s returned a non-JSON body", url)
            return None
        except requests.RequestException as exc:
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2 ** attempt))
                continue
            logger.warning("GET %s failed: %s", url, exc)
            return None
    return None


def get_text(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """GET a text/XML document (DailyMed SPL). Returns None on failure."""
    try:
        response = get_session().get(
            url, params=params, timeout=timeout or _timeout(), headers={"Accept": "*/*"}
        )
        if response.status_code != 200:
            return None
        return response.text
    except requests.RequestException as exc:
        logger.warning("GET %s failed: %s", url, exc)
        return None


def post_graphql(
    url: str,
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    *,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    POST a GraphQL query (Open Targets). Returns the `data` block, or None.

    GraphQL answers 200 with an `errors` array, so a status check alone is not
    enough to tell a good response from a broken query.
    """
    attempts = (retries if retries is not None else _retries()) + 1
    payload = {"query": query, "variables": variables or {}}
    for attempt in range(attempts):
        try:
            response = get_session().post(
                url,
                json=payload,
                timeout=timeout or _timeout(),
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt + 1 < attempts:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                logger.warning("GraphQL %s returned %s", url, response.status_code)
                return None
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                logger.warning("GraphQL %s reported errors: %s", url, body["errors"][:1])
            return body.get("data")
        except (ValueError, requests.RequestException) as exc:
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2 ** attempt))
                continue
            logger.warning("GraphQL %s failed: %s", url, exc)
            return None
    return None
