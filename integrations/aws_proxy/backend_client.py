"""
InnoDD backend client — the shared piece both the Lambda handler and the Fargate
proxy app import.

Holds both auth layers so the frontend (or whatever calls this proxy) never has
to know Databricks Apps is behind it:

  1. Databricks Apps' own OAuth gate — client-credentials token for the app's
     service principal, cached and refreshed automatically.
  2. The DRP /v1 API's own session — a login-derived access token sent as
     X-Drp-Token (not Authorization, which layer 1 already owns), refreshed via
     the 30-day refresh token rather than re-sending the password every time.

`call_backend()` is the one generic function that reaches every one of the 62
`/v1/*` endpoints (see the reference table below) — a method + path + body/params
is enough; this module handles both tokens and retries once on a 401 in case a
token expired between the cache check and the request.

Environment variables required:
    DATABRICKS_TOKEN_URL      e.g. https://<workspace-host>/oidc/v1/token
    DATABRICKS_CLIENT_ID      the app's service-principal client id
    DATABRICKS_CLIENT_SECRET  the app's service-principal OAuth secret
    INNODD_BASE_URL           e.g. https://innodd-api-<id>.aws.databricksapps.com
    DRP_EMAIL / DRP_PASSWORD  the DRP /v1 login used to mint X-Drp-Token

Reference — every endpoint this proxies (method, path):
    POST /v1/auth/login                                  POST /v1/auth/logout
    POST /v1/auth/refresh                                POST /v1/auth/forgot-password
    GET  /v1/users/me                                     GET  /v1/users/me/onboarding-status
    POST /v1/users/me/research-focus                      POST /v1/users/me/onboarding/complete
    GET  /v1/therapeutic-areas
    GET  /v1/dashboard/summary                            GET  /v1/dashboard/quick-actions
    GET  /v1/dashboard/pipelines                          GET  /v1/dashboard/pipelines/manage
    GET  /v1/modules                                      GET  /v1/modules/search
    GET  /v1/agents/jobs/{jobId}/status                    GET  /v1/agents/jobs/{jobId}/result
    GET  /v1/sessions                                      POST /v1/sessions
    POST /v1/sessions/draft                                PATCH /v1/sessions/draft
    GET  /v1/sessions/{sessionId}
    POST /v1/agents/txkg/query                             GET  /v1/agents/txkg/targets
    GET  /v1/agents/txkg/targets/{uniprotId}                GET  /v1/agents/txkg/insights/{jobId}
    POST /v1/agents/subgraph/generate                      GET  /v1/agents/subgraph/{jobId}
    GET  /v1/agents/subgraph/{jobId}/stats                  POST /v1/agents/subgraph/{jobId}/explore
    POST /v1/agents/metapath/analyze                       GET  /v1/agents/metapath/{jobId}
    GET  /v1/agents/metapath/{jobId}/scores                 GET  /v1/agents/metapath/{jobId}/traversals
    POST /v1/agents/litminex/targets/custom                 GET  /v1/agents/litminex/targets/custom
    POST /v1/agents/litminex/query                          GET  /v1/agents/litminex/{jobId}/results
    GET  /v1/agents/litminex/{jobId}/insights                GET  /v1/agents/litminex/preview
    POST /v1/agents/curatex/target-profile                  POST /v1/agents/curatex/compounds
    POST /v1/agents/screensuite/screen                      GET  /v1/agents/screensuite/{jobId}/hits
    POST /v1/agents/novsearch/assess                        GET  /v1/agents/novsearch/{jobId}/report
    POST /v1/agents/novsearch/ask
    GET  /v1/articles/{articleId}                            GET  /v1/articles/{articleId}/pmc-link
    POST /v1/articles/{articleId}/save                       POST /v1/articles/{articleId}/chat
    GET  /v1/articles/{articleId}/chat/history
    GET  /v1/projects                                        POST /v1/projects
    GET  /v1/projects/{projectId}                            GET  /v1/projects/{projectId}/items
    POST /v1/projects/{projectId}/results
    POST /v1/files/upload                                    GET  /v1/files/{fileId}/download
    POST /v1/exports                                         POST /v1/exports/pdf
    GET  /v1/exports/{exportId}/download
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx

DATABRICKS_TOKEN_URL = os.environ["DATABRICKS_TOKEN_URL"]
DATABRICKS_CLIENT_ID = os.environ["DATABRICKS_CLIENT_ID"]
DATABRICKS_CLIENT_SECRET = os.environ["DATABRICKS_CLIENT_SECRET"]
INNODD_BASE_URL = os.environ["INNODD_BASE_URL"].rstrip("/")
DRP_EMAIL = os.environ["DRP_EMAIL"]
DRP_PASSWORD = os.environ["DRP_PASSWORD"]

_REQUEST_TIMEOUT = 60.0

# Module-level cache: Lambda reuses this across warm invocations of the same
# execution environment; a Fargate process holds it for its whole lifetime.
# Neither target shares it across instances — each container/execution
# environment authenticates independently, which is fine at this call volume.
_databricks_token: dict[str, Any] = {"value": None, "expires_at": 0.0}
_drp_session: dict[str, Any] = {"access": None, "refresh": None, "expires_at": 0.0}


class BackendError(RuntimeError):
    """The InnoDD backend (or the Databricks auth layer in front of it) returned an error."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"{status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class BinaryResponse:
    """
    A backend response that is not JSON — an export file, a generated report.

    Carried rather than parsed, so the handler can base64-encode it and set the
    real content type. Kept as a plain class (no dataclass) to avoid adding an
    import to a Lambda that ships as a single zip.
    """

    __slots__ = ("content_type", "content")

    def __init__(self, content_type: str, content: bytes):
        self.content_type = content_type
        self.content = content


def _get_databricks_token() -> str:
    """Client-credentials token for the app's service principal, cached until near expiry."""
    now = time.time()
    if _databricks_token["value"] and now < _databricks_token["expires_at"] - 60:
        return _databricks_token["value"]

    resp = httpx.post(
        DATABRICKS_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": DATABRICKS_CLIENT_ID,
            "client_secret": DATABRICKS_CLIENT_SECRET,
            "scope": "all-apis",
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    body = resp.json()
    _databricks_token["value"] = body["access_token"]
    _databricks_token["expires_at"] = now + float(body.get("expires_in", 3600))
    return _databricks_token["value"]


def _get_drp_token() -> str:
    """
    DRP /v1 session token, cached until near expiry (30 min) — refreshed via the
    30-day refresh token when possible, falling back to a fresh login only when
    there's no refresh token yet (first call) or the refresh itself fails.
    """
    now = time.time()
    if _drp_session["access"] and now < _drp_session["expires_at"] - 30:
        return _drp_session["access"]

    headers = {"Authorization": f"Bearer {_get_databricks_token()}"}
    resp = None
    if _drp_session["refresh"]:
        resp = httpx.post(
            f"{INNODD_BASE_URL}/v1/auth/refresh",
            headers=headers,
            json={"refreshToken": _drp_session["refresh"]},
            timeout=15.0,
        )
        if resp.status_code >= 400:
            resp = None  # refresh token expired/invalid — fall through to login

    if resp is None:
        resp = httpx.post(
            f"{INNODD_BASE_URL}/auth/login",
            headers=headers,
            json={"email": DRP_EMAIL, "password": DRP_PASSWORD},
            timeout=15.0,
        )
    resp.raise_for_status()
    body = resp.json()
    _drp_session["access"] = body["accessToken"]
    _drp_session["refresh"] = body.get("refreshToken") or _drp_session["refresh"]
    _drp_session["expires_at"] = now + float(body.get("expiresIn", 1800))
    # Deliberately not logged: `_drp_session` holds a live access token and a
    # 30-day refresh token.
    return _drp_session["access"]


# Paths with no caller identity yet. INNODD_BASE_URL already ends in /v1, so
# these are written without that prefix — the same form the handler receives.
_PUBLIC_PATHS = frozenset({"/auth/login", "/auth/refresh", "/auth/forgot-password"})


def call_backend(
    method: str,
    path: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    user_token: Optional[str] = None,
    _retry: bool = True,
) -> Any:
    """
    Call any InnoDD /v1/* endpoint (see the reference list in the module
    docstring). `path` should start with /v1/... Returns the decoded JSON body,
    or None for a 204/empty response. Raises BackendError on a non-2xx response.

    `user_token` is the end user's Cognito token, taken from the caller's own
    Authorization header and forwarded as `X-Drp-Token` so the backend sees the
    real user. This proxy authenticates the *transport* (Databricks Apps' OAuth
    gate), never the *user*.

    Without it, a non-public path is refused rather than falling back to the
    shared DRP_EMAIL account — that fallback would silently make every anonymous
    caller the same real user.
    """
    # Nothing here logs `json_body`: on /auth/login it holds a plaintext
    # password, and on every other path it can hold user content. The method and
    # path are enough to trace a request in CloudWatch.
    print(f"proxy -> {method.upper()} {path}")
    if not path.startswith("/"):
        path = "/" + path

    if not user_token and path.rstrip("/") not in _PUBLIC_PATHS:
        raise BackendError(
            401,
            '{"error":"missing_token","message":"No Authorization bearer token was '
            'forwarded to the proxy. Send the Cognito id token from the frontend."}',
        )

    url = f"{INNODD_BASE_URL}{path}"
    headers = {
        # Layer 1: the Databricks Apps OAuth gate, which owns `Authorization`.
        "Authorization": f"Bearer {_get_databricks_token()}",
        "Content-Type": "application/json",
    }
    # Layer 2: who the user is. Absent only on the public paths checked above.
    if user_token:
        headers["X-Drp-Token"] = user_token

    resp = httpx.request(
        method.upper(), url, headers=headers, json=json_body, params=params, timeout=_REQUEST_TIMEOUT
    )

    if resp.status_code == 401 and _retry:
        # Only the Databricks transport token is worth retrying — it may have
        # expired between the cache check and this request. A 401 on a forwarded
        # user token means that user's Cognito session is genuinely expired, and
        # retrying just re-sends the same dead token, so let it surface.
        _databricks_token["value"] = None
        if not user_token:
            _drp_session["access"] = None
        return call_backend(
            method, path, json_body=json_body, params=params,
            user_token=user_token, _retry=False,
        )

    if resp.status_code >= 400:
        raise BackendError(resp.status_code, resp.text)

    if not resp.content:
        return None

    # Not every endpoint answers with JSON. `GET /v1/exports/{id}/download`
    # returns CSV, SVG or PDF, and parsing those as JSON raised
    # "Expecting value: line 1 column 1", which the handler turned into
    # `502 proxy_failure` — so every export except JSON, and every generated
    # report, failed at the download step while the backend had produced the
    # file correctly. The status code said nothing about what went wrong.
    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type in ("", "application/json") or content_type.endswith("+json"):
        try:
            return resp.json()
        except ValueError:
            # Declared JSON but is not: fall through and pass the bytes on rather
            # than losing the response to a parse error.
            pass
    return BinaryResponse(content_type or "application/octet-stream", resp.content)
