"""
AWS Lambda entry point — proxies API Gateway requests to InnoDD's /v1 API.

Named lambda_function.py because that's the file/handler AWS's tooling assumes
by default (`lambda_function.lambda_handler`) unless --handler overrides it.

Handles both HTTP API (payload format 2.0, `rawPath`/`requestContext.http.method`)
and REST API (payload format 1.0, `path`/`httpMethod`) event shapes, so it works
behind either API Gateway type without configuration.

Deploy: zip this file + backend_client.py + the httpx dependency (see
requirements.txt) as the Lambda package. Set the 6 env vars backend_client.py
requires on the Lambda function's configuration (via Secrets Manager/SSM
references, not plain env vars — DATABRICKS_CLIENT_SECRET is a live credential).
"""
from __future__ import annotations

import base64
import json
from typing import Any, Dict

from backend_client import BackendError, BinaryResponse, call_backend


def _bearer_token(event: Dict[str, Any]) -> str | None:
    """
    The caller's own bearer token — the end user's Cognito token from the browser.

    API Gateway lower-cases header names on payload format 2.0 but preserves the
    client's casing on 1.0, so the lookup has to be case-insensitive either way.
    """
    headers = event.get("headers") or {}
    value = next(
        (v for k, v in headers.items() if k and k.lower() == "authorization"), None
    )
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return value.strip() or None


def _extract_request(event: Dict[str, Any]) -> tuple[str, str, dict | None, dict | None]:
    """Normalise a v1 (REST API) or v2 (HTTP API) API Gateway event."""
    if "rawPath" in event:  # HTTP API, payload format 2.0
        method = event["requestContext"]["http"]["method"]
        path = event["rawPath"]
        params = event.get("queryStringParameters") or {}
    else:  # REST API, payload format 1.0
        method = event["httpMethod"]
        path = event["path"]
        params = event.get("queryStringParameters") or {}

    body = None
    raw_body = event.get("body")
    if raw_body:
        body = json.loads(raw_body)

    return method, path, body, params


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    method, path, body, params = _extract_request(event)

    # CORS preflight. The API Gateway routes are `ANY /<path>`, which matches
    # OPTIONS as well, so the gateway routes preflights here instead of answering
    # them itself — and its JWT authorizer would 401 them, because a browser
    # never sends Authorization on a preflight. Answering 204 here (behind an
    # `OPTIONS /<path>` route with no authorizer) is what lets a browser through;
    # the gateway's CorsConfiguration adds the Access-Control-* headers.
    if method.upper() == "OPTIONS":
        return {"statusCode": 204, "headers": {}, "body": ""}

    user_token = _bearer_token(event)

    try:
        result = call_backend(
            method, path, json_body=body, params=params, user_token=user_token
        )
        if isinstance(result, BinaryResponse):
            # File downloads (CSV / SVG / PDF exports, generated reports). API
            # Gateway needs the payload base64-encoded and flagged as such;
            # returning the raw bytes as `body` corrupts anything non-UTF-8.
            return {
                "statusCode": 200,
                "headers": {"Content-Type": result.content_type},
                "body": base64.b64encode(result.content).decode("ascii"),
                "isBase64Encoded": True,
            }
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result) if result is not None else "",
        }
    except BackendError as exc:
        return {
            "statusCode": exc.status_code,
            "headers": {"Content-Type": "application/json"},
            "body": exc.body,
        }
    except Exception as exc:  # noqa: BLE001 — surface as 502, never let Lambda crash silently
        return {
            "statusCode": 502,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "proxy_failure", "message": str(exc)}),
        }
