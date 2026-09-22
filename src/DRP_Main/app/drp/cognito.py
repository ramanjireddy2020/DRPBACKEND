"""
AWS Cognito token verification for the DRP `/v1` API.

The frontend authenticates against a Cognito user pool (Amplify SRP) and sends
the resulting JWT to this API. Those tokens are **RS256**, signed by Cognito's
private key — they cannot be validated with `drp/security.py`'s HS256 secret,
which is why every Cognito-authenticated request used to 401.

Verification here follows AWS's documented steps: fetch the pool's JWKS, pick
the key named by the token's `kid`, check the signature, then check `iss`,
`token_use`, `exp` and the client id.

Both token types are accepted:

  * **id token**  — `token_use: "id"`, `aud` = app client id. Carries `email`
    and `name`, so a first-time user can be provisioned with a real profile.
  * **access token** — `token_use: "access"`, `client_id` = app client id, and
    *no* email claim. Usable, but the provisioned profile will be bare.

Prefer sending the id token for that reason.

The module is inert unless `COGNITO_USER_POOL_ID` is configured, so local
development keeps working on the built-in HS256 tokens alone.
"""
from __future__ import annotations

import threading
import time

import httpx
from jose import JWTError, jwt

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

# Cognito rotates signing keys rarely; a long TTL keeps the hot path free of
# network calls, and an unknown `kid` forces an immediate refetch anyway.
_JWKS_TTL_SECONDS = 3600

_jwks_cache: dict | None = None
_jwks_fetched_at: float = 0.0
_jwks_lock = threading.Lock()


def is_enabled() -> bool:
    return bool(settings.COGNITO_USER_POOL_ID)


def issuer() -> str:
    return (
        f"https://cognito-idp.{settings.COGNITO_REGION}.amazonaws.com/"
        f"{settings.COGNITO_USER_POOL_ID}"
    )


def _jwks_url() -> str:
    return f"{issuer()}/.well-known/jwks.json"


def _load_jwks(force: bool = False) -> dict:
    """Return the pool's JWKS, refetching when stale or when `force` is set."""
    global _jwks_cache, _jwks_fetched_at
    with _jwks_lock:
        fresh = _jwks_cache is not None and (time.time() - _jwks_fetched_at) < _JWKS_TTL_SECONDS
        if fresh and not force:
            return _jwks_cache
        try:
            resp = httpx.get(_jwks_url(), timeout=10.0)
            resp.raise_for_status()
            _jwks_cache = resp.json()
            _jwks_fetched_at = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.error("Cognito JWKS fetch failed (%s): %s", _jwks_url(), exc)
            # Serve a stale copy rather than locking every user out on a blip.
            if _jwks_cache is None:
                raise
        return _jwks_cache


def _key_for(kid: str) -> dict | None:
    for jwks in (_load_jwks(), _load_jwks(force=True)):
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key
    return None


def decode_cognito_token(token: str) -> dict | None:
    """
    Return the verified claims of a Cognito id/access token, or None.

    None means "not a valid Cognito token" for any reason — wrong signature,
    expired, wrong pool, wrong app client, or Cognito not configured. The caller
    decides what to do about it; nothing here raises.
    """
    if not is_enabled():
        return None

    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        return None

    if header.get("alg") != "RS256":
        # HS256 here is one of our own local tokens, not a Cognito one.
        return None

    kid = header.get("kid")
    if not kid:
        return None

    key = _key_for(kid)
    if key is None:
        logger.warning("No Cognito JWKS key for kid=%s", kid)
        return None

    client_id = settings.COGNITO_APP_CLIENT_ID or None
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=issuer(),
            # Only id tokens carry `aud`; access tokens put the client id in
            # `client_id` instead, so audience is checked by hand below.
            options={"verify_aud": False},
        )
    except JWTError as exc:
        logger.info("Cognito token rejected: %s", exc)
        return None

    token_use = claims.get("token_use")
    if token_use not in ("id", "access"):
        return None

    if client_id:
        presented = claims.get("aud") if token_use == "id" else claims.get("client_id")
        if presented != client_id:
            logger.info("Cognito token for unexpected client %s", presented)
            return None

    if not claims.get("sub"):
        return None

    return claims


def identity_from_claims(claims: dict) -> dict:
    """Flatten the claims we provision a local user from."""
    username = claims.get("cognito:username") or claims.get("username") or ""
    email = claims.get("email") or ""
    name = claims.get("name") or " ".join(
        p for p in (claims.get("given_name"), claims.get("family_name")) if p
    )
    return {
        "sub": claims["sub"],
        "email": email,
        "username": username or (email.split("@")[0] if email else claims["sub"]),
        "full_name": name or email.split("@")[0] if email else name,
    }
