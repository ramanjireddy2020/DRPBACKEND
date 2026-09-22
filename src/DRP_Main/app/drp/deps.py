"""Shared FastAPI dependencies for the DRP `/v1` API."""
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import cognito
from DRP_Main.app.drp.models import DrpUserProfile
from DRP_Main.app.drp.security import decode_access_token
from DRP_Main.app.models.user import User

logger = get_logger(__name__)

bearer_scheme = HTTPBearer(auto_error=False, description="Bearer {accessToken}")

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def _user_from_cognito(db: Session, claims: dict) -> User | None:
    """
    Map verified Cognito claims onto a `users` row.

    Lookup order is `cognito_sub`, then email — the second lets an account that
    already exists locally (a seeded or pre-provisioned user) be claimed by its
    Cognito identity on first login rather than duplicated.
    """
    identity = cognito.identity_from_claims(claims)
    sub = identity["sub"]

    user = db.query(User).filter(User.cognito_sub == sub).first()
    if user is not None:
        return user

    if identity["email"]:
        user = db.query(User).filter(User.email == identity["email"]).first()
        if user is not None:
            user.cognito_sub = sub
            db.commit()
            logger.info("Linked existing user %s to Cognito sub %s", identity["email"], sub)
            return user

    if not settings.COGNITO_AUTO_PROVISION:
        return None

    # A Cognito access token carries no email, so `email` may be empty here. The
    # column is nullable and the row is keyed on `cognito_sub`, so the account is
    # still coherent; sending the id token instead fills the profile properly.
    user = User(
        username=identity["username"],
        email=identity["email"] or None,
        full_name=identity["full_name"] or identity["username"],
        hashed_password=None,  # Cognito owns the credential; no local password.
        cognito_sub=sub,
    )
    db.add(user)
    try:
        db.commit()
    except Exception:  # noqa: BLE001 — lost a race against a concurrent request
        db.rollback()
        return db.query(User).filter(User.cognito_sub == sub).first()
    db.refresh(user)
    logger.info("Provisioned user for Cognito sub %s (%s)", sub, identity["email"] or "no email")
    return user


def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    # Behind Databricks Apps, `Authorization` is consumed by the platform's own
    # OAuth proxy before the request reaches us — the app's access token travels
    # in this header instead so the two auth layers don't collide. `Authorization`
    # is still accepted directly (local dev, or anything not behind that proxy).
    x_drp_token: str | None = Header(default=None, alias="X-Drp-Token"),
    db: Session = Depends(get_db),
) -> User:
    """
    Resolve the caller's token to a `users` row, or 401.

    The token is read from `X-Drp-Token`, then bearer `Authorization`, and — only
    when `DRP_ALLOW_QUERY_TOKEN` is set — an `access_token` query parameter. The
    query fallback exists because the AWS Lambda proxy forwards query params and
    bodies but not headers, leaving a per-user token nowhere else to ride.

    The token may be either a Cognito RS256 token (the frontend's Amplify
    session) or one of this API's own HS256 tokens from `/v1/auth/login`. Cognito
    is tried first and is distinguished by its `alg`, so the two never collide.
    """
    token = x_drp_token or (credentials.credentials if credentials else None)
    if not token and settings.DRP_ALLOW_QUERY_TOKEN:
        token = request.query_params.get("access_token") or None
    if not token:
        raise _UNAUTHENTICATED

    claims = cognito.decode_cognito_token(token)
    if claims is not None:
        user = _user_from_cognito(db, claims)
        if user is None:
            raise _UNAUTHENTICATED
        return user

    payload = decode_access_token(token)
    if not payload:
        raise _UNAUTHENTICATED
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        raise _UNAUTHENTICATED
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise _UNAUTHENTICATED
    return user


def get_profile(db: Session, user: User) -> DrpUserProfile:
    """Fetch (creating on first access) the DRP profile row for a user."""
    profile = db.query(DrpUserProfile).filter(DrpUserProfile.user_id == user.id).first()
    if profile is None:
        profile = DrpUserProfile(
            user_id=user.id,
            display_name=user.full_name or user.username or "",
            role="Researcher",
            therapeutic_areas=[],
            onboarding_completed=False,
            onboarding_step=1,
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile
