"""
JWT access tokens + opaque refresh tokens for the DRP `/v1` API.

Uses `settings.SECRET_KEY` / `settings.ALGORITHM` (not the hardcoded key in
`app/core/security.py`, which predates this module).
"""
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from DRP_Main.app.core.config import settings
from DRP_Main.app.drp.models import DrpRefreshToken

ACCESS_TOKEN_TTL_SECONDS = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

# bcrypt is used directly rather than through passlib: passlib 1.7.4 cannot read
# the version metadata of bcrypt >= 4.1 and fails at hash time.
_BCRYPT_MAX_BYTES = 72


def _encode(password: str) -> bytes:
    """bcrypt ignores bytes past 72 — truncate explicitly instead of erroring."""
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_encode(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_encode(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):  # malformed/legacy hash
        return False


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "typ": "access",
        "iat": int(now.timestamp()),
        "exp": now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Return the token payload, or None if invalid/expired/wrong type."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
    if payload.get("typ") != "access":
        return None
    return payload


def issue_refresh_token(db: Session, user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    db.add(
        DrpRefreshToken(
            id=str(uuid.uuid4()),
            token=token,
            user_id=user_id,
            expires_at=datetime.now(timezone.utc).replace(tzinfo=None)
            + timedelta(days=settings.DRP_REFRESH_TOKEN_EXPIRE_DAYS),
        )
    )
    db.commit()
    return token


def resolve_refresh_token(db: Session, token: str) -> DrpRefreshToken | None:
    row = db.query(DrpRefreshToken).filter(DrpRefreshToken.token == token).first()
    if not row or row.revoked:
        return None
    if row.expires_at and row.expires_at < datetime.now(timezone.utc).replace(tzinfo=None):
        return None
    return row


def revoke_user_tokens(db: Session, user_id: int) -> int:
    count = (
        db.query(DrpRefreshToken)
        .filter(DrpRefreshToken.user_id == user_id, DrpRefreshToken.revoked == False)  # noqa: E712
        .update({"revoked": True})
    )
    db.commit()
    return count
