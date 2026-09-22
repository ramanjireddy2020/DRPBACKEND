"""Auth & Onboarding — /auth/*, /users/me/*, /therapeutic-areas."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.catalog import THERAPEUTIC_AREAS
from DRP_Main.app.drp.deps import current_user, get_profile
from DRP_Main.app.drp.security import (
    ACCESS_TOKEN_TTL_SECONDS,
    create_access_token,
    issue_refresh_token,
    resolve_refresh_token,
    revoke_user_tokens,
    verify_password,
)
from DRP_Main.app.models.user import User

logger = get_logger(__name__)
router = APIRouter()

TAG_AUTH = "Auth & Onboarding"


def _to_wire_user(user: User, profile) -> s.User:
    return s.User(
        id=str(user.id),
        name=profile.display_name or user.full_name or user.username or "",
        role=profile.role or "",
        avatarUrl=profile.avatar_url or "",
        email=user.email or "",
    )


@router.post("/auth/login", response_model=s.LoginResponse, tags=[TAG_AUTH])
def login(body: s.LoginRequest, db: Session = Depends(get_db)):
    """Log in with email/password (or SSO token exchange)."""
    user = db.query(User).filter(User.email == str(body.email)).first()
    if user is None or not verify_password(body.password, user.hashed_password or ""):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    profile = get_profile(db, user)
    return s.LoginResponse(
        accessToken=create_access_token(user.id),
        refreshToken=issue_refresh_token(db, user.id),
        expiresIn=ACCESS_TOKEN_TTL_SECONDS,
        user=_to_wire_user(user, profile),
    )


@router.post("/auth/logout", response_model=s.SuccessResponse, tags=[TAG_AUTH])
def logout(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Log out current session and invalidate refresh tokens."""
    revoked = revoke_user_tokens(db, user.id)
    return s.SuccessResponse(success=True, message=f"{revoked} refresh token(s) invalidated")


@router.post("/auth/refresh", response_model=s.RefreshResponse, tags=[TAG_AUTH])
def refresh(body: s.RefreshRequest, db: Session = Depends(get_db)):
    """Exchange a refresh token for a new access token."""
    row = resolve_refresh_token(db, body.refreshToken)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )
    return s.RefreshResponse(
        accessToken=create_access_token(row.user_id), expiresIn=ACCESS_TOKEN_TTL_SECONDS
    )


@router.post("/auth/forgot-password", response_model=s.SuccessResponse, tags=[TAG_AUTH])
def forgot_password(body: s.ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Request a password reset email.

    Always answers 200 so the endpoint cannot be used to enumerate accounts. Mail
    delivery is not wired up — the reset intent is logged for the operator.
    """
    if body.email:
        exists = db.query(User).filter(User.email == str(body.email)).first() is not None
        logger.info("password reset requested for %s (account exists: %s)", body.email, exists)
    return s.SuccessResponse(success=True, message="Reset email sent (if the account exists)")


@router.get("/users/me", response_model=s.User, tags=[TAG_AUTH])
def read_me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Get current authenticated user."""
    return _to_wire_user(user, get_profile(db, user))


@router.get("/users/me/onboarding-status", response_model=s.OnboardingStatus, tags=[TAG_AUTH])
def onboarding_status(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Check whether the welcome/onboarding flow should be shown."""
    profile = get_profile(db, user)
    return s.OnboardingStatus(
        completed=bool(profile.onboarding_completed), currentStep=profile.onboarding_step or 1
    )


@router.post("/users/me/research-focus", response_model=s.SuccessResponse, tags=[TAG_AUTH])
def save_research_focus(
    body: s.ResearchFocusRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Save selected therapeutic focus areas (Step 1 of onboarding)."""
    profile = get_profile(db, user)
    profile.therapeutic_areas = list(body.therapeuticAreas)
    profile.onboarding_step = max(2, profile.onboarding_step or 1)
    db.commit()
    return s.SuccessResponse(success=True, message="Saved")


@router.post("/users/me/onboarding/complete", response_model=s.SuccessResponse, tags=[TAG_AUTH])
def complete_onboarding(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Mark onboarding complete or skip."""
    profile = get_profile(db, user)
    profile.onboarding_completed = True
    db.commit()
    return s.SuccessResponse(success=True, message="Onboarding marked complete")


@router.get("/therapeutic-areas", response_model=list[str], tags=[TAG_AUTH])
def therapeutic_areas(user: User = Depends(current_user)):
    """List selectable therapeutic specialties."""
    return THERAPEUTIC_AREAS
