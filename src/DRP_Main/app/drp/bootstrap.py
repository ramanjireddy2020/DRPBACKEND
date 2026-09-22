"""
Startup wiring for the DRP `/v1` API: create tables, seed a login-able account.

Called once from `app/main.py`. Safe to call repeatedly and never fatal — on a
read-only or unreachable database the API still boots (endpoints that need the DB
will then error individually).
"""
from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.db.base import Base
from DRP_Main.app.db.session import engine
from DRP_Main.app.drp import models as drp_models  # noqa: F401 — registers tables
from DRP_Main.app.drp.jobs import JobSession
from DRP_Main.app.drp.models import DrpUserProfile
from DRP_Main.app.drp.security import hash_password
from DRP_Main.app.models.user import User  # noqa: F401 — `users` table

logger = get_logger(__name__)


def _ensure_cognito_sub_column() -> None:
    """
    Add `users.cognito_sub` to a database created before Cognito auth existed.

    `create_all` only creates missing tables, never missing columns, so an
    already-deployed database would keep a `users` table without this column and
    fail every Cognito lookup. There is no migration tool in this project, and
    the change is one nullable column, so it is applied directly.
    """
    from sqlalchemy import inspect, text

    try:
        columns = {c["name"] for c in inspect(engine).get_columns("users")}
    except Exception as exc:  # noqa: BLE001 — table not there yet is fine
        logger.debug("users table not inspectable (%s)", exc)
        return

    if "cognito_sub" in columns:
        return

    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN cognito_sub VARCHAR"))
            conn.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_cognito_sub ON users (cognito_sub)")
            )
        logger.info("Added users.cognito_sub")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not add users.cognito_sub (%s) — Cognito auth will fail", exc)


def _ensure_step_columns() -> None:
    """
    Add the step-chain columns to a database created before sessions had steps.

    `create_all` creates the new `drp_session_steps` table on its own but will not
    add columns to tables that already exist, so the two foreign keys pointing at
    it have to be applied directly — same reasoning as `_ensure_cognito_sub_column`.
    """
    from sqlalchemy import inspect, text

    additions = (
        ("drp_session_messages", "step_id", "VARCHAR"),
        ("drp_jobs", "session_step_id", "VARCHAR"),
    )
    inspector = inspect(engine)
    for table, column, sql_type in additions:
        try:
            columns = {c["name"] for c in inspector.get_columns(table)}
        except Exception as exc:  # noqa: BLE001 — table not there yet is fine
            logger.debug("%s not inspectable (%s)", table, exc)
            continue
        if column in columns:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
            logger.info("Added %s.%s", table, column)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not add %s.%s (%s)", table, column, exc)


def init_drp() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        _ensure_cognito_sub_column()
        _ensure_step_columns()
    except Exception as exc:  # noqa: BLE001
        logger.warning("DRP tables not created (%s) — /v1 persistence unavailable", exc)
        return

    # Jobs are asyncio tasks in this process; a restart kills them silently and
    # their rows would otherwise stay `running` for ever.
    try:
        from DRP_Main.app.drp.jobs import reap_orphaned_jobs

        reap_orphaned_jobs()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Job reaper skipped: %s", exc)

    if not settings.DRP_SEED_DEMO_USER:
        return

    db = JobSession()
    try:
        email = settings.DRP_DEMO_EMAIL
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            user = User(
                username=email.split("@")[0],
                email=email,
                full_name=settings.DRP_DEMO_NAME,
                hashed_password=hash_password(settings.DRP_DEMO_PASSWORD),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info("Seeded DRP demo user %s", email)

        if db.query(DrpUserProfile).filter(DrpUserProfile.user_id == user.id).first() is None:
            db.add(
                DrpUserProfile(
                    user_id=user.id,
                    display_name=settings.DRP_DEMO_NAME,
                    role=settings.DRP_DEMO_ROLE,
                    therapeutic_areas=[],
                    onboarding_completed=False,
                    onboarding_step=1,
                )
            )
            db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("DRP demo user seed skipped: %s", exc)
    finally:
        db.close()
