"""
SQLAlchemy models backing the DRP `/v1` API.

All tables are prefixed `drp_` and live in the same database as the existing
auth/items scaffolding (`settings.DATABASE_URL`). The existing `users` table is
reused for credentials; DRP-specific profile fields live in `drp_user_profiles`
so no migration of `users` is required.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from DRP_Main.app.db.base import Base


def utcnow() -> datetime:
    """Timezone-aware UTC now — stored naive-UTC by SQLite."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DrpUserProfile(Base):
    """DRP-facing profile for a row in `users` (display name, role, onboarding)."""

    __tablename__ = "drp_user_profiles"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    display_name = Column(String, default="")
    role = Column(String, default="Researcher")
    avatar_url = Column(String, default="")
    therapeutic_areas = Column(JSON, default=list)
    onboarding_completed = Column(Boolean, default=False)
    onboarding_step = Column(Integer, default=1)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class DrpRefreshToken(Base):
    __tablename__ = "drp_refresh_tokens"

    id = Column(String, primary_key=True)
    token = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)


class DrpProject(Base):
    __tablename__ = "drp_projects"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    name = Column(String, nullable=False)
    disease = Column(String, default="")
    module = Column(String, default="")
    status = Column(String, default="Active")  # Active | On Hold | Review
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    items = relationship("DrpProjectItem", back_populates="project", cascade="all, delete-orphan")


class DrpProjectItem(Base):
    __tablename__ = "drp_project_items"

    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("drp_projects.id"), index=True)
    session_id = Column(String, nullable=True)
    result_id = Column(String, nullable=True)
    result_type = Column(String, nullable=False)  # targets | subgraph | metapath | litminex_results | article
    payload = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)

    project = relationship("DrpProject", back_populates="items")


class DrpSession(Base):
    """
    A research session — one composer submission and the agent thread it opens.

    A session is a *chain of steps*, not a single module run: the researcher walks
    TxKG → LitMineX → CurateX → ScreenSuite → NovSearch in one thread, carrying
    their selections forward at each hand-off. `steps` is the source of truth for
    what happened; `module`, `status` and `summary` are denormalised headline
    fields kept in sync with the latest step so the sessions list can be rendered
    without loading every step.
    """

    __tablename__ = "drp_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    module = Column(String, default="")
    title = Column(String, default="")
    status = Column(String, default="In Progress")  # Completed | In Progress | Saved
    summary = Column(String, default="")
    query = Column(Text, default="")
    project_id = Column(String, ForeignKey("drp_projects.id"), nullable=True)
    is_draft = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    messages = relationship(
        "DrpSessionMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="DrpSessionMessage.created_at",
    )
    steps = relationship(
        "DrpSessionStep",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="DrpSessionStep.step_index",
    )


class DrpSessionStep(Base):
    """
    One module run within a session's chain.

    `selections` holds what the researcher chose on this step — the ticked target
    ids, the compounds sent to screening — which is both what the next step's
    parameters are built from and what the Lineage screen reads to show how a
    conclusion was reached.

    `parent_step_id` is normally the preceding step. A branch is a step whose
    parent is an *earlier* step rather than the latest one, so branching needs no
    further schema change.
    """

    __tablename__ = "drp_session_steps"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("drp_sessions.id"), index=True)
    parent_step_id = Column(String, ForeignKey("drp_session_steps.id"), nullable=True)
    step_index = Column(Integer, default=0)
    module = Column(String, default="")
    status = Column(String, default="In Progress")  # Completed | In Progress | Failed
    summary = Column(String, default="")
    query = Column(Text, default="")
    job_id = Column(String, ForeignKey("drp_jobs.id"), nullable=True)
    selections = Column(JSON, default=dict)
    # Set when this step re-runs an earlier one, so the UI can show attempts
    # together instead of as unrelated steps.
    rerun_of_step_id = Column(String, nullable=True)
    branch_name = Column(String, default="")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    session = relationship("DrpSession", back_populates="steps")
    messages = relationship(
        "DrpSessionMessage",
        back_populates="step",
        order_by="DrpSessionMessage.created_at",
    )


class DrpSessionMessage(Base):
    __tablename__ = "drp_session_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("drp_sessions.id"), index=True)
    # Which step of the chain this turn belongs to. Nullable: the opening user
    # message is recorded before the first step exists, and sessions created
    # before steps existed have none.
    step_id = Column(String, ForeignKey("drp_session_steps.id"), index=True, nullable=True)
    role = Column(String, default="user")  # user | agent
    agent_name = Column(String, default="")
    content = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    session = relationship("DrpSession", back_populates="messages")
    step = relationship("DrpSessionStep", back_populates="messages")


class DrpJob(Base):
    """An async agent job (TxKG query, subgraph build, screening run, …)."""

    __tablename__ = "drp_jobs"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    kind = Column(String, nullable=False, index=True)  # runner key, e.g. "txkg.query"
    module = Column(String, default="")  # TxKG | LitMinex | CuraTeX | ScreenSuite | NovSearch
    status = Column(String, default="queued")  # queued | running | completed | failed
    progress_message = Column(Text, default="")
    params = Column(JSON, default=dict)
    result = Column(JSON, default=dict)
    error = Column(Text, default="")
    session_id = Column(String, ForeignKey("drp_sessions.id"), nullable=True)
    # The step this job fulfils. Without it a session's jobs cannot be told apart,
    # which is why the old per-session sync could only ever read the newest one.
    session_step_id = Column(String, index=True, nullable=True)
    project_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class DrpArticle(Base):
    """A literature hit, persisted so /articles/{articleId} can serve detail."""

    __tablename__ = "drp_articles"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("drp_jobs.id"), index=True, nullable=True)
    pmid = Column(String, default="", index=True)
    title = Column(Text, default="")
    authors = Column(String, default="")
    year = Column(Integer, nullable=True)
    abstract = Column(Text, default="")
    keywords = Column(JSON, default=list)
    found_keywords = Column(JSON, default=list)
    confidence_score = Column(Float, default=0.0)
    preview = Column(Text, default="")
    pmc_link = Column(String, default="")
    pdf_url = Column(String, default="")
    created_at = Column(DateTime, default=utcnow)


class DrpSavedArticle(Base):
    __tablename__ = "drp_saved_articles"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    article_id = Column(String, ForeignKey("drp_articles.id"), index=True)
    project_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class DrpArticleChatMessage(Base):
    __tablename__ = "drp_article_chat_messages"

    id = Column(String, primary_key=True)
    article_id = Column(String, ForeignKey("drp_articles.id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    # Which research session the turn belongs to. Without it the thread was keyed
    # on (article, user) alone, so opening the same paper in a new session showed
    # the previous session's conversation — the article is the same object, the
    # line of enquiry is not. Nullable: turns written before this column existed
    # have no session, and are treated as belonging to none.
    session_id = Column(String, index=True, nullable=True)
    role = Column(String, default="user")  # user | agent
    content = Column(Text, default="")
    citations = Column(JSON, default=list)
    created_at = Column(DateTime, default=utcnow)


class DrpFile(Base):
    __tablename__ = "drp_files"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    file_name = Column(String, default="")
    content_type = Column(String, default="")
    size_bytes = Column(Integer, default=0)
    stored_path = Column(String, default="")
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class DrpExport(Base):
    __tablename__ = "drp_exports"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    result_id = Column(String, default="")
    fmt = Column(String, default="json")
    stored_path = Column(String, default="")
    created_at = Column(DateTime, default=utcnow)


class DrpPipeline(Base):
    """Dashboard pipeline progress rows (one per target under screening)."""

    __tablename__ = "drp_pipelines"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    target = Column(String, default="")
    pipeline_type = Column(String, default="Repurposing Screening")
    progress_percent = Column(Integer, default=0)
    status = Column(String, default="In Progress")
    project_id = Column(String, nullable=True)
    job_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
