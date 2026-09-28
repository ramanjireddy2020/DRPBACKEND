"""
Async agent-job lifecycle for the DRP `/v1` API.

The spec models long-running agent work as `202 {jobId}` + polling
(`/agents/jobs/{jobId}/status`, `/agents/jobs/{jobId}/result`). This module owns
that lifecycle; the actual science lives in `runners.py`, which delegates to the
existing pipeline modules.

Jobs run as asyncio tasks in the API process. Each job gets its own SQLAlchemy
session (never the request-scoped one) so progress writes are independent of the
request that created it.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from sqlalchemy.orm import Session, sessionmaker

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.db.session import engine
from DRP_Main.app.drp.models import DrpJob, utcnow

logger = get_logger(__name__)

# Independent session factory — job tasks must not share the request session.
JobSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class JobContext:
    """Handed to a runner: identity plus a progress reporter."""

    def __init__(self, job_id: str, user_id: int, session_id: Optional[str]):
        self.job_id = job_id
        self.user_id = user_id
        self.session_id = session_id

    def progress(self, message: str) -> None:
        """Publish a human-readable progress line for the polling endpoint."""
        logger.info("job %s: %s", self.job_id, message)
        db = JobSession()
        try:
            db.query(DrpJob).filter(DrpJob.id == self.job_id).update(
                {"progress_message": message, "updated_at": utcnow()}
            )
            db.commit()
        except Exception:  # noqa: BLE001 — progress must never fail a job
            db.rollback()
            logger.warning("could not record progress for job %s", self.job_id)
        finally:
            db.close()


Runner = Callable[[Dict[str, Any], JobContext], Awaitable[Dict[str, Any]]]
RUNNERS: Dict[str, Runner] = {}


def register(kind: str) -> Callable[[Runner], Runner]:
    """Decorator registering a runner under a job kind, e.g. `txkg.query`."""

    def _wrap(fn: Runner) -> Runner:
        RUNNERS[kind] = fn
        return fn

    return _wrap


def create_job(
    db: Session,
    *,
    user_id: int,
    kind: str,
    module: str,
    params: Dict[str, Any],
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    session_step_id: Optional[str] = None,
) -> DrpJob:
    """Persist a queued job. Call `enqueue()` afterwards to start it."""
    job = DrpJob(
        id=f"job_{uuid.uuid4().hex[:16]}",
        user_id=user_id,
        kind=kind,
        module=module,
        status="queued",
        progress_message="Queued",
        params=params,
        result={},
        session_id=session_id,
        session_step_id=session_step_id,
        project_id=project_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def reap_orphaned_jobs() -> int:
    """
    Fail any job left `queued`/`running` by a previous process.

    Jobs run as asyncio tasks inside the API process, so a restart kills them
    with no chance to record the fact. Without this their rows stay `running`
    for ever and the UI polls a job that will never finish. Called once at
    startup, before any new job can be created.
    """
    db = JobSession()
    try:
        orphans = (
            db.query(DrpJob)
            .filter(DrpJob.status.in_(("queued", "running")))
            .update(
                {
                    "status": "failed",
                    "error": "Interrupted by an API restart — please run this step again.",
                    "progress_message": "Interrupted",
                    "updated_at": utcnow(),
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if orphans:
            logger.warning("reaped %d job(s) orphaned by a restart", orphans)
        return orphans
    except Exception as exc:  # noqa: BLE001 — never block startup
        db.rollback()
        logger.warning("could not reap orphaned jobs: %s", exc)
        return 0
    finally:
        db.close()


_loop: Optional[asyncio.AbstractEventLoop] = None


def capture_event_loop() -> None:
    """
    Remember the server's event loop. Called from the app's startup hook.

    Job creation happens inside `def` (threadpool) endpoints, where there is no
    running loop, so `enqueue` needs an explicit handle to schedule work on.
    """
    global _loop
    _loop = asyncio.get_running_loop()


def enqueue(job: DrpJob) -> None:
    """Start a queued job as a background task on the server's event loop."""
    coro = _execute(job.id, job.kind, job.user_id, job.session_id, dict(job.params or {}))
    try:
        asyncio.get_running_loop().create_task(coro)
        return
    except RuntimeError:
        pass  # called from a threadpool worker — fall through to the stored loop

    if _loop is None or _loop.is_closed():
        coro.close()
        _finish(job.id, status="failed", error="No event loop is available to run agent jobs")
        return
    asyncio.run_coroutine_threadsafe(coro, _loop)


async def _execute(
    job_id: str, kind: str, user_id: int, session_id: Optional[str], params: Dict[str, Any]
) -> None:
    runner = RUNNERS.get(kind)
    ctx = JobContext(job_id, user_id, session_id)

    if runner is None:
        _finish(job_id, status="failed", error=f"No runner registered for job kind '{kind}'")
        return

    _set_running(job_id)
    try:
        result = await runner(params, ctx)
    except Exception as exc:  # noqa: BLE001 — surface as a failed job, never a crash
        logger.exception("job %s (%s) failed", job_id, kind)
        # A runner raises a deliberate, readable message for things the researcher
        # can act on (unknown disease, unmatched gene symbol). Prefixing those with
        # the exception class turns advice into what reads as a stack trace —
        # "RunnerError: No reviewed human UniProt entry matched…" — so the class
        # name is kept only for genuinely unexpected failures.
        message = str(exc) if getattr(exc, "user_facing", False) else f"{type(exc).__name__}: {exc}"
        _finish(job_id, status="failed", error=message)
        return
    _finish(job_id, status="completed", result=result, progress_message="Completed")


def _set_running(job_id: str) -> None:
    db = JobSession()
    try:
        db.query(DrpJob).filter(DrpJob.id == job_id).update(
            {"status": "running", "progress_message": "Starting…", "updated_at": utcnow()}
        )
        db.commit()
    finally:
        db.close()


def _finish(
    job_id: str,
    *,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
    progress_message: Optional[str] = None,
) -> None:
    db = JobSession()
    try:
        values: Dict[str, Any] = {"status": status, "error": error, "updated_at": utcnow()}
        if result is not None:
            values["result"] = result
        if progress_message is not None:
            values["progress_message"] = progress_message
        elif status == "failed":
            values["progress_message"] = error or "Failed"
        db.query(DrpJob).filter(DrpJob.id == job_id).update(values)
        db.commit()
    finally:
        db.close()


def get_job(db: Session, job_id: str, user_id: int) -> Optional[DrpJob]:
    return (
        db.query(DrpJob)
        .filter(DrpJob.id == job_id, DrpJob.user_id == user_id)
        .first()
    )


def require_completed(db: Session, job_id: str, user_id: int) -> DrpJob:
    """Fetch a job, 404 if unknown, 409 if it has not produced a result yet."""
    from fastapi import HTTPException

    job = get_job(db, job_id, user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status == "failed":
        raise HTTPException(status_code=409, detail=job.error or "Job failed")
    if job.status != "completed":
        # Worded for a researcher, not a developer: the UI surfaces this verbatim
        # when it requests results before polling, and "poll /agents/jobs/.../status"
        # reads like a crash. `detail` stays a **string** — every other error on this
        # API is one, and a client doing `detail` expecting text breaks on an object.
        raise HTTPException(
            status_code=409,
            detail="This step is still running. Results will appear once it finishes.",
        )
    return job
