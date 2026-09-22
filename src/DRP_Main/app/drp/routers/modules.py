"""Modules — /modules, /modules/search, and the async agent-job endpoints."""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.catalog import MODULES
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import get_job, require_completed
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Modules"


@router.get("/modules", response_model=list[s.Module], tags=[TAG])
def list_modules(user: User = Depends(current_user)):
    """List all available research modules."""
    return MODULES


@router.get("/modules/search", response_model=list[s.Module], tags=[TAG])
def search_modules(
    q: Optional[str] = Query(None, description="Partial module name typed after '@'"),
    user: User = Depends(current_user),
):
    """Autocomplete search for '@' module mentions in the composer."""
    if not q:
        return MODULES
    needle = q.strip().lstrip("@").lower()
    return [
        module
        for module in MODULES
        if needle in module.key.lower() or needle in module.displayName.lower()
    ]


@router.get("/agents/jobs/{jobId}/status", response_model=s.JobStatusResponse, tags=[TAG])
def job_status(jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Poll async agent job status."""
    job = get_job(db, jobId, user.id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{jobId}' not found")
    return s.JobStatusResponse(
        jobId=job.id,
        status=job.status,
        progressMessage=job.progress_message or "",
        module=job.module or "",
        error=job.error or "",
    )


@router.get("/agents/jobs/{jobId}/result", tags=[TAG])
def job_result(
    jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Fetch the final result payload for a completed agent job."""
    job = require_completed(db, jobId, user.id)
    return {"jobId": job.id, "module": job.module, "kind": job.kind, "result": job.result or {}}
