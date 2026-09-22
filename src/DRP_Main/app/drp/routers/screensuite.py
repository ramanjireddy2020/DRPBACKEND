"""Agents - ScreenSuite — /agents/screensuite/*."""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import create_job, enqueue, require_completed
from DRP_Main.app.drp.models import DrpPipeline
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Agents - ScreenSuite"


@router.post(
    "/agents/screensuite/screen",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def screen(
    body: s.ScreenRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    Run virtual/high-throughput compound screening against a target.

    Supply `pdbFilePath` and `compounds` to queue a new docking run; without them
    the job returns whatever hits already exist for the target.
    """
    if not body.target.strip():
        raise HTTPException(status_code=422, detail="target is required")
    job = create_job(
        db,
        user_id=user.id,
        kind="screensuite.screen",
        module="ScreenSuite",
        params={
            "target": body.target,
            "compoundLibrary": body.compoundLibrary,
            "pdbFilePath": body.pdbFilePath,
            "compounds": body.compounds,
        },
    )

    # Surface the run on the dashboard's pipeline overview.
    db.add(
        DrpPipeline(
            id=f"pipe_{uuid.uuid4().hex[:12]}",
            user_id=user.id,
            target=body.target,
            pipeline_type="Repurposing Screening",
            progress_percent=10,
            status="In Progress",
            job_id=job.id,
        )
    )
    db.commit()

    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.get("/agents/screensuite/{jobId}/hits", response_model=list[s.ScreeningHit], tags=[TAG])
def screening_hits(jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Get virtual screening hits / binding results."""
    job = require_completed(db, jobId, user.id)
    return [s.ScreeningHit(**hit) for hit in (job.result or {}).get("hits", [])]
