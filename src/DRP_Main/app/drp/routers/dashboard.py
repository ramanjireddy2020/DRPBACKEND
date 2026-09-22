"""Dashboard — /dashboard/summary, /quick-actions, /pipelines[, /manage]."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.catalog import QUICK_ACTIONS
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.models import DrpJob, DrpPipeline, DrpProject
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Dashboard"

# Progress attributed to a pipeline from the state of its backing job.
_JOB_PROGRESS = {"queued": 10, "running": 55, "completed": 100, "failed": 0}


def _sum_result_field(db: Session, user_id: int, kinds: list[str], field: str) -> int:
    jobs = (
        db.query(DrpJob)
        .filter(DrpJob.user_id == user_id, DrpJob.kind.in_(kinds), DrpJob.status == "completed")
        .all()
    )
    total = 0
    for job in jobs:
        value = (job.result or {}).get(field)
        if isinstance(value, (int, float)):
            total += int(value)
        elif isinstance(value, list):
            total += len(value)
    return total


@router.get("/dashboard/summary", response_model=s.DashboardSummary, tags=[TAG])
def summary(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Home dashboard stat cards."""
    active_projects = (
        db.query(DrpProject)
        .filter(DrpProject.user_id == user.id, DrpProject.status == "Active")
        .count()
    )
    return s.DashboardSummary(
        activeProjects=active_projects,
        targetsIdentified=_sum_result_field(db, user.id, ["txkg.query"], "count"),
        compoundsCurated=_sum_result_field(db, user.id, ["curatex.compounds"], "totalCompounds"),
        patentsAnalysed=_sum_result_field(db, user.id, ["novsearch.assess"], "totalPatents"),
    )


@router.get("/dashboard/quick-actions", response_model=list[s.QuickAction], tags=[TAG])
def quick_actions(user: User = Depends(current_user)):
    """Suggested quick action prompts for the composer."""
    return QUICK_ACTIONS


def _progress_for(db: Session, pipeline: DrpPipeline) -> int:
    if pipeline.job_id:
        job = db.query(DrpJob).filter(DrpJob.id == pipeline.job_id).first()
        if job is not None:
            return _JOB_PROGRESS.get(job.status, pipeline.progress_percent or 0)
    return pipeline.progress_percent or 0


@router.get("/dashboard/pipelines", response_model=list[s.PipelineProgress], tags=[TAG])
def pipelines(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Pipeline overview — repurposing screening progress per target."""
    rows = (
        db.query(DrpPipeline)
        .filter(DrpPipeline.user_id == user.id)
        .order_by(DrpPipeline.updated_at.desc())
        .all()
    )
    return [
        s.PipelineProgress(
            target=row.target,
            pipelineType=row.pipeline_type,
            progressPercent=_progress_for(db, row),
        )
        for row in rows
    ]


@router.get("/dashboard/pipelines/manage", response_model=list[s.PipelineDetail], tags=[TAG])
def manage_pipelines(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Full pipeline management view (the 'Manage Pipelines' link)."""
    rows = (
        db.query(DrpPipeline)
        .filter(DrpPipeline.user_id == user.id)
        .order_by(DrpPipeline.updated_at.desc())
        .all()
    )
    details = []
    for row in rows:
        progress = _progress_for(db, row)
        status = row.status
        if row.job_id:
            job = db.query(DrpJob).filter(DrpJob.id == row.job_id).first()
            if job is not None:
                status = {"completed": "Completed", "failed": "Failed"}.get(
                    job.status, "In Progress"
                )
        details.append(
            s.PipelineDetail(
                id=row.id,
                target=row.target,
                pipelineType=row.pipeline_type,
                progressPercent=progress,
                status=status,
                projectId=row.project_id,
                jobId=row.job_id,
                updatedAt=row.updated_at,
            )
        )
    return details
