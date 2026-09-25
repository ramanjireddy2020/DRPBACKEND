"""Agents - CuraTeX — /agents/curatex/*."""
import math
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import create_job, enqueue, require_completed
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Agents - CuraTeX"


@router.post(
    "/agents/curatex/target-profile",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def target_profile(
    body: s.TargetProfileRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Generate a target candidate profile."""
    if not body.target.strip():
        raise HTTPException(status_code=422, detail="target is required")
    job = create_job(
        db,
        user_id=user.id,
        kind="curatex.target_profile",
        module="CurateX",
        params={
            "target": body.target,
            "disease": body.disease,
            "weights": body.weights,
            "values": body.values,
            "numResults": 20,
        },
        session_id=body.sourceSessionId,
    )
    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.post(
    "/agents/curatex/compounds",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def curate_compounds(
    body: s.CurateCompoundsRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Curate compounds for a given target (e.g. 'Curate compounds for Aspirin')."""
    if not (body.target or "").strip() and not (body.compound or "").strip():
        raise HTTPException(status_code=422, detail="target or compound is required")
    job = create_job(
        db,
        user_id=user.id,
        kind="curatex.compounds",
        module="CurateX",
        params={
            "target": body.target,
            "compound": body.compound,
            "disease": body.disease,
            "weights": body.weights,
            "values": body.values,
            "numResults": body.numResults,
        },
    )
    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.get("/agents/curatex/{jobId}/results", response_model=s.CompoundPage, tags=[TAG])
def curatex_results(
    jobId: str,
    page: int = Query(1, ge=1),
    pageSize: int = Query(10, ge=1, le=100),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> s.CompoundPage:
    """
    Ranked compounds from a completed CurateX job.

    Paginated to match the results table's "Showing 1-6 of 124 compounds" footer;
    every other module exposes its results this way, and CurateX previously did
    not, leaving the UI to read the raw job payload.
    """
    job = require_completed(db, jobId, user.id)
    result: Dict[str, Any] = job.result or {}
    compounds = result.get("compounds", []) or []

    start = (page - 1) * pageSize
    window = compounds[start : start + pageSize]
    return s.CompoundPage(
        totalCompounds=result.get("totalCompounds", len(compounds)),
        page=page,
        totalPages=max(1, math.ceil(len(compounds) / pageSize)),
        target=result.get("target", ""),
        items=window,
    )


@router.get("/agents/curatex/{jobId}/profile", response_model=s.TargetProfile, tags=[TAG])
def curatex_profile(
    jobId: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> s.TargetProfile:
    """The target product profile and its scoring criteria, as shown before scoring."""
    job = require_completed(db, jobId, user.id)
    result: Dict[str, Any] = job.result or {}
    profile = result.get("profile", {}) or {}
    return s.TargetProfile(
        target=result.get("target", ""),
        profile=profile,
        criteria=result.get("criteria", profile.get("criteria", [])) or [],
        ligandCount=result.get("ligandCount", profile.get("ligandCount", 0)) or 0,
        editable=bool(result.get("editable", True)),
        warnings=result.get("warnings", []) or [],
    )
