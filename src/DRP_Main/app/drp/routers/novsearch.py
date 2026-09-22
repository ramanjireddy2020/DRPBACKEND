"""Agents - NovSearch — /agents/novsearch/*."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import create_job, enqueue, require_completed
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Agents - NovSearch"


@router.post(
    "/agents/novsearch/assess",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def assess(
    body: s.NoveltyAssessRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    Run the patent novelty assessment.

    Accepts either NovSearch input shape: a ScreenSuite carry-over (target / drug /
    disease, optionally with candidateId + dockingId) or a fresh free-text query.
    """
    carryover = any((getattr(body, f) or "").strip() for f in ("target", "drug", "disease"))
    if not carryover and not (body.query or "").strip():
        raise HTTPException(
            status_code=422,
            detail="Provide either a query, or a target/drug/disease combination.",
        )
    job = create_job(
        db,
        user_id=user.id,
        kind="novsearch.assess",
        module="NovSearch",
        params={
            "target": body.target,
            "drug": body.drug,
            "disease": body.disease,
            "query": body.query,
            "candidateId": body.candidateId,
            "dockingId": body.dockingId,
            "numResults": body.numResults,
        },
    )
    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.get(
    "/agents/novsearch/{jobId}/report",
    response_model=s.NoveltyReport,
    # candidateId / dockingId are omitted, not null-filled, for a fresh query.
    response_model_exclude_none=True,
    tags=[TAG],
)
def report(jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Get the novelty assessment report."""
    job = require_completed(db, jobId, user.id)
    result = job.result or {}
    return s.NoveltyReport(
        jobId=job.id,
        query=result.get("query", ""),
        inputSource=result.get("inputSource", "user_direct"),
        candidateId=result.get("candidateId"),
        dockingId=result.get("dockingId"),
        target=result.get("target", ""),
        disease=result.get("disease", ""),
        assessment=result.get("assessment", ""),
        recommendations=result.get("recommendations", []) or [],
        patents=result.get("patents", []) or [],
        patentsUsed=result.get("patentsUsed", []) or [],
        totalPatents=result.get("totalPatents", 0),
        totalChunks=result.get("totalChunks", 0),
        modelUsed=result.get("modelUsed"),
    )


@router.post("/agents/novsearch/ask", response_model=s.NoveltyQAResponse, tags=[TAG])
async def ask(
    body: s.NoveltyQARequest,
    user: User = Depends(current_user),
):
    """
    Follow-up question about the indexed patents.

    Scope follows `patentIds`: one id answers from that patent alone, several
    answer from just that subset, and omitting it runs against every indexed patent.
    """
    from DRP_Main.app.modules.novelty import novsearch_service

    try:
        result = await novsearch_service.answer_followup(
            body.question, body.patentIds, body.topK
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return s.NoveltyQAResponse(
        answer=result["answer"],
        mode=result["mode"],
        patentIdsUsed=result.get("patent_ids_used", []),
        chunksUsed=result.get("chunks_used", 0),
    )
