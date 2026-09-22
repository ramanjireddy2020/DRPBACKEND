"""Agents - LitMinex — /agents/litminex/*."""
import math
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import create_job, enqueue, require_completed
from DRP_Main.app.drp.models import DrpArticle, DrpJob
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Agents - LitMinex"
PAGE_SIZE = 20

# Kind used to persist ad-hoc targets the user typed into the selection list.
_CUSTOM_TARGETS_KIND = "litminex.custom_targets"


@router.post("/agents/litminex/targets/custom", response_model=s.SuccessResponse, tags=[TAG])
def add_custom_target(
    body: s.CustomTargetRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Add a custom target to the LitMinex target selection list."""
    name = body.targetName.strip()
    if not name:
        raise HTTPException(status_code=422, detail="targetName cannot be empty")

    row = (
        db.query(DrpJob)
        .filter(DrpJob.user_id == user.id, DrpJob.kind == _CUSTOM_TARGETS_KIND)
        .first()
    )
    if row is None:
        row = DrpJob(
            id=f"custom_{uuid.uuid4().hex[:12]}",
            user_id=user.id,
            kind=_CUSTOM_TARGETS_KIND,
            module="LitMineX",
            status="completed",
            params={},
            result={"targets": []},
        )
        db.add(row)

    targets = list((row.result or {}).get("targets", []))
    if name not in targets:
        targets.append(name)
    row.result = {"targets": targets}
    db.commit()
    return s.SuccessResponse(success=True, message=f"'{name}' added to the target list")


@router.get("/agents/litminex/targets/custom", response_model=list[str], tags=[TAG])
def list_custom_targets(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """List the custom targets previously added to the selection list."""
    row = (
        db.query(DrpJob)
        .filter(DrpJob.user_id == user.id, DrpJob.kind == _CUSTOM_TARGETS_KIND)
        .first()
    )
    return list((row.result or {}).get("targets", [])) if row else []


@router.post(
    "/agents/litminex/query",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def litminex_query(
    body: s.LitminexQueryRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Run literature mining across the selected targets."""
    if not body.targetIds and not body.query.strip():
        raise HTTPException(status_code=422, detail="targetIds or query is required")
    job = create_job(
        db,
        user_id=user.id,
        kind="litminex.query",
        module="LitMineX",
        params={
            "targetIds": body.targetIds,
            "query": body.query,
            "maxResults": body.maxResults,
        },
    )
    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.get("/agents/litminex/{jobId}/results", response_model=s.ArticlePage, tags=[TAG])
def litminex_results(
    jobId: str,
    page: int = Query(1, ge=1),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get paginated ranked article results."""
    job = require_completed(db, jobId, user.id)
    items = (job.result or {}).get("items", [])
    start = (page - 1) * PAGE_SIZE
    return s.ArticlePage(
        totalArticles=len(items),
        page=page,
        totalPages=max(1, math.ceil(len(items) / PAGE_SIZE)),
        items=[s.ArticleSummary(**item) for item in items[start : start + PAGE_SIZE]],
    )


@router.get("/agents/litminex/{jobId}/insights", response_model=s.InsightPanel, tags=[TAG])
def litminex_insights(
    jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Get the Article Relevance insights panel."""
    job = require_completed(db, jobId, user.id)
    result = job.result or {}
    items = result.get("items", [])
    if not items:
        return s.InsightPanel(tab="interpretation", content="No articles matched this query.")

    scores = [float(item.get("confidenceScore", 0) or 0) for item in items]
    high = [item for item, score in zip(items, scores) if score >= 80]

    # LitMineX Tool 3's query-scoped answer, built only from the extracted relation
    # sentences and cited to source PMIDs. The ranking stats are the supporting note.
    answer = (result.get("queryAnswer") or {}).get("text", "").strip()
    stats = (
        f"{len(items)} articles ranked, {len(high)} scoring 80 or above. "
        f"Mean relevance {sum(scores) / len(scores):.1f}."
    )
    content = f"{answer}\n\n{stats}" if answer else stats

    return s.InsightPanel(
        tab="interpretation",
        content=content,
        items=[f"{item.get('title', '')} — {item.get('confidenceScore')}" for item in high[:5]],
    )


@router.get(
    "/agents/litminex/{jobId}/results/{articleId}/preview",
    response_model=s.ArticlePreview,
    tags=[TAG],
)
def litminex_preview(
    jobId: str,
    articleId: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Quick preview (eye icon) of an article row."""
    require_completed(db, jobId, user.id)
    article = db.query(DrpArticle).filter(DrpArticle.id == articleId).first()
    if article is None:
        raise HTTPException(status_code=404, detail=f"Article '{articleId}' not found")
    snippet = (article.preview or article.abstract or "")[:600]
    return s.ArticlePreview(
        id=article.id,
        title=article.title or "",
        snippet=snippet,
        confidenceScore=article.confidence_score or 0.0,
        foundKeywords=article.found_keywords or [],
    )
