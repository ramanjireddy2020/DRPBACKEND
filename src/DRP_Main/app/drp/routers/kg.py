"""Knowledge Graph — /agents/subgraph/*, /agents/metapath/*."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import JobContext, create_job, enqueue, get_job, require_completed
from DRP_Main.app.drp.runners import RunnerError, run_subgraph_explore
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Knowledge Graph"


@router.post(
    "/agents/subgraph/generate",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def generate_subgraph(
    body: s.SubgraphRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Generate a knowledge sub-graph for a disease/target."""
    if not body.disease.strip():
        raise HTTPException(status_code=422, detail="disease is required")
    job = create_job(
        db,
        user_id=user.id,
        kind="subgraph.generate",
        module="TxKG",
        params={
            "disease": body.disease,
            "targetIds": body.targetIds,
            "selectedTarget": body.selectedTarget,
            "maxNodes": body.maxNodes,
        },
    )
    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.get("/agents/subgraph/{jobId}", response_model=s.KnowledgeGraph, tags=[TAG])
def get_subgraph(jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Get the generated graph (nodes/edges + legend)."""
    job = require_completed(db, jobId, user.id)
    graph = (job.result or {}).get("graph", {}) or {}
    return s.KnowledgeGraph(**graph)


@router.get("/agents/subgraph/{jobId}/stats", response_model=s.GraphStats, tags=[TAG])
def subgraph_stats(jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Graph summary stats — relationships found, drug candidates, pathway connections."""
    job = require_completed(db, jobId, user.id)
    return s.GraphStats(**((job.result or {}).get("stats", {}) or {}))


@router.post("/agents/subgraph/{jobId}/explore", response_model=s.KnowledgeGraph, tags=[TAG])
async def explore_subgraph(
    jobId: str,
    body: s.ExploreRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Explore Connections — expand the graph around a selected node."""
    require_completed(db, jobId, user.id)
    ctx = JobContext(jobId, user.id, None)
    try:
        expanded = await run_subgraph_explore({"nodeId": body.nodeId}, ctx)
    except RunnerError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 — TxKG data may be unavailable
        raise HTTPException(status_code=502, detail=f"Knowledge graph unavailable: {exc}")
    return s.KnowledgeGraph(**expanded)


@router.post(
    "/agents/metapath/analyze",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def analyze_metapaths(
    body: s.MetapathRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Run meta-path analysis over a generated graph."""
    source = get_job(db, body.jobId, user.id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Job '{body.jobId}' not found")
    disease = (source.result or {}).get("disease") or (source.params or {}).get("disease")
    if not disease:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{body.jobId}' has no resolved disease to analyse — "
            "generate a sub-graph first.",
        )
    job = create_job(
        db,
        user_id=user.id,
        kind="metapath.analyze",
        module="TxKG",
        params={"disease": disease, "sourceJobId": body.jobId, "limit": 10, "maxHops": 3},
    )
    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.get("/agents/metapath/{jobId}", response_model=s.MetapathSummary, tags=[TAG])
def metapath_summary(jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Meta-path analysis summary (paths, targets, pathways, nodes, edges, clusters)."""
    job = require_completed(db, jobId, user.id)
    return s.MetapathSummary(jobId=job.id, **((job.result or {}).get("summary", {}) or {}))


@router.get(
    "/agents/metapath/{jobId}/scores", response_model=list[s.TargetPredictionScore], tags=[TAG]
)
def metapath_scores(jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Target Prediction Scores."""
    job = require_completed(db, jobId, user.id)
    return [s.TargetPredictionScore(**row) for row in (job.result or {}).get("scores", [])]


@router.get(
    "/agents/metapath/{jobId}/traversals", response_model=list[s.MetapathTraversal], tags=[TAG]
)
def metapath_traversals(
    jobId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Meta-Path Traversals per target."""
    job = require_completed(db, jobId, user.id)
    return [s.MetapathTraversal(**row) for row in (job.result or {}).get("traversals", [])]
