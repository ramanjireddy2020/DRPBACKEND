"""Projects — /projects, /projects/{projectId}[/items|/results]."""
import math
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.catalog import canonical_module
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import get_job
from DRP_Main.app.drp.models import DrpProject, DrpProjectItem, DrpSession, utcnow
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Projects"
PAGE_SIZE = 20

_SORT_COLUMNS = {
    "Latest Activity": DrpProject.updated_at.desc(),
    "Name": DrpProject.name.asc(),
    "Created": DrpProject.created_at.desc(),
}


def _to_wire(project: DrpProject) -> s.Project:
    return s.Project(
        id=project.id,
        name=project.name,
        disease=project.disease or "",
        module=project.module or "",
        status=project.status or "Active",
        updatedAt=project.updated_at,
    )


def _own_project(db: Session, project_id: str, user_id: int) -> DrpProject:
    project = (
        db.query(DrpProject)
        .filter(DrpProject.id == project_id, DrpProject.user_id == user_id)
        .first()
    )
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return project


@router.get("/projects", response_model=s.ProjectPage, tags=[TAG])
def list_projects(
    search: Optional[str] = Query(None),
    module: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    sortBy: str = Query("Latest Activity"),
    page: int = Query(1, ge=1),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """List / search / filter projects."""
    query = db.query(DrpProject).filter(DrpProject.user_id == user.id)
    if search:
        like = f"%{search}%"
        query = query.filter(or_(DrpProject.name.ilike(like), DrpProject.disease.ilike(like)))
    if module and module.lower() != "all":
        query = query.filter(DrpProject.module == canonical_module(module))
    if status_filter:
        query = query.filter(DrpProject.status == status_filter)

    total = query.count()
    order = _SORT_COLUMNS.get(sortBy, _SORT_COLUMNS["Latest Activity"])
    rows = query.order_by(order).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    return s.ProjectPage(
        items=[_to_wire(row) for row in rows],
        totalCount=total,
        page=page,
        totalPages=max(1, math.ceil(total / PAGE_SIZE)),
    )


@router.post(
    "/projects", response_model=s.Project, status_code=status.HTTP_201_CREATED, tags=[TAG]
)
def create_project(
    body: s.CreateProjectRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Create a new project."""
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="name is required")
    project = DrpProject(
        id=f"prj_{uuid.uuid4().hex[:12]}",
        user_id=user.id,
        name=body.name.strip(),
        disease=body.disease,
        module=canonical_module(body.module),
        status=body.status,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _to_wire(project)


@router.get("/projects/{projectId}", response_model=s.Project, tags=[TAG])
def get_project(
    projectId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Get project metadata."""
    return _to_wire(_own_project(db, projectId, user.id))


@router.patch("/projects/{projectId}", response_model=s.Project, tags=[TAG])
def update_project(
    projectId: str,
    body: s.UpdateProjectRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Rename a project or change its status (the "Set Status" / "Edit Project" actions)."""
    project = _own_project(db, projectId, user.id)
    if body.name is not None:
        if not body.name.strip():
            raise HTTPException(status_code=422, detail="name cannot be empty")
        project.name = body.name.strip()[:200]
    if body.disease is not None:
        project.disease = body.disease.strip()
    if body.module is not None:
        project.module = canonical_module(body.module)
    if body.status is not None:
        project.status = body.status
    project.updated_at = utcnow()
    db.commit()
    db.refresh(project)
    return _to_wire(project)


@router.delete("/projects/{projectId}", response_model=s.SuccessResponse, tags=[TAG])
def delete_project(
    projectId: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Delete a project and the saved results filed under it.

    Sessions are left alone: a session is research in its own right and outlives
    the project it happened to be filed under.
    """
    project = _own_project(db, projectId, user.id)
    db.query(DrpSession).filter(
        DrpSession.project_id == projectId, DrpSession.user_id == user.id
    ).update({"project_id": None}, synchronize_session=False)
    db.delete(project)
    db.commit()
    return s.SuccessResponse(success=True, message=f"Project '{projectId}' deleted")


@router.get("/projects/{projectId}/items", response_model=list[s.ProjectItem], tags=[TAG])
def project_items(
    projectId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Get all saved results/items within a project."""
    _own_project(db, projectId, user.id)
    rows = (
        db.query(DrpProjectItem)
        .filter(DrpProjectItem.project_id == projectId)
        .order_by(DrpProjectItem.created_at.desc())
        .all()
    )
    return [
        s.ProjectItem(
            id=row.id,
            resultType=row.result_type,
            sessionId=row.session_id,
            resultId=row.result_id,
            payload=row.payload or {},
            createdAt=row.created_at,
        )
        for row in rows
    ]


# What each result type pulls out of a job result when saved into a project.
_RESULT_EXTRACTORS = {
    "targets": lambda result: {
        "disease": result.get("disease"),
        "count": result.get("count"),
        "targets": result.get("targets", []),
        "interpretation": result.get("interpretation", ""),
    },
    "subgraph": lambda result: {
        "disease": result.get("disease"),
        "graph": result.get("graph", {}),
        "stats": result.get("stats", {}),
    },
    "metapath": lambda result: {
        "disease": result.get("disease"),
        "summary": result.get("summary", {}),
        "scores": result.get("scores", []),
    },
    "litminex_results": lambda result: {
        "totalArticles": result.get("totalArticles", 0),
        "targetIds": result.get("targetIds", []),
        "items": result.get("items", []),
    },
}


@router.post("/projects/{projectId}/results", response_model=s.SaveResultResponse, tags=[TAG])
def save_result(
    projectId: str,
    body: s.SaveResultRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    Save a research result (targets, graph, article list) into a project.

    `resultId` is the job id that produced the result — or, for `resultType:
    article`, the article id.
    """
    project = _own_project(db, projectId, user.id)

    if body.resultType == "article":
        from DRP_Main.app.drp.models import DrpArticle

        article = db.query(DrpArticle).filter(DrpArticle.id == body.resultId).first()
        if article is None:
            raise HTTPException(status_code=404, detail=f"Article '{body.resultId}' not found")
        payload = {"title": article.title, "pmcLink": article.pmc_link, "year": article.year}
    else:
        job = get_job(db, body.resultId, user.id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Result '{body.resultId}' not found")
        if job.status != "completed":
            raise HTTPException(
                status_code=409, detail=f"Job '{body.resultId}' has not completed yet"
            )
        extractor = _RESULT_EXTRACTORS.get(body.resultType)
        payload = extractor(job.result or {}) if extractor else (job.result or {})

    item = DrpProjectItem(
        id=f"item_{uuid.uuid4().hex[:12]}",
        project_id=project.id,
        session_id=body.sessionId,
        result_id=body.resultId,
        result_type=body.resultType,
        payload=payload,
    )
    db.add(item)

    if body.sessionId:
        session = (
            db.query(DrpSession)
            .filter(DrpSession.id == body.sessionId, DrpSession.user_id == user.id)
            .first()
        )
        if session is not None:
            session.project_id = project.id
            session.status = "Saved"

    project.updated_at = utcnow()
    db.commit()
    return s.SaveResultResponse(success=True, projectName=project.name, itemId=item.id)
