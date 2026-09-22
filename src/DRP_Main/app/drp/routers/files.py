"""Files & Export — /files/upload, /exports, /exports/pdf and their downloads."""
import csv
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import get_job
from DRP_Main.app.drp.models import DrpExport, DrpFile
from DRP_Main.app.models.user import User

logger = get_logger(__name__)
router = APIRouter()
TAG = "Files & Export"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@router.post("/files/upload", response_model=s.UploadResponse, status_code=201, tags=[TAG])
async def upload_file(
    file: UploadFile = File(...),
    sessionId: str | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Upload a file or dataset to attach to a query/session."""
    file_id = f"file_{uuid.uuid4().hex[:12]}"
    safe_name = _SAFE_NAME.sub("_", os.path.basename(file.filename or "upload.bin"))
    try:
        directory = _ensure_dir(settings.DRP_UPLOAD_DIR)
        stored = directory / f"{file_id}_{safe_name}"
        contents = await file.read()
        stored.write_bytes(contents)
    except OSError as exc:
        raise HTTPException(status_code=507, detail=f"Could not store the upload: {exc}")

    db.add(
        DrpFile(
            id=file_id,
            user_id=user.id,
            file_name=safe_name,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(contents),
            stored_path=str(stored),
            session_id=sessionId,
        )
    )
    db.commit()
    return s.UploadResponse(fileId=file_id, fileName=safe_name, sizeBytes=len(contents))


@router.get("/files/{fileId}/download", tags=[TAG])
def download_file(fileId: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Download a previously uploaded file."""
    row = db.query(DrpFile).filter(DrpFile.id == fileId, DrpFile.user_id == user.id).first()
    if row is None or not os.path.exists(row.stored_path):
        raise HTTPException(status_code=404, detail=f"File '{fileId}' not found")
    return FileResponse(row.stored_path, filename=row.file_name, media_type=row.content_type)


# ── Export ────────────────────────────────────────────────────────────────────
def _result_for_export(db: Session, result_id: str, user_id: int) -> Dict[str, Any]:
    job = get_job(db, result_id, user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job '{result_id}' has not completed yet")
    return job.result or {}


def _tabular_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pick the most table-like collection out of a job result."""
    for key in ("targets", "items", "hits", "scores", "compounds", "patents"):
        rows = result.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return rows
    graph = result.get("graph") or {}
    if graph.get("nodes"):
        return graph["nodes"]
    raise HTTPException(status_code=422, detail="This result has no tabular data to export")


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
        )
    path.write_text(buffer.getvalue(), encoding="utf-8")


_NODE_COLORS = {
    "Disease Hub": "#0225AA",
    "Protein": "#1E88E5",
    "Pathway": "#065B52",
    "Compound": "#00897B",
    "Comorbidity": "#E61919",
}


def _write_graph_svg(result: Dict[str, Any], path: Path) -> None:
    """Render a graph result as a standalone SVG using a circular layout."""
    import math

    graph = result.get("graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not nodes:
        raise HTTPException(status_code=422, detail="This result contains no graph to export")

    size, radius = 1000, 420
    center = size / 2
    positions: Dict[str, tuple[float, float]] = {}
    hubs = [n for n in nodes if n.get("type") == "Disease Hub"]
    ring = [n for n in nodes if n not in hubs]
    for hub in hubs:
        positions[hub["id"]] = (center, center)
    for index, node in enumerate(ring):
        angle = (2 * math.pi * index) / max(1, len(ring))
        positions[node["id"]] = (
            center + radius * math.cos(angle),
            center + radius * math.sin(angle),
        )

    def esc(text: Any) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}"><rect width="100%" height="100%" fill="#ffffff"/>'
    ]
    for edge in edges:
        src, tgt = positions.get(edge.get("source")), positions.get(edge.get("target"))
        if not src or not tgt:
            continue
        parts.append(
            f'<line x1="{src[0]:.1f}" y1="{src[1]:.1f}" x2="{tgt[0]:.1f}" y2="{tgt[1]:.1f}" '
            f'stroke="#c7ccd6" stroke-width="1"/>'
        )
    for node in nodes:
        x, y = positions.get(node["id"], (center, center))
        color = _NODE_COLORS.get(node.get("type", ""), "#A7A4E0")
        r = 14 if node.get("type") == "Disease Hub" else 7
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + r + 3:.1f}" y="{y + 4:.1f}" font-family="sans-serif" '
            f'font-size="10" fill="#1f2933">{esc(node.get("label", node["id"]))}</text>'
        )
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


@router.post("/exports", response_model=s.ExportResponse, tags=[TAG])
def create_export(
    body: s.ExportRequest,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Export a result (graph, target table) — e.g. 'Export Graph'."""
    result = _result_for_export(db, body.resultId, user.id)
    export_id = f"exp_{uuid.uuid4().hex[:12]}"
    directory = _ensure_dir(settings.DRP_EXPORT_DIR)

    if body.format == "json":
        path = directory / f"{export_id}.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    elif body.format == "csv":
        path = directory / f"{export_id}.csv"
        _write_csv(_tabular_rows(result), path)
    elif body.format == "svg":
        path = directory / f"{export_id}.svg"
        _write_graph_svg(result, path)
    else:  # png
        raise HTTPException(
            status_code=415,
            detail="PNG export requires a raster renderer, which is not installed. "
            "Request format 'svg' instead.",
        )

    db.add(
        DrpExport(
            id=export_id,
            user_id=user.id,
            result_id=body.resultId,
            fmt=body.format,
            stored_path=str(path),
        )
    )
    db.commit()
    base = str(request.base_url).rstrip("/")
    return s.ExportResponse(downloadUrl=f"{base}/v1/exports/{export_id}/download")


@router.post("/exports/pdf", response_model=s.ExportResponse, tags=[TAG])
def create_pdf_export(
    body: s.PdfExportRequest,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Export results as PDF (e.g. LitMinex 'Export to PDF')."""
    result = _result_for_export(db, body.resultId, user.id)
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PDF export needs the 'reportlab' package, which is not installed "
            "in this environment. Use POST /v1/exports with format 'json' or 'csv'.",
        )

    export_id = f"exp_{uuid.uuid4().hex[:12]}"
    path = _ensure_dir(settings.DRP_EXPORT_DIR) / f"{export_id}.pdf"
    styles = getSampleStyleSheet()
    story = [Paragraph(f"DRP result {body.resultId}", styles["Title"]), Spacer(1, 12)]
    if result.get("summary"):
        story += [Paragraph(str(result["summary"]), styles["Heading2"]), Spacer(1, 8)]
    if result.get("interpretation"):
        story += [Paragraph(str(result["interpretation"]), styles["BodyText"]), Spacer(1, 8)]
    for row in _safe_rows(result):
        story.append(
            Paragraph(
                " · ".join(f"<b>{k}</b>: {v}" for k, v in row.items() if not isinstance(v, (dict, list))),
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 4))

    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)
    db.add(
        DrpExport(
            id=export_id,
            user_id=user.id,
            result_id=body.resultId,
            fmt="pdf",
            stored_path=str(path),
        )
    )
    db.commit()
    base = str(request.base_url).rstrip("/")
    return s.ExportResponse(downloadUrl=f"{base}/v1/exports/{export_id}/download")


def _safe_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        return _tabular_rows(result)[:200]
    except HTTPException:
        return []


@router.get("/exports/{exportId}/download", tags=[TAG])
def download_export(
    exportId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Download a previously generated export."""
    row = db.query(DrpExport).filter(DrpExport.id == exportId, DrpExport.user_id == user.id).first()
    if row is None or not os.path.exists(row.stored_path):
        raise HTTPException(status_code=404, detail=f"Export '{exportId}' not found")
    media_types = {
        "json": "application/json",
        "csv": "text/csv",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
    }
    return FileResponse(
        row.stored_path,
        filename=os.path.basename(row.stored_path),
        media_type=media_types.get(row.fmt, "application/octet-stream"),
    )
