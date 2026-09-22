"""
NovSearch endpoints — the spec-aligned chain (§2-§6).

`/novsearch/run` takes either input shape and returns the supervisor object.
`/novsearch/ask` answers a follow-up in single, multiple or all-patent scope.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from DRP_Main.app.modules.novelty import novsearch_service
from DRP_Main.app.modules.novelty.novsearch_service import AmbiguousQueryError
from DRP_Main.app.modules.novelty.schemas import (
    NovSearchQARequest,
    NovSearchQAResponse,
    NovSearchRequest,
    NovSearchResponse,
)

router = APIRouter()


@router.post(
    "/novsearch/run",
    response_model=NovSearchResponse,
    # candidate_id / docking_id are omitted rather than null-filled for a fresh
    # query, so consumers can branch on their presence (§6).
    response_model_exclude_none=True,
)
async def run(body: NovSearchRequest):
    """Tool 1 → Tool 2 → Tool 3: retrieval, indexing, and the novelty report."""
    try:
        return await novsearch_service.run_novsearch(
            body.model_dump(exclude_none=True), num_results=body.num_results
        )
    except AmbiguousQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/novsearch/ask", response_model=NovSearchQAResponse)
async def ask(body: NovSearchQARequest):
    """Follow-up Q&A scoped to one, several, or all indexed patents."""
    try:
        return await novsearch_service.answer_followup(
            body.question, body.patent_ids, body.top_k
        )
    except AmbiguousQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/novsearch/status")
def status():
    """What is indexed, and which Databricks backends are wired up."""
    return novsearch_service.status()
