"""
Literature Mining router.

`/search-keywords/` is the legacy keyword surface. `/litminex/*` is the functional
spec's three-tool pipeline.
"""
import time

from fastapi import APIRouter, HTTPException, Query
from langfuse.decorators import observe, langfuse_context

from DRP_Main.app.modules.literature.litminex_service import (
    LitMineXService,
    MissingEntityError,
)
from DRP_Main.app.modules.literature.schemas import (
    ClarificationResponse,
    LitMineXFollowUpRequest,
    LitMineXFollowUpResponse,
    LitMineXRequest,
    LitMineXResponse,
    SearchResponse,
)
from DRP_Main.app.modules.literature.scoring_service import LiteratureService

router = APIRouter()
literature_service = LiteratureService()
litminex_service = LitMineXService()


@router.get("/search-keywords/", response_model=SearchResponse)
@observe(name="literature_mining_search")
async def search_keywords(
    article_keywords: str = Query(
        ...,
        description="REQUIRED: Protein names — articles MUST contain these",
        min_length=1,
    ),
    search_keywords: str = Query(
        "",
        description="OPTIONAL: Context keywords for bonus scoring",
    ),
    max_results: int = Query(20, description="Max results", ge=1, le=100),
):
    """
    Fast literature search with parallel LLM scoring.

    - article_keywords (protein names): MANDATORY — all results contain at least one
    - search_keywords: OPTIONAL — provides bonus scoring but NOT required for inclusion
    """
    langfuse_context.update_current_observation(
        input={
            "article_keywords": article_keywords,
            "search_keywords": search_keywords,
            "max_results": max_results,
        }
    )
    start = time.time()
    article_kw = [k.strip() for k in article_keywords.split(",") if k.strip()]
    search_kw = [k.strip() for k in search_keywords.split(",") if k.strip()] if search_keywords else []

    if not article_kw:
        raise HTTPException(status_code=400, detail="article_keywords (protein names) are required")

    results = literature_service.search_keywords(article_kw, search_kw, max_results)
    execution_time = round(time.time() - start, 2)
    langfuse_context.update_current_observation(
        output={"total_count": len(results), "execution_time": execution_time}
    )
    return {
        "results": results,
        "total_count": len(results),
        "article_keywords": article_kw,
        "search_keywords": search_kw,
        "execution_time": execution_time,
    }


@router.post(
    "/litminex/query",
    response_model=LitMineXResponse,
    responses={422: {"model": ClarificationResponse}},
)
@observe(name="litminex_query_endpoint")
async def litminex_query(body: LitMineXRequest):
    """
    Run the LitMineX chain: query expansion + retrieval → relevance scoring →
    summarisation. Returns the object the supervisor merges into session state.

    Accepts either input shape — the TxKG carry-over (`target` + `disease`) or a fresh
    `query`. If a fresh query names neither concept, this responds 422 with the fields
    to prompt the user for rather than guessing.
    """
    import asyncio

    try:
        return await asyncio.to_thread(
            litminex_service.run,
            body.model_dump(exclude_none=True),
            session_id=body.session_id,
            max_results=body.max_results,
            top_n=body.top_n,
            exclude_reviews=body.exclude_reviews,
        )
    except MissingEntityError as exc:
        raise HTTPException(
            status_code=422,
            detail=ClarificationResponse(
                missing=exc.missing,
                extracted=exc.extracted,
                prompt=(
                    "I need the "
                    + " and ".join(exc.missing)
                    + " to search the literature. Which should I use?"
                ),
            ).model_dump(),
        )


@router.post("/litminex/followup", response_model=LitMineXFollowUpResponse)
@observe(name="litminex_followup_endpoint")
async def litminex_followup(body: LitMineXFollowUpRequest):
    """
    Answer a follow-up question over an already-retrieved set (spec §6), scoped to
    selected PMIDs or the whole set, citing every claim to its source PMID.
    """
    import asyncio

    try:
        return await asyncio.to_thread(
            litminex_service.answer_followup,
            body.question,
            body.session_id,
            body.pmids,
            body.deep,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\""))


@router.get("/health/")
async def health_check():
    from DRP_Main.app.core.config import settings

    return {
        "status": "healthy",
        "service": "Literature Mining",
        "version": "3.2.0",
        "config": {
            "max_workers": literature_service.max_workers,
            "litminex": {
                "retrieval_limit": settings.LITMINEX_RETRIEVAL_LIMIT,
                "summary_top_n": settings.LITMINEX_SUMMARY_TOP_N,
                "ner_backend": (
                    "scispacy"
                    if litminex_service.relevance.ner.models_available
                    else "rule-based"
                ),
                "similarity_backend": litminex_service.relevance.embeddings.backend,
            },
        },
    }
