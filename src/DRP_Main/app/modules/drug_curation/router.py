"""
Drug Curation router.
Replaces app/api/v1/endpoints/drug_curation.py.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException

from DRP_Main.app.modules.drug_curation.schemas import (
    CurateXRunRequest,
    CurationResponse,
    FetchCompoundsRequest,
    CompoundVerificationRequest,
    ProfileRequest,
    ScoreRequest,
    VerificationResponse,
    WeightValidationRequest,
)
from DRP_Main.app.modules.drug_curation.curation_service import DrugCurationService
from DRP_Main.app.modules.drug_curation.verification_service import URLValidator, VerificationService
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()
drug_curation_service = DrugCurationService()
verification_service = VerificationService()


@router.post("/fetch_compounds", response_model=CurationResponse)
async def fetch_compounds(request: FetchCompoundsRequest):
    """Curate drug compounds using Groq AI + optional PubMed integration."""
    try:
        criteria_text = request.criteria or request.story_content
        if not criteria_text:
            raise HTTPException(status_code=422, detail="Either 'criteria' or 'story_content' must be provided")
        if len(criteria_text.strip()) < 10:
            raise HTTPException(status_code=422, detail="Criteria must be at least 10 characters long")
        result = drug_curation_service.curate_compounds(
            criteria_text,
            request.num_results,
            skip_pubmed=request.skip_pubmed,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Fetch compounds failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fetch compounds failed: {str(e)}")


@router.post("/verify_compounds", response_model=VerificationResponse)
async def verify_compounds(request: CompoundVerificationRequest):
    """Verify drug compounds against their source websites."""
    try:
        if not request.compounds:
            raise HTTPException(status_code=422, detail="Compounds list cannot be empty")
        result = verification_service.verify_compounds(request.compounds)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Verify compounds failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# ── CurateX (functional spec) ─────────────────────────────────────────────────
@router.post("/curatex/profile")
async def curatex_profile(request: ProfileRequest):
    """
    Tool 1 (§5) — resolve the target/disease, pull known ligands from the six
    §2.1 sources and return the editable 19-criteria weighted profile.

    With a disease but no target, responds `stage: awaiting_target_confirmation`
    and the upstream shortlist instead of choosing a target.
    """
    import asyncio

    from DRP_Main.app.modules.drug_curation.curatex_service import CurateXError, CurateXService

    try:
        return await asyncio.to_thread(
            CurateXService().build_profile,
            target=request.target,
            disease=request.disease,
            targets=request.targets,
            candidate_targets=[t.model_dump() for t in (request.candidateTargets or [])],
            weights=request.weights,
            skip_dailymed=request.skipDailyMed,
            session_id=request.sessionId,
        )
    except CurateXError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("CurateX profile build failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile build failed: {str(e)}")


@router.post("/curatex/score")
async def curatex_score(request: ScoreRequest):
    """
    Tools 2 + 3 (§6-§7) — submit the (edited) profile: exclusion filter, scoring,
    ranking, evidence retrieval and validation.
    """
    import asyncio

    from DRP_Main.app.modules.drug_curation.curatex_service import (
        CurateXError,
        CurateXService,
        get_session,
    )

    state = get_session(request.sessionId)
    if state is None or not state.profiles:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No profile is held for session '{request.sessionId}'. "
                "Call /curatex/profile first."
            ),
        )
    if request.target:
        profile = state.profiles.get(request.target)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session holds no profile for target '{request.target}'",
            )
    else:
        profile = next(iter(state.profiles.values()))

    try:
        return await asyncio.to_thread(
            CurateXService().score_candidates,
            profile,
            weights=request.weights,
            top_n=request.topN,
            universe_limit=request.universeLimit,
            min_phase=request.minPhase,
            skip_evidence=request.skipEvidence,
        )
    except CurateXError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("CurateX scoring failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Candidate scoring failed: {str(e)}")


@router.post("/curatex/run")
async def curatex_run(request: CurateXRunRequest):
    """The whole chain in one call — profile → candidates → evidence."""
    import asyncio

    from DRP_Main.app.modules.drug_curation.curatex_service import CurateXError, CurateXService

    try:
        return await asyncio.to_thread(
            CurateXService().run,
            target=request.target,
            disease=request.disease,
            targets=request.targets,
            candidate_targets=[t.model_dump() for t in (request.candidateTargets or [])],
            weights=request.weights,
            top_n=request.topN,
            universe_limit=request.universeLimit,
            min_phase=request.minPhase,
            skip_dailymed=request.skipDailyMed,
            skip_evidence=request.skipEvidence,
            session_id=request.sessionId,
        )
    except CurateXError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("CurateX run failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"CurateX run failed: {str(e)}")


@router.get("/curatex/criteria")
async def curatex_criteria():
    """The locked §4.1 criteria table with default weights — what the UI renders."""
    from DRP_Main.app.modules.drug_curation.criteria_config import (
        ALL_ROWS,
        CATEGORY_BUDGETS,
        category_totals,
    )

    return {
        "categories": CATEGORY_BUDGETS,
        "categoryTotals": category_totals(),
        "criteria": [
            {
                "key": c.key,
                "number": c.number,
                "label": c.label,
                "category": c.category,
                "kind": c.kind,
                "defaultWeight": c.default_weight or None,
                "sources": list(c.sources),
                "direction": c.direction,
                "unit": c.unit,
                "editable": c.editable,
                "valueMap": c.value_map or None,
                "note": c.note,
            }
            for c in ALL_ROWS
        ],
    }


@router.post("/curatex/validate-weights")
async def curatex_validate_weights(request: WeightValidationRequest):
    """Check an edited weight table before submitting it for scoring."""
    from DRP_Main.app.modules.drug_curation.criteria_config import (
        category_totals,
        validate_weight_table,
    )

    problems = validate_weight_table(request.weights)
    return {
        "valid": not problems,
        "problems": problems,
        "categoryTotals": category_totals(request.weights),
        "total": sum(float(v or 0) for v in request.weights.values()),
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Drug Curation Service",
        "api": "Groq",
        "max_compounds": 50,
        "min_compounds": 15,
        "pubmed_integration": True,
        "url_validation": True,
        "blocked_sites": ["Medscape", "FDA"],
        "curatex": {
            "tools": [
                "target resolution + profile builder",
                "candidate matching + scoring",
                "evidence retrieval + validation",
            ],
            "ligandSources": [
                "chembl", "open_targets", "bindingdb",
                "pubchem_bioassay", "iuphar", "dgidb",
            ],
            "criteriaSources": ["chembl", "open_targets", "dailymed"],
            "weightedCriteria": 19,
        },
        "version": "3.0.0",
    }


@router.post("/validate_url")
async def validate_url(url: str, drug_name: Optional[str] = None):
    """Validate and fix a single URL."""
    try:
        if URLValidator.is_blocked_url(url):
            return {"original_url": url, "blocked": True, "reason": "Medscape and FDA URLs are excluded"}
        fixed_url = URLValidator.fix_common_url_issues(url, drug_name)
        if not fixed_url:
            return {"original_url": url, "blocked": True, "reason": "URL is from blocked domain"}
        validation = URLValidator.validate_url_fast(fixed_url)
        return {
            "original_url": url,
            "fixed_url": fixed_url,
            "validation": validation,
            "drug_name": drug_name,
            "blocked": False,
        }
    except Exception as e:
        logger.error("URL validation test failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"URL validation failed: {str(e)}")
