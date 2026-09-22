"""
Pydantic schemas for the Drug Curation module.
Extracted from app/api/v1/endpoints/drug_curation.py.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional


class VerificationDetail(BaseModel):
    website: str
    status: str
    mentions: int
    content_length: Optional[int] = 0


class CompoundResult(BaseModel):
    compound_name: str = Field(..., description="Name of the drug compound", example="Metformin")
    content: Dict = Field(..., description="Detailed drug profile information")
    confidence_score: float = Field(..., description="Confidence score (0-100)", example=87.5, ge=0, le=100)
    exact_criteria_match: Dict = Field(..., description="Criteria matching analysis")
    websites: List[str] = Field(default_factory=list, description="Relevant websites and PubMed links")
    pubmed_articles: List[Dict] = Field(default_factory=list, description="PubMed article details")
    pubmed_query_used: str = Field(default="", description="Query used for PubMed search")

    @field_validator("websites")
    @classmethod
    def validate_urls(cls, v, info):
        """Filter blocked domains and validate URLs."""
        from DRP_Main.app.modules.drug_curation.verification_service import URLValidator
        if not v:
            return v
        compound_name = info.data.get("compound_name", "")
        filtered_urls = [url for url in v if not URLValidator.is_blocked_url(url)]
        if not filtered_urls:
            return []
        return URLValidator.validate_and_fix_urls(filtered_urls, compound_name)


class CurationResponse(BaseModel):
    compounds: List[CompoundResult]
    total_compounds: int
    criteria: str
    search_metadata: Dict


class CompoundVerificationResult(BaseModel):
    compound: str
    verification_details: List[VerificationDetail]
    overall_status: str
    summary: Dict


class VerificationResponse(BaseModel):
    verification_results: List[CompoundVerificationResult]
    summary: Dict


class FetchCompoundsRequest(BaseModel):
    story_content: Optional[str] = Field(None, description="Detailed criteria (alternative field name)", min_length=10)
    criteria: Optional[str] = Field(None, description="Search criteria for drug compound selection", min_length=10)
    num_results: int = Field(20, description="Number of compounds (15-50)", ge=15, le=50)
    validate_urls: bool = Field(True, description="Whether to validate URLs")
    skip_pubmed: bool = Field(False, description="Skip PubMed enhancement for faster results")

    def model_post_init(self, __context):
        if self.criteria is None and self.story_content is not None:
            self.criteria = self.story_content
        elif self.criteria is None and self.story_content is None:
            raise ValueError("Either criteria or story_content must be provided")


class CompoundVerificationRequest(BaseModel):
    compounds: List[Dict] = Field(..., description="List of compounds to verify with their website sources")


# ── CurateX (functional spec) ─────────────────────────────────────────────────
class CandidateTargetRef(BaseModel):
    """One row of the upstream target shortlist carried in from target identification."""

    name: str = Field(..., description="Gene symbol or protein name")
    uniprotId: Optional[str] = None
    score: Optional[float] = None
    category: Optional[str] = None


class ProfileRequest(BaseModel):
    """Tool 1 — target resolution and profile builder (§5)."""

    target: Optional[str] = Field(None, description="Gene symbol or protein name", example="JAK2")
    targets: Optional[List[str]] = Field(
        None, description="Several targets — one profile object is built per target"
    )
    disease: Optional[str] = Field(
        None, description="Disease name; resolved to MeSH + EFO and used by the exclusion filter"
    )
    candidateTargets: Optional[List[CandidateTargetRef]] = Field(
        None,
        description=(
            "Target shortlist from upstream scoring. Sent when only a disease is known — "
            "CurateX surfaces it for confirmation rather than selecting a target itself."
        ),
    )
    weights: Optional[Dict[str, float]] = Field(
        None, description="Criterion key → weight. Omit to use the §4.1 defaults."
    )
    skipDailyMed: bool = Field(
        False, description="Skip SPL enrichment (criteria 15-19 then carry no data)"
    )
    sessionId: Optional[str] = Field(None, description="Binds the profile to a session for reuse")


class ScoreRequest(BaseModel):
    """Tool 2 + Tool 3 — submit an (optionally edited) profile for scoring (§6-§7)."""

    sessionId: str = Field(..., description="Session whose profile is being submitted")
    target: Optional[str] = Field(
        None, description="Which profile to score when the session holds several"
    )
    weights: Optional[Dict[str, float]] = Field(
        None, description="Edited weights; a weight of 0 excludes that criterion"
    )
    topN: int = Field(25, ge=1, le=200, description="Candidates returned")
    universeLimit: int = Field(500, ge=50, le=5000, description="Candidate universe size")
    minPhase: int = Field(
        1, ge=0, le=4, description="Minimum max-clinical-phase for the candidate universe"
    )
    skipEvidence: bool = Field(False, description="Skip Tool 3 (no evidence links, no validation)")


class CurateXRunRequest(ProfileRequest):
    """The whole chain in one call: profile → candidates → evidence."""

    topN: int = Field(25, ge=1, le=200)
    universeLimit: int = Field(500, ge=50, le=5000)
    minPhase: int = Field(1, ge=0, le=4)
    skipEvidence: bool = Field(False)


class WeightValidationRequest(BaseModel):
    weights: Dict[str, float] = Field(..., description="Criterion key → weight")
