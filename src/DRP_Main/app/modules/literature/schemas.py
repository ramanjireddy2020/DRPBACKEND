"""
Pydantic schemas for the Literature Mining module.

`ArticleResult` / `SearchResponse` back the legacy keyword endpoint. Everything below
the divider is the LitMineX functional spec's contract (§3–§7).
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ArticleResult(BaseModel):
    protein_name: str = Field(..., description="Protein names from keywords")
    title: str = Field(..., description="Article title")
    score: float = Field(..., description="Relevance score (0-100)", ge=0, le=100)
    pdf_file_path: str = Field(..., description="Direct PDF URL")
    found_keywords: List[str] = Field(..., description="Matched keywords")
    preview: str = Field(..., description="Article preview text")


class SearchResponse(BaseModel):
    results: List[ArticleResult] = Field(..., description="Relevant articles")
    total_count: int = Field(..., description="Total results returned")
    article_keywords: List[str] = Field(..., description="Search keywords used")
    search_keywords: List[str] = Field(..., description="Scoring keywords used")
    execution_time: float = Field(..., description="Query execution time in seconds")


# ── LitMineX (functional spec) ───────────────────────────────────────────────
class LitMineXRequest(BaseModel):
    """Either input shape from spec §2 — structured carry-over or a fresh query."""

    target: Optional[str] = Field(None, description="Target/protein, e.g. 'AMPK'")
    disease: Optional[str] = Field(None, description="Disease, e.g. 'Type 2 Diabetes'")
    query: Optional[str] = Field(None, description="Raw natural-language question")
    source: Literal["txkg_carryover", "user_direct"] = "user_direct"
    session_id: Optional[str] = Field(
        None, description="Caches the ranked set so follow-up QA can be scoped to it"
    )
    max_results: Optional[int] = Field(
        None, ge=1, le=500, description="PMID retrieval cap (default LITMINEX_RETRIEVAL_LIMIT)"
    )
    top_n: Optional[int] = Field(None, ge=1, le=50, description="Articles fed to Tool 3")
    exclude_reviews: Optional[bool] = Field(
        None, description="Restrict to primary research articles"
    )

    @model_validator(mode="after")
    def _needs_something(self) -> "LitMineXRequest":
        if not (self.query or self.target or self.disease):
            raise ValueError("Provide 'query', or 'target' and/or 'disease'")
        return self


class RelationSentence(BaseModel):
    text: str
    section: str
    relation_found: bool
    relation_phrase: Optional[str] = None
    evidence_type: Literal["primary", "cited"]
    position_weight: float
    similarity_score: float


class RankedArticle(BaseModel):
    pmid: str
    title: str = ""
    relevance_score: float = Field(..., ge=0, le=100)
    relation_sentences: List[RelationSentence] = []
    pdf_link: str = ""
    abstract_preview: str = ""
    journal: str = ""
    pub_date: Optional[str] = None
    authors: List[str] = []
    full_text_available: bool = False


class QueryAnswer(BaseModel):
    text: str
    cited_pmids: List[str] = []


class ArticleTableRow(BaseModel):
    pmid: str
    title: str = ""
    relevance_score: float = 0.0
    pdf_link: str = ""
    journal: str = ""
    pub_date: Optional[str] = None
    per_article_summary: str = ""


class SessionStateUpdate(BaseModel):
    litminex_last_query: str
    litminex_articles: List[str] = []


class LitMineXResponse(BaseModel):
    """The object the supervisor merges into shared session state (spec §7)."""

    module: Literal["litminex"] = "litminex"
    target: Optional[str] = None
    disease: Optional[str] = None
    query_answer: QueryAnswer
    article_table: List[ArticleTableRow] = []
    session_state_update: SessionStateUpdate
    ranked_articles: List[RankedArticle] = []
    query_terms: Dict[str, List[str]] = {}
    pubmed_query: str = ""
    pipeline_backends: Dict[str, Any] = {}


class LitMineXFollowUpRequest(BaseModel):
    """Spec §6 — QA scoped to one article, several, or the full retrieved set."""

    question: str = Field(..., min_length=1)
    session_id: str = Field(..., description="Session whose retrieved set to answer over")
    pmids: Optional[List[str]] = Field(
        None, description="Scope to these PMIDs; omit for the whole retrieved set"
    )
    deep: bool = Field(
        False, description="Widen context from relation sentences to full section text"
    )


class LitMineXFollowUpResponse(BaseModel):
    module: Literal["litminex"] = "litminex"
    question: str
    scope_pmids: List[str] = []
    query_answer: QueryAnswer


class ClarificationResponse(BaseModel):
    """Returned instead of results when the query names neither concept (spec §2)."""

    module: Literal["litminex"] = "litminex"
    needs_clarification: bool = True
    missing: List[str]
    extracted: Dict[str, Optional[str]] = {}
    prompt: str
