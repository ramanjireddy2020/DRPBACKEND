"""NovSearch request/response models for the internal `/api/v1/novelty/*` surface."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NovSearchRequest(BaseModel):
    """
    Either input shape from spec §2.

    Case A (`screensuite_carryover`): target / drug / disease plus the ids that
    tie the report back to a screening result.
    Case B (`user_direct`): free-text `query`.
    """

    source: Optional[str] = Field(
        None, examples=["screensuite_carryover", "user_direct"]
    )
    query: Optional[str] = Field(
        None, examples=["Is there existing patent coverage for JAK inhibitors?"]
    )
    target: Optional[str] = Field(None, examples=["JAK2"])
    drug: Optional[str] = Field(None, examples=["Fedratinib"])
    disease: Optional[str] = Field(None, examples=["Thrombocytosis"])
    candidate_id: Optional[str] = Field(None, examples=["cand_0091"])
    docking_id: Optional[str] = Field(None, examples=["dock_0142"])
    num_results: int = Field(5, ge=1, le=20)


class NovSearchQARequest(BaseModel):
    """
    Follow-up question. `patent_ids` chooses the QA mode: one id → single patent,
    several → the selected subset, omitted → every indexed patent.
    """

    question: str = Field(..., examples=["Who owns US10123456B2?"])
    patent_ids: Optional[List[str]] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)


class NovSearchQAResponse(BaseModel):
    answer: str
    mode: str
    patent_ids_used: List[str] = Field(default_factory=list)
    chunks_used: int = 0


class NovSearchReport(BaseModel):
    agent_answer: str = ""
    recommendations: str = ""
    patents_used: List[str] = Field(default_factory=list)
    total_patents: int = 0
    total_chunks: int = 0
    model_used: Optional[str] = None


class NovSearchResponse(BaseModel):
    """The §6 object returned to the supervisor."""

    module: str = "novsearch"
    input_source: str
    query: str
    candidate_id: Optional[str] = None
    docking_id: Optional[str] = None
    report: NovSearchReport
    patents_table: List[Dict[str, Any]] = Field(default_factory=list)
    indexing: Dict[str, Any] = Field(default_factory=dict)
    session_state_update: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}
