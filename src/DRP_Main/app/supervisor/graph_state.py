"""LangGraph state schema for the supervisor graph."""
from typing import Any, Dict, List, Optional, TypedDict

from DRP_Main.app.session.models import SessionState


class GraphState(TypedDict, total=False):
    """Shared state carried through the supervisor StateGraph.

    Mirrors DRP_Main.app.session.models.SessionState field-for-field so no
    field is silently dropped on conversion (SessionState.to_dict() omits
    several of its own fields, which this schema does not repeat).
    """

    session_id: str
    created_at: str
    updated_at: str

    message: str
    intent: str
    agent_used: str

    # TxKG
    disease: Optional[str]
    targets: List[Dict[str, Any]]
    subgraph_artifact: Optional[Dict[str, Any]]

    # LitMineX
    literature_query: Optional[str]
    articles: List[Dict[str, Any]]

    # CurateX
    selected_drugs: List[Dict[str, Any]]
    drug_profile: Optional[Dict[str, Any]]

    # ScreenSuite
    docking_results: List[Dict[str, Any]]
    top_hits: List[Dict[str, Any]]
    interaction_artifact: Optional[Dict[str, Any]]

    # NovSearch
    patent_findings: List[Dict[str, Any]]
    novelty_verdict: Optional[Dict[str, Any]]

    # Genie
    genie_response: Optional[str]

    messages: List[Dict[str, str]]
    last_agent: Optional[str]
    agent_output: Dict[str, Any]


def session_state_to_graph_state(session: SessionState, message: str = "") -> GraphState:
    """Build a GraphState from a SessionState for a new graph turn."""
    return GraphState(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message=message,
        intent="",
        agent_used="",
        disease=session.disease,
        targets=session.targets,
        subgraph_artifact=session.subgraph_artifact,
        literature_query=session.literature_query,
        articles=session.articles,
        selected_drugs=session.selected_drugs,
        drug_profile=session.drug_profile,
        docking_results=session.docking_results,
        top_hits=session.top_hits,
        interaction_artifact=session.interaction_artifact,
        patent_findings=session.patent_findings,
        novelty_verdict=session.novelty_verdict,
        genie_response=None,
        messages=session.messages,
        last_agent=session.last_agent,
        agent_output={},
    )


def apply_graph_state_to_session(session: SessionState, state: GraphState) -> SessionState:
    """Copy every GraphState field the SessionState dataclass understands back onto it."""
    for key, value in state.items():
        if hasattr(session, key):
            setattr(session, key, value)
    return session
