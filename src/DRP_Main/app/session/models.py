"""Session state models for agentic conversations."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class SessionTypes(Enum):
    """Types of sessions for intent classification."""
    FULL_PIPELINE = "full_pipeline"
    STEP_SPECIFIC = "step_specific"
    FRESH_QUERY = "fresh_query"
    FOLLOW_UP = "follow_up"
    GENERAL_QUERY = "general_query"


@dataclass
class SessionState:
    """Shared session state read/written by all agents."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # TxKG state
    disease: Optional[str] = None
    targets: List[Dict[str, Any]] = field(default_factory=list)
    subgraph_artifact: Optional[Dict[str, Any]] = None
    
    # LitMineX state
    literature_query: Optional[str] = None
    articles: List[Dict[str, Any]] = field(default_factory=list)
    
    # CurateX state
    selected_drugs: List[Dict[str, Any]] = field(default_factory=list)
    drug_profile: Optional[Dict[str, Any]] = None
    
    # ScreenSuite state
    docking_results: List[Dict[str, Any]] = field(default_factory=list)
    top_hits: List[Dict[str, Any]] = field(default_factory=list)
    interaction_artifact: Optional[Dict[str, Any]] = None
    
    # NovSearch state
    patent_findings: List[Dict[str, Any]] = field(default_factory=list)
    novelty_verdict: Optional[Dict[str, Any]] = None
    
    # Conversation history
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "disease": self.disease,
            "targets": self.targets,
            "selected_drugs": self.selected_drugs,
            "docking_results": self.docking_results,
            "top_hits": self.top_hits,
            "patent_findings": self.patent_findings,
            "messages": self.messages,
            "last_agent": self.last_agent,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state