"""State management for supervisor-agent communication."""
from typing import Any, Dict, Optional

from DRP_Main.app.session.memory import session_store
from DRP_Main.app.session.models import SessionState
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class StateManager:
    """Manages session state access and updates for agents."""

    @staticmethod
    def get_state(session_id: str) -> Optional[SessionState]:
        """Get current session state."""
        return session_store.get_session(session_id)

    @staticmethod
    def create_state() -> SessionState:
        """Create a new session state."""
        return session_store.create_session()

    @staticmethod
    def update_state(session_id: str, updates: Dict[str, Any]) -> Optional[SessionState]:
        """Update session state with new values."""
        return session_store.update_session(session_id, updates)

    @staticmethod
    def merge_agent_output(session_id: str, agent_name: str, output: Dict[str, Any]) -> Optional[SessionState]:
        """
        Merge agent output into session state.
        
        Args:
            session_id: Session identifier
            agent_name: Name of the agent (txkg, litminex, etc.)
            output: Agent output dictionary
            
        Returns:
            Updated session state
        """
        session = StateManager.get_state(session_id)
        if not session:
            session = StateManager.create_state()

        # Merge based on agent type
        if agent_name == "txkg":
            if "targets" in output:
                session.targets = output["targets"]
            if "subgraph_artifact" in output:
                session.subgraph_artifact = output["subgraph_artifact"]
            if "disease" in output:
                session.disease = output["disease"]
                
        elif agent_name == "litminex":
            if "articles" in output:
                session.articles = output["articles"]
            if "literature_query" in output:
                session.literature_query = output["literature_query"]
                
        elif agent_name == "curatex":
            if "selected_drugs" in output:
                session.selected_drugs = output["selected_drugs"]
            if "drug_profile" in output:
                session.drug_profile = output["drug_profile"]
                
        elif agent_name == "screensuite":
            if "docking_results" in output:
                session.docking_results = output["docking_results"]
            if "top_hits" in output:
                session.top_hits = output["top_hits"]
            if "interaction_artifact" in output:
                session.interaction_artifact = output["interaction_artifact"]
                
        elif agent_name == "novsearch":
            if "patent_findings" in output:
                session.patent_findings = output["patent_findings"]
            if "novelty_verdict" in output:
                session.novelty_verdict = output["novelty_verdict"]

        session.last_agent = agent_name
        
        return session_store.update_session(session_id, session.to_dict())