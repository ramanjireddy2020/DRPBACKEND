"""Main supervisor agent that orchestrates all specialist agents via LangGraph."""
from typing import Any, Dict, Optional

from DRP_Main.app.agents.base.base_agent import BaseAgent
from DRP_Main.app.supervisor.state_manager import StateManager
from DRP_Main.app.supervisor.graph import get_compiled_graph
from DRP_Main.app.supervisor.graph_state import session_state_to_graph_state
from DRP_Main.app.session.models import SessionTypes
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Main orchestrator. Owns session bootstrapping and delegates routing/execution
    to the compiled LangGraph StateGraph (see supervisor/graph.py).
    """

    def __init__(self):
        super().__init__(tools=[])
        self._state_manager = StateManager()

    @property
    def name(self) -> str:
        return "supervisor"

    def execute(self, input_data: Dict[str, Any], session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user message and route to the appropriate agent via the LangGraph.

        Args:
            input_data: Must contain 'message' and optionally 'session_id'
            session_state: Unused; retained for BaseAgent signature compatibility.

        Returns:
            Response with agent output and recommendation.
        """
        message = input_data.get("message", "")
        session_id = input_data.get("session_id")

        # Get or create session (used only for session_id bootstrap + legacy JSON export)
        state = self._state_manager.get_state(session_id) if session_id else None
        if not state:
            state = self._state_manager.create_state()
            session_id = state.session_id

        graph = get_compiled_graph()
        config = {"configurable": {"thread_id": session_id}}
        graph_input = session_state_to_graph_state(state, message=message)
        result = graph.invoke(graph_input, config=config)

        agent_name = result.get("agent_used", "")
        agent_output = result.get("agent_output", {})
        intent = SessionTypes(result.get("intent", SessionTypes.FRESH_QUERY.value))

        # Keep the legacy in-memory SessionState mirror up to date for any code
        # still reading through StateManager/SessionMemory directly.
        self._state_manager.update_state(session_id, dict(result))

        next_step = self._get_next_step(agent_name, intent)

        return {
            "session_id": session_id,
            "agent_used": agent_name,
            "intent": intent.value,
            "output": agent_output,
            "next_step": next_step,
        }

    def _get_next_step(self, agent_name: str, intent: SessionTypes) -> str:
        """Generate the next step question for the user."""
        if intent == SessionTypes.FULL_PIPELINE:
            return "Would you like to continue with the next step in the pipeline?"
        
        if intent == SessionTypes.GENERAL_QUERY:
            return "How else can I help you with your data analysis?"
            
        if intent == SessionTypes.FOLLOW_UP:
            return "What would you like to explore next?"
            
        return "What would you like to do next?"


# Global supervisor instance
supervisor = SupervisorAgent()