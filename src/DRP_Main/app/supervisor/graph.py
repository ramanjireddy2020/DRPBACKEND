"""LangGraph StateGraph wiring the supervisor to the specialist agents."""
import os
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from DRP_Main.app.supervisor.graph_state import GraphState
from DRP_Main.app.supervisor.router import IntentRouter
from DRP_Main.app.session.models import SessionTypes
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_DB_PATH = os.path.join("data", "langgraph_checkpoints.db")

# Per-agent field names each agent's execute() output may contribute to GraphState,
# mirroring the mapping previously hard-coded in StateManager.merge_agent_output.
_AGENT_OUTPUT_FIELDS = {
    "txkg": ("disease", "targets", "subgraph_artifact"),
    "litminex": ("literature_query", "articles"),
    "curatex": ("selected_drugs", "drug_profile"),
    "screensuite": ("docking_results", "top_hits", "interaction_artifact"),
    "novsearch": ("patent_findings", "novelty_verdict"),
    "genie": ("genie_response",),
}


def _make_agent_node(agent_name: str, agent_factory):
    """Build a graph node that runs one specialist agent and merges its output into state."""

    def node(state: GraphState) -> Dict[str, Any]:
        agent = agent_factory()
        input_data = {"message": state.get("message", "")}
        output = agent.execute(input_data, dict(state))

        updates: Dict[str, Any] = {"agent_used": agent_name, "last_agent": agent_name, "agent_output": output}
        if agent_name == "genie" and "response" in output:
            updates["genie_response"] = output["response"]
        for field in _AGENT_OUTPUT_FIELDS.get(agent_name, ()):
            if field in output:
                updates[field] = output[field]
        return updates

    return node


def _router_node(state: GraphState) -> Dict[str, Any]:
    intent = IntentRouter.classify(state.get("message", ""), dict(state))
    return {"intent": intent.value}


def _route_to_agent(state: GraphState) -> str:
    """Conditional edge out of the router node.

    Replaces the old IntentRouter.route_to_agent, which always returned "txkg"
    for STEP_SPECIFIC intent regardless of which agent's keywords matched.
    """
    intent = SessionTypes(state.get("intent", SessionTypes.FRESH_QUERY.value))
    message_lower = state.get("message", "").lower()

    if intent == SessionTypes.GENERAL_QUERY:
        return "genie"

    if intent == SessionTypes.FOLLOW_UP:
        return state.get("last_agent") or "txkg"

    if intent == SessionTypes.FULL_PIPELINE:
        return "txkg"

    if intent == SessionTypes.STEP_SPECIFIC:
        keyword_groups = (
            ("txkg", IntentRouter.TXKG_KEYWORDS),
            ("litminex", IntentRouter.LITMEDX_KEYWORDS),
            ("curatex", IntentRouter.CURATEX_KEYWORDS),
            ("screensuite", IntentRouter.SCREENSUITE_KEYWORDS),
            ("novsearch", IntentRouter.NOVSEARCH_KEYWORDS),
        )
        for agent_name, keywords in keyword_groups:
            if any(kw in message_lower for kw in keywords):
                return agent_name
        return "txkg"

    # FRESH_QUERY default
    return "txkg"


def build_graph():
    """Construct (uncompiled) the supervisor StateGraph."""
    from DRP_Main.app.agents.txkg.agent import TxKGAgent
    from DRP_Main.app.agents.litminex.agent import LitMineXAgent
    from DRP_Main.app.agents.curatex.agent import CurateXAgent
    from DRP_Main.app.agents.screensuite.agent import ScreenSuiteAgent
    from DRP_Main.app.agents.novsearch.agent import NovSearchAgent
    from DRP_Main.app.agents.genie.agent import GenieAgent

    graph = StateGraph(GraphState)
    graph.add_node("router", _router_node)
    graph.add_node("txkg", _make_agent_node("txkg", TxKGAgent))
    graph.add_node("litminex", _make_agent_node("litminex", LitMineXAgent))
    graph.add_node("curatex", _make_agent_node("curatex", CurateXAgent))
    graph.add_node("screensuite", _make_agent_node("screensuite", ScreenSuiteAgent))
    graph.add_node("novsearch", _make_agent_node("novsearch", NovSearchAgent))
    graph.add_node("genie", _make_agent_node("genie", GenieAgent))

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_to_agent,
        {
            "txkg": "txkg",
            "litminex": "litminex",
            "curatex": "curatex",
            "screensuite": "screensuite",
            "novsearch": "novsearch",
            "genie": "genie",
        },
    )
    for agent_name in _AGENT_OUTPUT_FIELDS:
        graph.add_edge(agent_name, END)

    return graph


def _make_checkpointer():
    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


_compiled_graph = None
_checkpointer = None


def get_compiled_graph():
    """Return the module-level compiled graph, building it lazily on first use."""
    global _compiled_graph, _checkpointer
    if _compiled_graph is None:
        _checkpointer = _make_checkpointer()
        _compiled_graph = build_graph().compile(checkpointer=_checkpointer)
        logger.info("Compiled supervisor LangGraph with SQLite checkpointer at %s", CHECKPOINT_DB_PATH)
    return _compiled_graph
