"""NovSearch Agent — patent novelty assessment (spec §2, §6)."""
from typing import Any, Dict, Optional

from DRP_Main.app.agents.base.base_agent import BaseAgent
from DRP_Main.app.core.async_utils import run_sync
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

# Fields a ScreenSuite carry-over supplies, in the order the query string is built.
_CARRYOVER_FIELDS = ("target", "drug", "disease", "candidate_id", "docking_id")


class NovSearchAgent(BaseAgent):
    """
    Calls the NovSearch functional-spec tools in sequence
    (`modules/novelty/novsearch_service`): PatentsView retrieval and reranking,
    indexing into Databricks Vector Search, then one SaulLM synthesis call.

    The agent branches on the input shape before Tool 1 runs (§2): a resolved
    drug-target-disease combination carried forward from ScreenSuite, or a fresh
    standalone query.
    """

    def __init__(self):
        super().__init__(tools=[])

    @property
    def name(self) -> str:
        return "novsearch"

    def execute(
        self, input_data: Dict[str, Any], session_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        session_state = session_state or {}
        payload = self._build_payload(input_data, session_state)
        if not payload:
            return {
                "error": "No query provided",
                "recommendation": "Please provide a drug/target/disease query.",
            }

        num_results = int(input_data.get("num_results", 5))
        try:
            result = run_sync(self._run(payload, num_results))
        except Exception as exc:  # noqa: BLE001
            logger.error("Novelty search failed: %s", exc)
            return {"error": str(exc), "recommendation": "Try a different query."}

        report = result.get("report", {})
        # §6: the supervisor persists this so a follow-up question knows what is
        # indexed without re-running the chain.
        session_state.update(result.get("session_state_update", {}))

        verdict: Dict[str, Any] = {
            "assessment": report.get("agent_answer", ""),
            "recommendations": report.get("recommendations", ""),
            "model_used": report.get("model_used"),
        }
        out: Dict[str, Any] = {
            "module": "novsearch",
            "input_source": result.get("input_source"),
            "query": result.get("query", ""),
            "patent_findings": result.get("patents_table", []),
            "novelty_verdict": verdict,
            "session_state_update": result.get("session_state_update", {}),
            "recommendation": (
                f"Analyzed {report.get('total_patents', 0)} patents for "
                f"'{result.get('query', '')}'. See recommendations for details."
            ),
        }
        # Present only for a carry-over, so the supervisor can tie this report
        # back to the screening result that triggered it.
        for key in ("candidate_id", "docking_id"):
            if result.get(key):
                out[key] = result[key]
        return out

    def answer_followup(
        self, question: str, patent_ids: Optional[list] = None
    ) -> Dict[str, Any]:
        """Follow-up Q&A scoped to one, several, or all indexed patents (§5)."""
        from DRP_Main.app.modules.novelty import novsearch_service as ns

        return run_sync(ns.answer_followup(question, patent_ids))

    # ── §2: pick the input shape ─────────────────────────────────────────────
    @staticmethod
    def _build_payload(
        input_data: Dict[str, Any], session_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Case A when a resolved combination is available — from the message itself
        or carried in session state by ScreenSuite. Case B otherwise.
        """
        payload = {
            field: (input_data.get(field) or session_state.get(field) or "").strip()
            for field in _CARRYOVER_FIELDS
        }
        payload = {k: v for k, v in payload.items() if v}
        if any(payload.get(k) for k in ("target", "drug", "disease")):
            payload["source"] = "screensuite_carryover"
            return payload

        query = (input_data.get("message") or input_data.get("query") or "").strip()
        if not query:
            return {}
        return {"source": "user_direct", "query": query}

    @staticmethod
    async def _run(payload: Dict[str, Any], num_results: int) -> Dict[str, Any]:
        from DRP_Main.app.modules.novelty import novsearch_service as ns

        return await ns.run_novsearch(payload, num_results)
