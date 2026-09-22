"""LitMineX Agent — calls the spec's three tools in sequence (spec §1)."""
from typing import Any, Dict, List, Optional

from DRP_Main.app.agents.base.base_agent import BaseAgent
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.literature.litminex_service import (
    LitMineXService,
    MissingEntityError,
)

logger = get_logger(__name__)


class LitMineXAgent(BaseAgent):
    """
    Literature mining agent: query expansion + retrieval → relevance scoring →
    summarisation, returning the final object the supervisor merges into shared
    session state.
    """

    def __init__(self):
        super().__init__(tools=[])
        self._service = LitMineXService()

    @property
    def name(self) -> str:
        return "litminex"

    def execute(self, input_data: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        session_state = session_state or {}
        session_id = str(
            input_data.get("session_id") or session_state.get("session_id") or "default"
        )

        # A follow-up over the already-retrieved set reuses Tool 3 only (spec §6).
        if input_data.get("followup") and self._service.get_session(session_id):
            return self._followup(input_data, session_id)

        payload = self._build_payload(input_data, session_state)
        try:
            result = self._service.run(
                payload,
                session_id=session_id,
                max_results=input_data.get("max_results"),
                top_n=input_data.get("top_n"),
                exclude_reviews=input_data.get("exclude_reviews"),
            )
        except MissingEntityError as exc:
            # Ask the supervisor to prompt the user once, rather than guessing (§2).
            return {
                "needs_clarification": True,
                "missing": exc.missing,
                "extracted": exc.extracted,
                "recommendation": (
                    "I need the "
                    + " and ".join(exc.missing)
                    + " to search the literature. Which should I use?"
                ),
            }
        except Exception as exc:
            logger.error("LitMineX pipeline failed: %s", exc, exc_info=True)
            return {"error": str(exc), "recommendation": "Try rephrasing the literature query."}

        return {
            **result,
            # GraphState field names the supervisor already merges for this agent.
            "literature_query": result["session_state_update"]["litminex_last_query"],
            "articles": result["article_table"],
            "recommendation": self.get_recommendation(result),
        }

    def _followup(self, input_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        question = input_data.get("question") or input_data.get("message") or ""
        try:
            return self._service.answer_followup(
                question,
                session_id,
                pmids=_as_list(input_data.get("pmids")),
                deep=bool(input_data.get("deep")),
            )
        except KeyError as exc:
            return {"error": str(exc), "recommendation": "Run a literature query first."}

    @staticmethod
    def _build_payload(input_data: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise to one of the two spec §2 input shapes.

        A target/disease carried forward by TxKG — whether passed directly or already
        sitting in shared session state — takes the structured path and skips NER.
        """
        target = input_data.get("target") or _carried_target(session_state)
        disease = input_data.get("disease") or session_state.get("disease")
        query = input_data.get("query") or input_data.get("message") or ""

        if target and disease and not input_data.get("query"):
            return {"target": target, "disease": disease, "source": "txkg_carryover"}
        return {
            "query": query,
            "target": target,
            "disease": disease,
            "source": input_data.get("source", "user_direct"),
        }

    def get_recommendation(self, results: Dict[str, Any]) -> str:
        table = results.get("article_table", [])
        if not table:
            return "No articles carried scored target-disease evidence. Try widening the query."
        cited = len(results.get("query_answer", {}).get("cited_pmids", []))
        return (
            f"{len(table)} articles ranked; the answer cites {cited}. "
            f"Top hit: {table[0].get('title', '')[:120]} "
            f"(score {table[0].get('relevance_score')})."
        )


def _carried_target(session_state: Dict[str, Any]) -> Optional[str]:
    """Highest-ranked TxKG target, if the pipeline carried one forward."""
    targets = session_state.get("targets") or []
    if not targets:
        return None
    first = targets[0]
    if isinstance(first, dict):
        return first.get("name") or first.get("geneName") or first.get("uniprotId")
    return str(first)


def _as_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]
