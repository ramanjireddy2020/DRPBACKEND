"""CurateX Agent — drug curation per the CurateX functional specification."""
from typing import Any, Dict, List, Optional

from DRP_Main.app.agents.base.base_agent import BaseAgent
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.curatex_service import (
    AWAITING_TARGET,
    CurateXError,
    CurateXService,
)

logger = get_logger(__name__)


class CurateXAgent(BaseAgent):
    """
    Drug curation agent.

    §1: runs when a target and/or disease is available, either carried in from
    target identification (`session_state["targets"]`) or entered fresh. Given a
    disease with no confirmed target it surfaces the upstream shortlist for
    confirmation instead of selecting the top-scored target itself.
    """

    def __init__(self):
        super().__init__(tools=[])
        self._service = CurateXService()

    @property
    def name(self) -> str:
        return "curatex"

    def execute(self, input_data: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        target = (input_data.get("target") or "").strip()
        disease = (input_data.get("disease") or session_state.get("disease") or "").strip()
        shortlist = self._shortlist(session_state)

        # A target confirmed upstream (or named here) is the only thing that
        # starts a profile build; an unconfirmed shortlist is not a selection.
        if not target:
            target = (session_state.get("selected_target") or "").strip()

        if not target and not disease and not shortlist:
            return {
                "error": "No target or disease provided",
                "recommendation": "Provide a target (gene symbol or protein name) or a disease.",
            }

        try:
            result = self._service.run(
                target=target or None,
                disease=disease or None,
                candidate_targets=shortlist,
                top_n=int(input_data.get("num_results", 25)),
                session_id=input_data.get("session_id"),
            )
        except CurateXError as exc:
            logger.warning("CurateX chain stopped: %s", exc)
            return {"error": str(exc), "recommendation": str(exc)}
        except Exception as exc:
            logger.error("CurateX chain failed: %s", exc, exc_info=True)
            return {"error": str(exc), "recommendation": "CurateX could not complete this run."}

        if result.get("stage") == AWAITING_TARGET:
            return {
                "stage": AWAITING_TARGET,
                "disease": result.get("disease"),
                "candidate_targets": result.get("candidateTargets", []),
                "recommendation": result.get("message", ""),
                "session_state_update": {"disease": disease},
            }

        table = result.get("candidateTable", [])
        return {
            "selected_drugs": table,
            "drug_profile": result.get("profile", {}),
            "excluded_drugs": result.get("excludedCandidates", []),
            "score_only": result.get("scoreOnlyCandidates", []),
            "recommendation": result.get("recommendation", ""),
            "warnings": result.get("warnings", []),
            "metadata": result.get("metadata", {}),
            "session_state_update": {
                "curatex_target": (result.get("target") or {}).get("geneSymbol"),
                "curatex_candidates": [row.get("name") for row in table[:10]],
            },
        }

    @staticmethod
    def _shortlist(session_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize whatever target identification left in the session."""
        raw: Optional[List[Dict[str, Any]]] = session_state.get("targets")
        if not raw:
            return []
        return [
            {
                "name": t.get("name") or t.get("geneName") or t.get("uniprotId", ""),
                "uniprotId": t.get("uniprotId") or t.get("uniprot_id", ""),
                "score": t.get("score"),
                "category": t.get("category"),
            }
            for t in raw
            if isinstance(t, dict)
        ]
