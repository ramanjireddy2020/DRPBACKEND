"""Evidence and interpretation tool for TxKG."""
import asyncio
from typing import Any, Dict, List

from DRP_Main.app.agents.base.base_tool import BaseTool, ToolResult
from DRP_Main.app.core.async_utils import run_sync
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class EvidenceInterpretationTool(BaseTool):
    """
    Retrieves PubMed evidence and generates an LLM interpretation, backed by
    the BioKG service in app.modules.txkg (wraps txkg_test.py).
    """

    def run(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Get evidence for targets.

        Args:
            input_data: Contains 'targets' and 'disease'

        Returns:
            ToolResult with evidence list (includes an 'interpretation' summary
            for the target set as a whole, on the first item's metadata)
        """
        targets = input_data.get("targets", [])
        disease = input_data.get("disease", "")

        if not targets:
            return ToolResult(success=True, data=[])

        try:
            evidence, interpretation = run_sync(self._get_evidence(targets, disease))
            return ToolResult(
                success=True,
                data=evidence,
                metadata={"evidence_count": len(evidence), "interpretation": interpretation},
            )
        except Exception as e:
            logger.error("Evidence retrieval failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _get_evidence(self, targets: List[Dict], disease: str):
        from DRP_Main.app.modules.txkg.service import (
            fetch_pubmed_articles_for_target,
            get_llm_interpretation,
        )

        top_targets = targets[:5]

        articles_per_target = await asyncio.gather(
            *(
                fetch_pubmed_articles_for_target(disease, t.get("name", ""), limit=2)
                for t in top_targets
            ),
            return_exceptions=True,
        )

        evidence = []
        for target, articles in zip(top_targets, articles_per_target):
            if isinstance(articles, Exception):
                logger.warning("PubMed fetch failed for %s: %s", target.get("name"), articles)
                articles = []
            evidence.append(
                {
                    "target_id": target.get("id", ""),
                    "target_name": target.get("name", ""),
                    "pubmed_articles": articles,
                }
            )

        interpretation = await get_llm_interpretation(disease, top_targets)
        for item in evidence:
            item["interpretation"] = interpretation

        return evidence, interpretation

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data.get("targets"), list)
