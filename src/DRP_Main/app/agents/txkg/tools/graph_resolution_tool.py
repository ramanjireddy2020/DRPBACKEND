"""Graph resolution and traversal tool for TxKG."""
from typing import Any, Dict, List

from DRP_Main.app.agents.base.base_tool import BaseTool, ToolResult
from DRP_Main.app.core.async_utils import run_sync
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class GraphResolutionTool(BaseTool):
    """
    Resolves disease in knowledge graph and performs multi-hop traversal,
    backed by the BioKG service in app.modules.txkg (wraps txkg_test.py).
    """

    def run(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Run graph resolution for a disease.

        Args:
            input_data: Contains 'disease' name

        Returns:
            ToolResult with target list
        """
        disease = input_data.get("disease", "")

        if not disease:
            return ToolResult(success=False, error="No disease provided")

        try:
            targets, resolved_name = self._resolve_disease(disease)
            return ToolResult(
                success=True,
                data={"targets": targets, "disease": resolved_name or disease},
                metadata={"target_count": len(targets)},
            )
        except Exception as e:
            logger.error("Graph resolution failed: %s", e)
            return ToolResult(success=False, error=str(e))

    def _resolve_disease(self, disease: str) -> tuple[List[Dict], str]:
        """Fuzzy-match the disease against the KG and extract its top targets."""
        from DRP_Main.app.modules.txkg.service import find_disease, extract_targets

        disease_id, disease_name = find_disease(disease)
        if not disease_id:
            return [], disease

        targets = run_sync(extract_targets(disease_id, max_targets=25, max_hops=3))
        return targets, disease_name

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "disease" in input_data and bool(input_data["disease"])
