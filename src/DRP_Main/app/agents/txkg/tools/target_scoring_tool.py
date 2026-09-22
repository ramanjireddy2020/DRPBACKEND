"""Target scoring tool for TxKG."""
from typing import Any, Dict, List, Optional

from DRP_Main.app.agents.base.base_tool import BaseTool, ToolResult
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class TargetScoringTool(BaseTool):
    """
    Scores targets based on hop distance, connection strength, and shared context.
    """

    def run(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Score a list of targets.
        
        Args:
            input_data: Contains 'targets' list
            
        Returns:
            ToolResult with scored targets
        """
        targets = input_data.get("targets", [])
        
        if not targets:
            return ToolResult(success=True, data=[])

        try:
            scored = self._score_targets(targets)
            return ToolResult(
                success=True,
                data=scored,
                metadata={"scored_count": len(scored)}
            )
        except Exception as e:
            logger.error("Target scoring failed: %s", e)
            return ToolResult(success=False, error=str(e))

    def _score_targets(self, targets: List[Dict]) -> List[Dict]:
        """
        Apply scoring logic to targets.
        """
        for target in targets:
            # The KG's extract_targets() already produces a composite 0-100 score
            # (hop distance + direct edges + shared context) — normalize it to
            # 0-1 rather than recomputing from fields it doesn't set.
            if "score" in target and isinstance(target["score"], (int, float)):
                target["score"] = round(min(target["score"], 100) / 100.0, 3)
                continue

            if "hop_distance" in target:
                target["hop_score"] = max(0, 1.0 - (target.get("hop_distance", 3) / 3.0))

            if "connection_strength" in target:
                target["connection_score"] = target.get("connection_strength", 0) / 100.0

            hop_score = target.get("hop_score", 0.5)
            conn_score = target.get("connection_score", 0.5)
            target["score"] = round((hop_score + conn_score) / 2.0, 3)

        return sorted(targets, key=lambda x: x.get("score", 0), reverse=True)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data.get("targets"), list)