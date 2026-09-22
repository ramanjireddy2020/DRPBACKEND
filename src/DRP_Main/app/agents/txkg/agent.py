"""TxKG Agent - Target identification from disease name."""
from typing import Any, Dict, List, Optional

from DRP_Main.app.agents.base.base_agent import BaseAgent
from DRP_Main.app.agents.base.base_tool import ToolResult
from DRP_Main.app.agents.txkg.tools.graph_resolution_tool import GraphResolutionTool
from DRP_Main.app.agents.txkg.tools.target_scoring_tool import TargetScoringTool
from DRP_Main.app.agents.txkg.tools.evidence_interpretation_tool import EvidenceInterpretationTool
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class TxKGAgent(BaseAgent):
    """
    Target identification agent using knowledge graph traversal.
    """

    def __init__(self):
        super().__init__(tools=[
            GraphResolutionTool(),
            TargetScoringTool(),
            EvidenceInterpretationTool(),
        ])

    @property
    def name(self) -> str:
        return "txkg"

    def execute(self, input_data: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TxKG pipeline: resolve disease -> extract targets -> score -> evidence.
        
        Args:
            input_data: User input (disease name)
            session_state: Current session state
            
        Returns:
            Targets, evidence, recommendation, and subgraph artifact
        """
        disease = input_data.get("message", "") or session_state.get("disease", "")
        
        if not disease:
            return {"error": "No disease provided", "recommendation": "Please specify a disease to analyze."}

        # Step 1: Graph resolution and traversal
        graph_result = self.run_tool("GraphResolutionTool", {"disease": disease})
        
        if not graph_result.success:
            return {"error": graph_result.error, "recommendation": "Failed to resolve disease in knowledge graph."}

        targets = graph_result.data.get("targets", [])
        
        # Step 2: Score targets
        scoring_result = self.run_tool("TargetScoringTool", {"targets": targets})
        scored_targets = scoring_result.data if scoring_result.success else targets

        # Step 3: Get evidence
        evidence_result = self.run_tool("EvidenceInterpretationTool", {"targets": scored_targets, "disease": disease})
        evidence = evidence_result.data if evidence_result.success else []

        # Generate recommendation
        recommendation = self._generate_recommendation(scored_targets, disease)

        # Generate subgraph artifact
        from DRP_Main.app.artifacts.subgraph_renderer import SubgraphRenderer
        renderer = SubgraphRenderer()
        subgraph_artifact = renderer.render(scored_targets, disease)

        return {
            "disease": disease,
            "targets": scored_targets,
            "evidence": evidence,
            "recommendation": recommendation,
            "subgraph_artifact": subgraph_artifact,
        }

    def _generate_recommendation(self, targets: List[Dict], disease: str) -> str:
        """Generate recommendation based on top targets."""
        if not targets:
            return f"No targets found for {disease}. Consider refining the disease name."
        
        top_target = targets[0] if targets else {}
        return f"Recommended target: {top_target.get('name', 'unknown')} with score {top_target.get('score', 0):.2f}"