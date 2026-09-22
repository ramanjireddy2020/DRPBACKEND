"""Generate subgraph JSON for TxKG visualizations."""
from typing import Any, Dict, List, Optional


class SubgraphRenderer:
    """
    Renders knowledge graph subgraphs as JSON for frontend visualization.
    """

    def __init__(self):
        pass

    def render(self, targets: List[Dict], disease: str) -> Dict[str, Any]:
        """
        Generate subgraph JSON structure.
        
        Args:
            targets: List of target protein dictionaries
            disease: Disease name for context
            
        Returns:
            JSON structure for Cytoscape.js visualization
        """
        nodes = [{"data": {"id": "disease", "label": disease, "type": "disease"}}]
        edges = []

        for target in targets:
            node_id = target.get("id", target.get("name", "unknown"))
            node_label = target.get("name", node_id)
            nodes.append({
                "data": {
                    "id": node_id,
                    "label": node_label,
                    "type": "target",
                    "score": target.get("score", 0),
                }
            })
            edges.append({
                "data": {
                    "id": f"disease-{node_id}",
                    "source": "disease",
                    "target": node_id,
                    "relationship": "ASSOCIATED_WITH",
                }
            })

        return {
            "elements": {
                "nodes": nodes,
                "edges": edges,
            },
            "metadata": {
                "disease": disease,
                "target_count": len(targets),
                "generated_by": "txkg_agent",
            }
        }

    def render_metapaths(self, targets: List[Dict]) -> Dict[str, Any]:
        """
        Generate metapath structure for targets.
        
        Returns:
            Metapath JSON with relationship paths
        """
        metapaths = []
        for target in targets:
            paths = target.get("paths", [])
            metapaths.append({
                "target": target.get("id", target.get("name")),
                "metapaths": paths[:3],  # Top 3 paths
                "hop_distance": target.get("hop_distance", 0),
            })
        return {"metapaths": metapaths}