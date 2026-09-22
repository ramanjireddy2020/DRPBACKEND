"""Generate binding pose JSON for ScreenSuite visualizations."""
from typing import Any, Dict, List, Optional


class BindingPoseRenderer:
    """
    Renders binding pose data as JSON for frontend visualization.
    """

    def __init__(self):
        pass

    def render(self, docking_results: List[Dict], protein_name: str) -> Dict[str, Any]:
        """
        Generate binding pose visualization JSON.
        
        Args:
            docking_results: List of docking result dictionaries
            protein_name: Protein identifier
            
        Returns:
            JSON structure for 3D visualization
        """
        poses = []
        for result in docking_results[:10]:  # Top 10 poses
            poses.append({
                "drug": result.get("ligand"),
                "affinity": result.get("affinity", 0),
                "pose_file": result.get("out_pdbqt_file"),
                "interaction_score": self._calculate_interaction_score(result),
            })

        return {
            "poses": poses,
            "protein": protein_name,
            "total_poses": len(docking_results),
        }

    def _calculate_interaction_score(self, result: Dict) -> float:
        """Calculate interaction score from affinity."""
        affinity = result.get("affinity", 0)
        # Lower (more negative) is better
        return max(0, min(100, -affinity))


class InteractionRenderer:
    """
    Renders interaction table data for ScreenSuite.
    """

    def render(self, affinity_records: List[Dict]) -> Dict[str, Any]:
        """
        Generate interaction table JSON.
        
        Args:
            affinity_records: List of affinity records
            
        Returns:
            Interaction table JSON
        """
        interactions = []
        for record in affinity_records[:20]:  # Top 20 interactions
            interactions.append({
                "mode": record.get("Mode", 0),
                "protein_ligand": record.get("protien_ligand", ""),
                "protein": record.get("protein", ""),
                "ligand": record.get("ligand", ""),
                "affinity_kcal": record.get("Affinity_kcal_per_mol", 0),
            })

        return {
            "interactions": interactions,
            "total_interactions": len(affinity_records),
        }