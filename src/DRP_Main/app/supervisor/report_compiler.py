"""Report compilation for consolidated export."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from DRP_Main.app.session.models import SessionState


def generate_report(session_state: SessionState) -> Dict[str, Any]:
    """
    Generate a consolidated report from session state.
    
    Args:
        session_state: Completed session state
        
    Returns:
        Report dictionary with all agent results
    """
    return {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "session_id": session_state.session_id,
            "created_at": session_state.created_at,
        },
        "query_summary": {
            "disease": session_state.disease,
            "literature_query": session_state.literature_query,
        },
        "txkg_results": {
            "targets": session_state.targets,
            "target_count": len(session_state.targets),
        },
        "litminex_results": {
            "articles": session_state.articles,
            "article_count": len(session_state.articles),
        },
        "curatex_results": {
            "selected_drugs": session_state.selected_drugs,
            "drug_count": len(session_state.selected_drugs),
        },
        "screensuite_results": {
            "docking_results": session_state.docking_results,
            "top_hits": session_state.top_hits,
        },
        "novsearch_results": {
            "patent_findings": session_state.patent_findings,
            "novelty_verdict": session_state.novelty_verdict,
        },
        "recommendation_trail": _build_recommendation_trail(session_state.messages),
    }


def _build_recommendation_trail(messages: List[Dict]) -> List[Dict]:
    """Extract recommendations from message history."""
    recommendations = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("recommendation"):
            recommendations.append({
                "agent": msg.get("agent", "unknown"),
                "recommendation": msg.get("recommendation"),
                "timestamp": msg.get("timestamp", ""),
            })
    return recommendations