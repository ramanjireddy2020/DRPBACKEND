"""Intent classification and agent routing."""
from typing import Optional

from DRP_Main.app.session.models import SessionTypes
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class IntentRouter:
    """Classifies user intent and determines routing."""

    # Keywords for each agent
    TXKG_KEYWORDS = ["disease", "target", "protein", "gene", "pathway"]
    LITMEDX_KEYWORDS = ["literature", "paper", "article", "pubmed", "publication", "research"]
    CURATEX_KEYWORDS = ["drug", "compound", "molecule", "ligand", "cure", "treat"]
    SCREENSUITE_KEYWORDS = ["dock", "bind", "affinity", "screen", "vina", "pdb"]
    NOVSEARCH_KEYWORDS = ["patent", "novelty", "freedom", "operate", "claim"]
    GENIE_KEYWORDS = ["sql", "lakehouse", "table", "database", "warehouse", "schema", "analyze data"]

    @classmethod
    def classify(cls, message: str, session_state: Optional[dict] = None) -> SessionTypes:
        """
        Classify the intent of a user message.
        
        Args:
            message: User input message
            session_state: Current session state for context
            
        Returns:
            SessionTypes enum value indicating intent
        """
        message_lower = message.lower()
        
        # Check for general queries (Genie fallback)
        if any(kw in message_lower for kw in cls.GENIE_KEYWORDS):
            return SessionTypes.GENERAL_QUERY
            
        # Check for follow-up patterns
        follow_up_patterns = ["what about", "tell me more", "continue", "next", "go on", "then"]
        if any(p in message_lower for p in follow_up_patterns) and session_state:
            return SessionTypes.FOLLOW_UP

        # Check for specific agent keywords
        if any(kw in message_lower for kw in cls.TXKG_KEYWORDS):
            return SessionTypes.STEP_SPECIFIC  # Will route to TxKG based on context
        elif any(kw in message_lower for kw in cls.LITMEDX_KEYWORDS):
            return SessionTypes.STEP_SPECIFIC
        elif any(kw in message_lower for kw in cls.CURATEX_KEYWORDS):
            return SessionTypes.STEP_SPECIFIC
        elif any(kw in message_lower for kw in cls.SCREENSUITE_KEYWORDS):
            return SessionTypes.STEP_SPECIFIC
        elif any(kw in message_lower for kw in cls.NOVSEARCH_KEYWORDS):
            return SessionTypes.STEP_SPECIFIC
            
        # Full pipeline indicators
        pipeline_patterns = ["run full", "complete analysis", "full workflow", "do everything"]
        if any(p in message_lower for p in pipeline_patterns):
            return SessionTypes.FULL_PIPELINE

        # Default to fresh query
        return SessionTypes.FRESH_QUERY

    @classmethod
    def route_to_agent(cls, intent: SessionTypes, session_state: Optional[dict] = None) -> str:
        """
        Determine which agent to route to based on intent.
        
        Args:
            intent: Classified intent
            session_state: Current session state
            
        Returns:
            Agent name string
        """
        if intent == SessionTypes.GENERAL_QUERY:
            return "genie"
            
        if intent == SessionTypes.FOLLOW_UP:
            # Return last agent or next in pipeline
            if session_state and session_state.get("last_agent"):
                return session_state["last_agent"]
            return "txkg"  # Default start
            
        if intent == SessionTypes.FULL_PIPELINE:
            return "txkg"  # Start of full pipeline
            
        # For step-specific, use keyword matching
        # This is simplified - actual implementation would need more context
        return "txkg"  # Default


def classify_intent(message: str, session_state: Optional[dict] = None) -> SessionTypes:
    """Convenience function for intent classification."""
    return IntentRouter.classify(message, session_state)