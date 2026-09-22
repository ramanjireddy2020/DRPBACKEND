"""Conversation flow management."""
from typing import Any, Dict, List, Optional

from DRP_Main.session.models import SessionState


class ConversationManager:
    """
    Manages conversation history and step validation.
    """

    @staticmethod
    def add_message(session_id: str, role: str, content: str, agent: Optional[str] = None) -> None:
        """Add a message to the conversation history."""
        from DRP_Main.session.memory import session_store
        from datetime import datetime
        
        session = session_store.get_session(session_id)
        if session:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if agent:
                message["agent"] = agent
            session.messages.append(message)

    @staticmethod
    def get_history(session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        from DRP_Main.session.memory import session_store
        
        session = session_store.get_session(session_id)
        return session.messages if session else []

    @staticmethod
    def validate_step(current_step: str, next_step: str) -> bool:
        """Validate that next_step follows pipeline order."""
        pipeline_order = ["txkg", "litminex", "curatex", "screensuite", "novsearch"]
        
        if next_step == "genie":
            return True  # Genie can be called anytime
            
        if current_step in pipeline_order and next_step in pipeline_order:
            current_idx = pipeline_order.index(current_step)
            next_idx = pipeline_order.index(next_step)
            # Allow same step or next step in pipeline
            return next_idx >= current_idx
            
        return True  # Allow branching out of normal order