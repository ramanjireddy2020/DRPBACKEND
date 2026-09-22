"""In-memory session store with optional persistence."""
import json
import os
from typing import Dict, Optional

from .models import SessionState, SessionTypes
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

SESSION_STORAGE_DIR = "data/sessions"


class SessionMemory:
    """
    Manages session state in memory with optional file persistence.
    """

    def __init__(self, storage_dir: str = SESSION_STORAGE_DIR):
        self._sessions: Dict[str, SessionState] = {}
        self._storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def create_session(self) -> SessionState:
        """Create a new session and return it."""
        session = SessionState()
        self._sessions[session.session_id] = session
        logger.info("Created session: %s", session.session_id)
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, updates: Dict) -> Optional[SessionState]:
        """Update session with new data and return updated session."""
        session = self._sessions.get(session_id)
        if session:
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.updated_at = session.updated_at  # triggers update
            return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def save_session(self, session_id: str) -> bool:
        """Persist session to JSON file."""
        session = self._sessions.get(session_id)
        if session:
            filepath = os.path.join(self._storage_dir, f"{session_id}.json")
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(session.to_dict(), f, indent=2)
                logger.info("Saved session: %s", session_id)
                return True
            except Exception as e:
                logger.error("Failed to save session %s: %s", session_id, e)
        return False

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session from JSON file."""
        filepath = os.path.join(self._storage_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = SessionState.from_dict(data)
                self._sessions[session_id] = session
                return session
            except Exception as e:
                logger.error("Failed to load session %s: %s", session_id, e)
        return None


# Global session store singleton
session_store = SessionMemory()