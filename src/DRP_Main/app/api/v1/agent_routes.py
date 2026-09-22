"""REST and WebSocket endpoints for agentic architecture."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from DRP_Main.app.supervisor.supervisor import supervisor
from DRP_Main.app.supervisor.report_compiler import generate_report
from DRP_Main.app.session.memory import session_store
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SupervisorRequest(BaseModel):
    """Request model for supervisor endpoint."""
    message: str
    session_id: Optional[str] = None


class SupervisorResponse(BaseModel):
    """Response model for supervisor endpoint."""
    session_id: str
    agent_used: str
    intent: str
    output: Dict[str, Any]
    next_step: str


@router.post("/message", response_model=SupervisorResponse)
async def supervisor_message(request: SupervisorRequest):
    """
    Send a message to the supervisor agent.
    Returns agent output and next step recommendation.
    """
    try:
        result = supervisor.execute({
            "message": request.message,
            "session_id": request.session_id,
        })
        return result
    except Exception as e:
        logger.error("Supervisor message failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.get("/report/{session_id}")
async def get_report(session_id: str):
    """Generate consolidated report for session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return generate_report(session)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_store.delete_session(session_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")