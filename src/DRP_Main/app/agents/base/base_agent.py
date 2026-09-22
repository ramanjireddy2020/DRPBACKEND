"""Abstract base agent that all specialist agents inherit from."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from DRP_Main.app.agents.base.base_tool import BaseTool, ToolResult
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialist agents.
    Provides common tool execution framework and state management.
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self._tools = tools or []

    @property
    def tools(self) -> List[BaseTool]:
        return self._tools

    @abstractmethod
    def execute(self, input_data: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's workflow given input and session state.
        
        Args:
            input_data: User input / request parameters
            session_state: Shared session state from supervisor
            
        Returns:
            Agent output dict with results, recommendation, and optional artifacts
        """
        pass

    def run_tool(self, tool_name: str, input_data: Dict[str, Any]) -> ToolResult:
        """Find and execute a specific tool by name."""
        for tool in self._tools:
            if tool.__class__.__name__.lower() == tool_name.lower():
                if tool.validate_input(input_data):
                    return tool.run(input_data)
        return ToolResult(success=False, error=f"Tool {tool_name} not found")

    def get_recommendation(self, results: Dict[str, Any]) -> str:
        """Generate recommendation based on agent results."""
        return "No recommendation available"

    @property
    @abstractmethod
    def name(self) -> str:
        """Return agent identifier."""
        pass