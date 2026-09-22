"""
Base tool interface for specialist agents.
Each tool encapsulates one or more related operations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Standardized result from any tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for agent tools.
    All tools must implement the run method.
    """

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool's operation.
        
        Args:
            input_data: Dictionary containing all required inputs
            
        Returns:
            ToolResult with success status, data, and optional metadata
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Override to add input validation logic."""
        return True