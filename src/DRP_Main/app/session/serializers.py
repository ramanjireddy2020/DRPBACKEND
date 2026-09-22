"""Session serialization utilities."""
import json
from typing import Any, Dict

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


def serialize_for_export(state_dict: Dict[str, Any]) -> str:
    """Serialize session state for report export."""
    try:
        return json.dumps(state_dict, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        logger.error("Serialization failed: %s", e)
        return "{}"


def deserialize_from_export(json_str: str) -> Dict[str, Any]:
    """Deserialize session state from report."""
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error("Deserialization failed: %s", e)
        return {}