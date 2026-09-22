"""Databricks client wrapper for WorkspaceClient and Genie API."""
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class DatabricksClient:
    """Wrapper for Databricks SDK operations including Genie."""

    def __init__(self):
        try:
            self._client = WorkspaceClient()
            logger.info("Databricks WorkspaceClient initialized")
        except Exception as e:
            logger.warning("Databricks SDK not available: %s", e)
            self._client = None

    def run_job(self, job_id: int, parameters: Optional[Dict[str, str]] = None) -> Optional[int]:
        """Trigger a Databricks job and return run_id."""
        if not self._client:
            raise RuntimeError("Databricks client not configured")

        try:
            run = self._client.jobs.run_now(job_id=job_id, job_parameters=parameters)
            return run.run_id
        except Exception as e:
            logger.error("Job run failed: %s", e)
            return None

    def get_run_status(self, run_id: int) -> Dict[str, Any]:
        """Get status of a job run."""
        if not self._client:
            raise RuntimeError("Databricks client not configured")

        try:
            run = self._client.jobs.get_run(run_id=run_id)
            state = run.state
            return {
                "run_id": run_id,
                "life_cycle_state": getattr(state, "life_cycle_state", None) and state.life_cycle_state.value,
                "result_state": getattr(state, "result_state", None) and state.result_state.value,
                "run_page_url": run.run_page_url,
            }
        except NotFound:
            return {"error": "Run not found"}
        except Exception as e:
            logger.error("Get run status failed: %s", e)
            return {"error": str(e)}

    def query_sql(self, statement: str, warehouse_id: Optional[str] = None) -> List[Dict]:
        """Execute SQL statement and return results."""
        if not self._client:
            raise RuntimeError("Databricks client not configured")

        try:
            # This is a placeholder - actual implementation needs warehouse ID
            logger.info("SQL query would execute: %s", statement[:50])
            return []
        except Exception as e:
            logger.error("SQL query failed: %s", e)
            return []

    def genie_start_conversation(self, space_id: str) -> Optional[int]:
        """Start a new Genie conversation."""
        # Placeholder - Genie API not yet available in SDK
        logger.warning("Genie conversation start not implemented")
        return None


# Global singleton
databricks_client = DatabricksClient()