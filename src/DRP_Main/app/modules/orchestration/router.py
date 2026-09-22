"""
Orchestration module — the app's link to the Databricks Jobs API.

Mirrors the architecture arrow `POST /api/2.1/jobs/run-now`: the FastAPI app
(running as a Databricks App service principal) triggers the per-module jobs
(TxKG / LitMineX / CurateX / ScreenSuite / Novelty) and reports run status.

Job IDs are supplied via environment variables (set in app.yaml after the jobs
are created), so no IDs are hardcoded:
    JOB_ID_TXKG, JOB_ID_LITMINEX, JOB_ID_CURATEX, JOB_ID_SCREENSUITE, JOB_ID_NOVELTY
"""
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# module key -> (env var holding the job id, human label)
MODULES = {
    "txkg":        ("JOB_ID_TXKG",        "TxKG — Knowledge Graph"),
    "litminex":    ("JOB_ID_LITMINEX",    "LitMineX — Literature Mining"),
    "curatex":     ("JOB_ID_CURATEX",     "CurateX — Drug Curation"),
    "screensuite": ("JOB_ID_SCREENSUITE", "ScreenSuite — Docking"),
    "novelty":     ("JOB_ID_NOVELTY",     "Novelty — Patent Search"),
}


class RunRequest(BaseModel):
    input: Optional[str] = None        # disease / protein-drug / query, per module
    catalog: Optional[str] = None
    schema_: Optional[str] = None      # 'schema' is reserved on BaseModel


def _job_id(module: str) -> str:
    if module not in MODULES:
        raise HTTPException(404, f"Unknown module '{module}'. Valid: {list(MODULES)}")
    env_var, _ = MODULES[module]
    job_id = os.getenv(env_var)
    if not job_id:
        raise HTTPException(503, f"{env_var} not configured — job not linked yet")
    return job_id


def _client():
    """Lazy import so the app boots even if the SDK isn't installed.
    On Databricks Apps the SDK auto-authenticates as the app service principal."""
    try:
        from databricks.sdk import WorkspaceClient
        return WorkspaceClient()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Databricks SDK unavailable: {exc}")


@router.get("/", tags=["0. Orchestration"])
def list_modules():
    """List each module and whether its job is linked (job id configured)."""
    return {
        m: {"label": label, "job_id": os.getenv(env), "linked": bool(os.getenv(env))}
        for m, (env, label) in MODULES.items()
    }


@router.post("/{module}/run-now", tags=["0. Orchestration"])
def run_now(module: str, body: RunRequest | None = None):
    """Trigger a module's Databricks job (POST /api/2.1/jobs/run-now)."""
    job_id = _job_id(module)
    body = body or RunRequest()
    params = {}
    if body.input is not None:
        params["input"] = body.input
    if body.catalog:
        params["catalog"] = body.catalog
    if body.schema_:
        params["schema"] = body.schema_

    w = _client()
    run = w.jobs.run_now(job_id=int(job_id), job_parameters=params or None)
    return {"module": module, "job_id": job_id, "run_id": run.run_id,
            "params": params, "status": "TRIGGERED"}


@router.get("/{module}/runs/{run_id}", tags=["0. Orchestration"])
def run_status(module: str, run_id: int):
    """Get the status of a triggered run."""
    _job_id(module)  # validates module
    w = _client()
    r = w.jobs.get_run(run_id=run_id)
    state = r.state
    return {
        "module": module,
        "run_id": run_id,
        "life_cycle_state": getattr(state, "life_cycle_state", None) and state.life_cycle_state.value,
        "result_state": getattr(state, "result_state", None) and state.result_state.value,
        "run_page_url": r.run_page_url,
    }
