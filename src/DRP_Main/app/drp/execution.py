"""
Databricks Jobs as the execution backend for DRP agent jobs.

The `/v1` API models a module run as a `DrpJob` row that a registered runner
fulfils (see `drp/jobs.py`). A runner is free to do the work in-process or to
delegate it; this module is the delegating half — it triggers the per-module
Databricks job, follows the run, and hands back the identifiers needed to read
the module's output out of Gold.

Nothing above the runner layer knows which backend ran the work: sessions,
job status, polling and every shaped response stay identical either way, so the
frontend contract is unaffected by the switch.

Job ids come from the same `JOB_ID_*` environment variables the orchestration
router already uses, so both paths stay pointed at one set of jobs.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.drp.jobs import JobContext

logger = get_logger(__name__)

# Canonical module key -> the env var holding its Databricks job id. Mirrors
# `modules/orchestration/router.py`; NovSearch is spelled "novelty" there.
JOB_ID_ENV_BY_MODULE = {
    "TxKG": "JOB_ID_TXKG",
    "LitMineX": "JOB_ID_LITMINEX",
    "CurateX": "JOB_ID_CURATEX",
    "ScreenSuite": "JOB_ID_SCREENSUITE",
    "NovSearch": "JOB_ID_NOVELTY",
}

# Databricks life-cycle states that mean "still going".
_ACTIVE_STATES = {"PENDING", "QUEUED", "RUNNING", "TERMINATING"}


class JobExecutionError(RuntimeError):
    """A Databricks run could not be started, or finished unsuccessfully."""


def is_databricks_backend() -> bool:
    """Whether runners should delegate to Databricks Jobs rather than run in-process."""
    return settings.DRP_EXECUTION_BACKEND.strip().lower() == "databricks"


def job_id_for(module: str) -> str:
    """Resolve a module's configured Databricks job id, or raise a readable error."""
    env_var = JOB_ID_ENV_BY_MODULE.get(module)
    if env_var is None:
        raise JobExecutionError(f"No Databricks job is mapped to module '{module}'")
    job_id = os.getenv(env_var)
    if not job_id:
        raise JobExecutionError(
            f"{env_var} is not set — module '{module}' has no Databricks job linked"
        )
    return job_id


def has_job_linked(module: str) -> bool:
    """Whether `module` has a Databricks job id configured to delegate to."""
    env_var = JOB_ID_ENV_BY_MODULE.get(module)
    return bool(env_var and os.getenv(env_var))


def delegated_modules() -> set:
    """Module keys opted into their Databricks job via `DRP_DATABRICKS_MODULES`."""
    raw = getattr(settings, "DRP_DATABRICKS_MODULES", "") or ""
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def should_delegate(module: str) -> bool:
    """
    Whether this module's work should go to Databricks on this deployment.

    True when the global backend is Databricks, or when this module is listed in
    `DRP_DATABRICKS_MODULES` — the per-module opt-in, so a module whose job
    notebook is real can move without dragging along modules whose notebooks are
    still placeholders.

    False otherwise, and also when delegation is requested but this module has no
    job linked yet — during the migration the modules are ported one at a time,
    and an unported one must keep working rather than start failing the moment
    the flag is flipped.
    """
    if not (is_databricks_backend() or module.strip().lower() in delegated_modules()):
        return False
    if not has_job_linked(module):
        logger.warning(
            "execution: backend is 'databricks' but %s has no %s configured — "
            "running %s in-process",
            module, JOB_ID_ENV_BY_MODULE.get(module, "job id"), module,
        )
        return False
    return True


def _client():
    """Lazily build a WorkspaceClient. On Databricks Apps this auto-authenticates
    as the app's service principal, which is the identity granted CAN_MANAGE_RUN
    on each module job."""
    try:
        from databricks.sdk import WorkspaceClient

        if settings.DATABRICKS_CONFIG_PROFILE:
            return WorkspaceClient(profile=settings.DATABRICKS_CONFIG_PROFILE)
        if settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN:
            return WorkspaceClient(
                host=settings.DATABRICKS_HOST, token=settings.DATABRICKS_TOKEN
            )
        return WorkspaceClient()
    except Exception as exc:  # noqa: BLE001
        raise JobExecutionError(f"Databricks SDK unavailable: {exc}")


def _job_parameters(drp_job_id: str, params: Dict[str, Any]) -> Dict[str, str]:
    """
    Flatten runner params onto the widgets the module notebooks declare.

    `drp_job_id` is passed on every run and written into each Gold row by the
    notebook, so a completed run's output can be read back without depending on
    the Databricks run id — which the API would otherwise have to store per job.

    `catalog`/`schema` are sent from the API's own settings rather than left to the
    notebook's widget defaults — the two default differently ("drug_repository" here,
    "workspace" there), and a job writing to a table the API does not read would
    otherwise look like a run that simply produced nothing.

    Job parameters are strings; non-scalars are JSON-encoded for the notebook to
    parse, and `None` is dropped so a widget keeps its declared default.
    """
    import json

    flat: Dict[str, str] = {
        "drp_job_id": drp_job_id,
        "catalog": settings.DRP_UC_CATALOG,
        "schema": settings.DRP_GOLD_SCHEMA,
    }
    for key, value in (params or {}).items():
        if value is None:
            continue
        flat[key] = value if isinstance(value, str) else json.dumps(value)
    return flat


async def run_module(
    module: str,
    params: Dict[str, Any],
    ctx: JobContext,
    *,
    poll_seconds: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> int:
    """
    Trigger `module`'s Databricks job for this DRP job and wait for it to finish.

    Returns the Databricks run id on success; raises `JobExecutionError` otherwise.
    Progress is reported through `ctx.progress()`, so the run's life-cycle shows up
    on `GET /v1/agents/jobs/{jobId}/status` exactly like in-process progress does.

    Every SDK call is a blocking HTTP request, so each is pushed to a worker
    thread — these runners are coroutines on the server's own event loop, and
    blocking it would stall every other request in the process.
    """
    poll_seconds = poll_seconds or settings.DRP_JOB_POLL_SECONDS
    timeout_seconds = timeout_seconds or settings.DRP_JOB_TIMEOUT_SECONDS

    job_id = job_id_for(module)
    client = _client()
    parameters = _job_parameters(ctx.job_id, params)

    ctx.progress(f"Starting the {module} job on Databricks...")
    try:
        run = await asyncio.to_thread(
            client.jobs.run_now, job_id=int(job_id), job_parameters=parameters
        )
    except Exception as exc:  # noqa: BLE001
        raise JobExecutionError(f"Could not start the {module} job: {exc}")

    run_id = run.run_id
    logger.info("drp job %s -> databricks run %s (%s)", ctx.job_id, run_id, module)

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_reported = ""

    while True:
        if asyncio.get_running_loop().time() >= deadline:
            # Leave the Databricks run alone: cancelling it here would discard
            # partial Gold output that a later read could still use, and the run
            # page stays available for diagnosis.
            raise JobExecutionError(
                f"{module} run {run_id} did not finish within {timeout_seconds}s"
            )

        await asyncio.sleep(poll_seconds)

        try:
            info = await asyncio.to_thread(client.jobs.get_run, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 — a transient poll failure is not
            # a failed run; keep polling until the deadline rather than giving up.
            logger.warning("poll of run %s failed (%s), retrying", run_id, exc)
            continue

        state = info.state
        life_cycle = getattr(getattr(state, "life_cycle_state", None), "value", "") or ""

        if life_cycle in _ACTIVE_STATES:
            message = _progress_message(module, life_cycle, info)
            if message != last_reported:
                ctx.progress(message)
                last_reported = message
            continue

        result_state = getattr(getattr(state, "result_state", None), "value", "") or ""
        if result_state == "SUCCESS":
            ctx.progress(f"{module} job finished — collecting results...")
            return run_id

        detail = getattr(state, "state_message", "") or result_state or life_cycle
        raise JobExecutionError(
            f"{module} run {run_id} ended as {result_state or life_cycle}: {detail}"
        )


def _progress_message(module: str, life_cycle: str, info: Any) -> str:
    """Turn a Databricks life-cycle state into a line worth showing a researcher."""
    if life_cycle in ("PENDING", "QUEUED"):
        return f"Waiting for compute to start the {module} job..."
    if life_cycle == "TERMINATING":
        return f"{module} job finishing..."
    # RUNNING: the notebook's own task state is the most informative thing available.
    tasks = getattr(info, "tasks", None) or []
    for task in tasks:
        task_state = getattr(getattr(task, "state", None), "life_cycle_state", None)
        if getattr(task_state, "value", "") == "RUNNING":
            return f"Running {module}: {getattr(task, 'task_key', 'analysis')}..."
    return f"Running the {module} analysis on Databricks..."
