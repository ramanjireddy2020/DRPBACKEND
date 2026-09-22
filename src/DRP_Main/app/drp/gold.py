"""
Reading module output back out of Gold.

A Databricks-backed runner does not receive its results directly: the notebook
writes them to a Gold Delta table and exits. This module is the read side — it
pulls the rows one DRP job produced and hands them to the runner, which shapes
them into the same response the in-process path returns.

Rows are selected on `drp_job_id`, the value `execution.run_module()` passes to
every run. Selecting on that rather than on the module name is what makes
concurrent runs safe: two researchers querying TxKG at once write to the same
table and each reads back only their own rows.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.shared import uc_store

logger = get_logger(__name__)

# Gold tables each module job writes, in the order a runner consumes them.
# These names must match the `write_gold(...)` calls in `jobs/*_job.py` exactly —
# a typo here reads an empty/absent table, which looks identical to a job that
# ran fine and found nothing.
GOLD_TABLES = {
    "TxKG": ("kg_targets", "kg_relationships"),
    "LitMineX": ("lit_evidence",),
    "CurateX": ("drug_candidates",),
    "ScreenSuite": ("docking_results",),
    "NovSearch": ("novelty_report",),
}


def _quote(value: str) -> str:
    """Escape a value for inlining in a SQL string literal."""
    return str(value).replace("'", "''")


def read_job_rows(table: str, drp_job_id: str, *, schema: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch the Gold rows a single DRP job wrote to `table`.

    Raises `GoldUnavailable` when the query could not run at all, so a runner can
    report "could not read results" rather than silently returning an empty set
    that looks like a job which legitimately found nothing.
    """
    schema = schema or settings.DRP_GOLD_SCHEMA
    full_table = f"{settings.DRP_UC_CATALOG}.{schema}.{table}"
    statement = (
        f"SELECT * FROM {full_table} WHERE drp_job_id = '{_quote(drp_job_id)}'"
    )

    rows = uc_store.fetch_rows(statement)
    if rows is None:
        raise GoldUnavailable(
            f"Could not read {full_table} — check DRP_UC_WAREHOUSE_ID and that the job wrote its output"
        )
    logger.info("gold: read %d row(s) from %s for job %s", len(rows), full_table, drp_job_id)
    return rows


class GoldUnavailable(RuntimeError):
    """Gold could not be queried — distinct from a job that produced no rows."""


def decode(row: Dict[str, Any], column: str, default: Any = None) -> Any:
    """
    Read a column that a notebook stored as a JSON string.

    The Statement Execution API returns every value as text, so a struct or array
    column arrives as its JSON encoding. A value that is not valid JSON is handed
    back unchanged — some columns are plain strings and must survive this.
    """
    raw = row.get(column)
    if raw in (None, ""):
        return default
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def as_float(row: Dict[str, Any], column: str, default: Optional[float] = None) -> Optional[float]:
    """Coerce a text-typed numeric column, tolerating NULL and non-numeric text."""
    raw = row.get(column)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def as_int(row: Dict[str, Any], column: str, default: Optional[int] = None) -> Optional[int]:
    """Coerce a text-typed integer column, tolerating NULL and non-numeric text."""
    value = as_float(row, column)
    return int(value) if value is not None else default
