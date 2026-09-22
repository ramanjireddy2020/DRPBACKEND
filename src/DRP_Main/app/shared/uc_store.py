"""
Unity Catalog persistence for module output (bronze/silver/gold tables).

Each pipeline module writes its results here as a best-effort side effect after a
job completes — a write failure or missing warehouse must degrade to a logged
warning, never fail the job that already has a valid in-memory/API result.

Uses the Databricks SDK's Statement Execution API (`WorkspaceClient.statement_execution`)
rather than `databricks-sql-connector`, since the SDK is already a dependency
(`core/databricks_client.py`) and this avoids adding a second Databricks client library.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Optional

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

# Statement Execution's own `wait_timeout` caps at 50s; a cold serverless warehouse
# can take longer than that to spin up, in which case the call returns early with
# the statement still PENDING/RUNNING rather than raising — polling picks up from
# there instead of the caller silently treating "no exception" as "row landed".
_POLL_INTERVAL_SECONDS = 3
_POLL_TIMEOUT_SECONDS = 120

_client = None
_client_init_failed = False


def _get_client():
    """Lazily construct the WorkspaceClient. Never raises — returns None on failure."""
    global _client, _client_init_failed
    if _client is not None or _client_init_failed:
        return _client
    try:
        from databricks.sdk import WorkspaceClient

        if settings.DATABRICKS_CONFIG_PROFILE:
            _client = WorkspaceClient(profile=settings.DATABRICKS_CONFIG_PROFILE)
        elif settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN:
            _client = WorkspaceClient(host=settings.DATABRICKS_HOST, token=settings.DATABRICKS_TOKEN)
        else:
            _client = WorkspaceClient()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("uc_store: Databricks SDK unavailable, writes disabled: %s", exc)
        _client_init_failed = True
    return _client


def _sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal for a parameterless INSERT statement."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (dict, list)):
        value = json.dumps(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _await_statement(client, statement: str):
    """
    Execute `statement` and block until it reaches a terminal state.

    Returns the final response on SUCCEEDED, or None otherwise — never on a bare
    "no exception raised", since `execute_statement` can return with the statement
    still PENDING/RUNNING if the warehouse was cold-starting.
    """
    from databricks.sdk.service.sql import StatementState

    response = client.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=settings.DRP_UC_WAREHOUSE_ID,
        wait_timeout="30s",
    )
    state = response.status.state
    statement_id = response.statement_id

    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while state in (StatementState.PENDING, StatementState.RUNNING):
        if time.monotonic() >= deadline:
            logger.warning("uc_store: statement %s still %s after %ds, giving up", statement_id, state, _POLL_TIMEOUT_SECONDS)
            return None
        time.sleep(_POLL_INTERVAL_SECONDS)
        response = client.statement_execution.get_statement(statement_id)
        state = response.status.state

    if state != StatementState.SUCCEEDED:
        error = response.status.error
        logger.warning("uc_store: statement %s ended in %s: %s", statement_id, state, error)
        return None
    return response


def _run_statement(client, statement: str) -> bool:
    """Execute `statement` for effect. True only once the warehouse reports SUCCEEDED."""
    return _await_statement(client, statement) is not None


def fetch_rows(statement: str) -> Optional[list[Dict[str, Any]]]:
    """
    Run a SELECT and return its rows as dicts keyed by column name.

    The read counterpart of `insert_rows`/`upsert_rows`, and the path by which the
    API reads back what a Databricks job wrote to Gold. Returns None — never raises
    and never an empty list — when the query could not be run at all (no warehouse,
    no SDK, statement failed), so a caller can tell "job produced nothing" apart
    from "we could not look".

    Values arrive from the Statement Execution API as strings regardless of the
    column's SQL type; callers coerce what they need.
    """
    if not settings.DRP_UC_WAREHOUSE_ID:
        logger.warning("uc_store: DRP_UC_WAREHOUSE_ID not set, cannot read")
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        response = _await_statement(client, statement)
    except Exception as exc:  # noqa: BLE001 — a read must not take the job down
        logger.warning("uc_store: query failed: %s", exc)
        return None
    if response is None:
        return None

    manifest = getattr(response, "manifest", None)
    schema = getattr(manifest, "schema", None) if manifest else None
    columns = [c.name for c in (getattr(schema, "columns", None) or [])]
    result = getattr(response, "result", None)
    # A SELECT that matched nothing carries no `data_array` at all, which is a
    # legitimately empty result rather than a failure.
    data_array = (getattr(result, "data_array", None) or []) if result else []

    if not columns:
        logger.warning("uc_store: query returned no column metadata")
        return None
    return [dict(zip(columns, row)) for row in data_array]


def insert_rows(schema: str, table: str, rows: Iterable[Dict[str, Any]]) -> bool:
    """
    Insert rows into `{DRP_UC_CATALOG}.{schema}.{table}`.

    Returns True if the insert ran (or there was nothing to insert), False if it was
    skipped or failed — callers should log context but must not treat False as fatal.
    """
    rows = [r for r in rows if r]
    if not rows:
        return True
    if not settings.DRP_UC_WRITES_ENABLED:
        return False
    if not settings.DRP_UC_WAREHOUSE_ID:
        logger.warning(
            "uc_store: DRP_UC_WAREHOUSE_ID not set, skipping write to %s.%s.%s",
            settings.DRP_UC_CATALOG, schema, table,
        )
        return False

    client = _get_client()
    if client is None:
        return False

    columns = list(rows[0].keys())
    col_list = ", ".join(columns)
    value_rows = [
        "(" + ", ".join(_sql_literal(row.get(col)) for col in columns) + ")"
        for row in rows
    ]
    full_table = f"{settings.DRP_UC_CATALOG}.{schema}.{table}"
    statement = f"INSERT INTO {full_table} ({col_list}) VALUES {', '.join(value_rows)}"

    try:
        if not _run_statement(client, statement):
            return False
        logger.info("uc_store: inserted %d row(s) into %s", len(rows), full_table)
        return True
    except Exception as exc:
        logger.warning("uc_store: insert into %s failed: %s", full_table, exc)
        return False


def upsert_rows(
    schema: str, table: str, key_columns: Iterable[str], rows: Iterable[Dict[str, Any]]
) -> bool:
    """
    MERGE rows into `{DRP_UC_CATALOG}.{schema}.{table}`, matched on `key_columns`.

    Use this for reference/dimension tables (target, disease, compound, patent) that
    are re-populated across many invocations and must not accumulate duplicate keys.
    Fact/log tables (one new row per invocation) should use `insert_rows` instead.
    """
    rows = [r for r in rows if r]
    if not rows:
        return True
    if not settings.DRP_UC_WRITES_ENABLED:
        return False
    if not settings.DRP_UC_WAREHOUSE_ID:
        logger.warning(
            "uc_store: DRP_UC_WAREHOUSE_ID not set, skipping upsert to %s.%s.%s",
            settings.DRP_UC_CATALOG, schema, table,
        )
        return False

    client = _get_client()
    if client is None:
        return False

    key_columns = list(key_columns)
    columns = list(rows[0].keys())
    full_table = f"{settings.DRP_UC_CATALOG}.{schema}.{table}"

    # A VALUES-derived source, deduplicated on the key so MERGE doesn't error on
    # "multiple source rows matched the same target row" within one batch.
    seen = set()
    deduped = []
    for row in rows:
        key = tuple(row.get(k) for k in key_columns)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    # Databricks SQL rejects a column-list alias on a VALUES-derived table inside
    # MERGE's USING clause ("[COLUMN_ALIASES_NOT_ALLOWED] Column aliases are not
    # allowed in MERGE") — build the source as SELECT ... UNION ALL ... instead,
    # with the column names aliased per-SELECT rather than on the derived table.
    source_rows = [
        "SELECT " + ", ".join(f"{_sql_literal(row.get(col))} AS {col}" for col in columns)
        for row in deduped
    ]
    on_clause = " AND ".join(f"target.{k} = source.{k}" for k in key_columns)
    update_cols = [c for c in columns if c not in key_columns]
    set_clause = ", ".join(f"target.{c} = source.{c}" for c in update_cols) or None
    insert_cols = ", ".join(columns)
    insert_vals = ", ".join(f"source.{c}" for c in columns)

    statement = (
        f"MERGE INTO {full_table} AS target "
        f"USING ({' UNION ALL '.join(source_rows)}) AS source "
        f"ON {on_clause} "
        + (f"WHEN MATCHED THEN UPDATE SET {set_clause} " if set_clause else "")
        + f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
    )

    try:
        if not _run_statement(client, statement):
            return False
        logger.info("uc_store: upserted %d row(s) into %s", len(deduped), full_table)
        return True
    except Exception as exc:
        logger.warning("uc_store: upsert into %s failed: %s", full_table, exc)
        return False
