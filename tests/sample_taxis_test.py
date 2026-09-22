"""
Databricks bundle sample — `src/DRP_Main/taxis.py`.

Template leftover, kept because it is the bundle's own smoke test for Spark
access. It needs a configured Databricks workspace: importing
`databricks.sdk.runtime` raises `ValueError: default auth:` when no credentials
are resolvable, which aborted collection for the entire tests/ directory rather
than failing this one file.

The imports are now guarded, so the test skips with a readable reason off-cluster
and still runs wherever `databricks auth login` (or a PAT) is configured.
"""
import pytest

pyspark = pytest.importorskip("pyspark", reason="pyspark not installed")

try:
    from databricks.sdk.runtime import spark  # noqa: F401

    from DRP_Main import taxis

    _SPARK_AVAILABLE = True
    _SKIP_REASON = ""
except Exception as exc:  # noqa: BLE001 — missing SDK, or unresolvable auth
    _SPARK_AVAILABLE = False
    _SKIP_REASON = f"no Databricks Spark session available: {exc}"


@pytest.mark.skipif(not _SPARK_AVAILABLE, reason=_SKIP_REASON)
def test_find_all_taxis():
    results = taxis.find_all_taxis()
    assert results.count() > 5
