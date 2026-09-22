# Databricks notebook source
# MAGIC %md
# MAGIC # DRP — Shared Job Helpers (placeholders)
# MAGIC
# MAGIC Common utilities shared by every module job in the Drug Repurposing Platform.
# MAGIC Everything here is a **placeholder** that mirrors the target architecture
# MAGIC (AWS + Databricks). Replace the bodies during the migration.
# MAGIC
# MAGIC Medallion layout (all Delta, UC-governed):
# MAGIC `bronze` (raw) -> `silver` (cleaned/deduped) -> `gold` (curated, versioned).

# COMMAND ----------

from datetime import datetime, timezone


def get_params():
    """Read standard job parameters (catalog/schema/input) from widgets.

    `drp_job_id` is passed by the API on every run it triggers (see
    `app/drp/execution.py`) and must be written onto every Gold row this run
    produces — it is the key the API reads its own results back on. It is empty
    for a run started by hand from the Jobs UI.

    `query` is the module's natural-language input under the name the API sends;
    it falls back to `input` so a manually-started run still works.
    """
    # Defaults match the deployed medallion (drug_repository.bronze/silver/gold),
    # so a run started by hand from the Jobs UI writes where the API reads. The
    # API overrides both on every run it triggers.
    dbutils.widgets.text("catalog", "drug_repository")
    dbutils.widgets.text("schema", "gold")
    dbutils.widgets.text("input", "")  # disease name / protein-drug / query, per module
    dbutils.widgets.text("query", "")
    dbutils.widgets.text("drp_job_id", "")
    return {
        "catalog": dbutils.widgets.get("catalog"),
        "schema": dbutils.widgets.get("schema"),
        "input": dbutils.widgets.get("query") or dbutils.widgets.get("input"),
        "drp_job_id": dbutils.widgets.get("drp_job_id"),
    }


def ensure_medallion(catalog: str, schema: str):
    """Create the catalog schema and the bronze/silver/gold namespaces.

    PLACEHOLDER: real deployment provisions UC schemas + external locations
    on S3 (see architecture note: Bronze/Silver/Gold are Delta on S3, UC-enforced).
    """
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    print(f"[medallion] using {catalog}.{schema} (bronze/silver/gold)")


def step(name: str):
    """Log an architecture step so job runs read as a component checklist."""
    print(f"  → [step] {name}")


def write_gold_placeholder(catalog: str, schema: str, table: str, rows: list[dict]):
    """Write a placeholder Gold Delta table (versioned, audit-friendly).

    PLACEHOLDER: real modules write the curated module output here; the app
    reads Gold through Databricks SQL Warehouse / compute.
    """
    full = f"{catalog}.{schema}.{table}"
    if not rows:
        rows = [{"placeholder": True, "note": "no-op run"}]
    for r in rows:
        r.setdefault("_ingested_at", datetime.now(timezone.utc).isoformat())
    df = spark.createDataFrame(rows)
    df.write.mode("overwrite").option("mergeSchema", "true").format("delta").saveAsTable(full)
    print(f"[gold] wrote {df.count()} row(s) -> {full}")
    return full


def write_gold(catalog: str, schema: str, table: str, rows: list[dict], drp_job_id: str):
    """Append this run's rows to a Gold Delta table, tagged with the DRP job id.

    Use this, not `write_gold_placeholder`, for anything the API reads back.
    Appending rather than overwriting is what makes concurrent runs safe: two
    researchers querying the same module at once both write here, and each reads
    back only the rows carrying their own `drp_job_id`.

    JSON-encodes list/dict values so a column stays a plain string — the API
    reads Gold through the Statement Execution API, which returns every value as
    text, and `app/drp/gold.decode()` parses these back.
    """
    import json

    full = f"{catalog}.{schema}.{table}"
    if not rows:
        print(f"[gold] nothing to write -> {full}")
        return full

    stamped = []
    for r in rows:
        row = {
            k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
            for k, v in r.items()
        }
        row["drp_job_id"] = drp_job_id
        row["_ingested_at"] = datetime.now(timezone.utc).isoformat()
        stamped.append(row)

    df = spark.createDataFrame(stamped)
    (df.write.mode("append")
       .option("mergeSchema", "true")
       .format("delta")
       .saveAsTable(full))
    print(f"[gold] appended {df.count()} row(s) -> {full} (drp_job_id={drp_job_id})")
    return full


# ── AI Services & Model Serving (Databricks-native) — placeholders ──────────────

def mosaic_llm(prompt: str, model: str = "databricks-meta-llama-3-3-70b-instruct") -> str:
    """Mosaic AI Model Serving (Llama 3.3 70B) — replaces Groq/Gemini.

    PLACEHOLDER: real call uses the serving endpoint via the OpenAI-compatible
    client or `mlflow.deployments`. Returns a stub string.
    """
    step(f"Mosaic AI LLM call [{model}] — {prompt[:60]!r}")
    return "<<llm-placeholder-response>>"


def mlflow_log(run_name: str, params: dict, metrics: dict | None = None):
    """MLflow Tracking & Registry — replaces Langfuse.

    PLACEHOLDER: real code wraps the module in an mlflow run / traces LLM calls.
    """
    step(f"MLflow log run={run_name} params={params} metrics={metrics or {}}")


def vector_search(query: str, endpoint: str = "drp-vs", index: str = "patent_chunks",
                  k: int = 10) -> list[dict]:
    """Databricks Vector Search (bge-large-en embeddings) — replaces Qdrant.

    PLACEHOLDER: real call uses databricks-vectorsearch client similarity_search.
    """
    step(f"Vector Search ANN k={k} on {endpoint}/{index} — {query[:60]!r}")
    return []


def cross_encoder_rerank(query: str, candidates: list[str],
                         model: str = "databricks-minilm-cross-encoder") -> list[str]:
    """MiniLM Cross-Encoder reranking on Model Serving — replaces HF Inference API.

    PLACEHOLDER: returns candidates unchanged.
    """
    step(f"Cross-encoder rerank {len(candidates)} candidate(s) [{model}]")
    return candidates
