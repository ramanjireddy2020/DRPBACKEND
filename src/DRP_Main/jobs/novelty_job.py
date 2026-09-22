# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5 — Novelty Search (Patents)
# MAGIC Reads `bronze.pharma_patents` (USPTO PatentsView) -> Vector Search ->
# MAGIC cross-encoder rerank -> Mosaic AI patent analysis -> NOVEL verdict.
# MAGIC Writes `gold.novelty_report`.
# MAGIC
# MAGIC **Placeholder** implementation mirroring the platform architecture.

# COMMAND ----------
# MAGIC %run ./_common

# COMMAND ----------
p = get_params()
catalog, schema, combo = p["catalog"], p["schema"], (p["input"] or "protein-drug combination")
drp_job_id = p["drp_job_id"]
ensure_medallion(catalog, schema)
mlflow_log("novelty", {"query": combo})

# COMMAND ----------
def read_bronze_patents():
    """Reads bronze.pharma_patents (pre-ingested USPTO PatentsView, disease-filtered)."""
    step("Read bronze.pharma_patents (USPTO PatentsView, disease-filtered subset)")
    return []


def vector_search_similarity(query: str):
    """Vector Search ANN similarity (bge-large-en embeddings)."""
    return vector_search(query, index="pharma_patents", k=20)


def rerank(query: str, candidates):
    """MiniLM cross-encoder reranking (Model Serving)."""
    return cross_encoder_rerank(query, candidates)


def patent_analysis(query: str):
    """Mosaic AI patent analysis (Llama 3.3 70B) + NOVEL / POSSIBLY_NOVEL verdict."""
    step("Mosaic AI patent analysis (Llama 3.3 70B)")
    mosaic_llm(f"Assess freedom-to-operate / novelty for: {query}")
    return "POSSIBLY_NOVEL"

# COMMAND ----------
read_bronze_patents()
hits = vector_search_similarity(combo)
ranked = rerank(combo, [h.get("id", "") for h in hits])
verdict = patent_analysis(combo)

report = [{"query": combo, "verdict": verdict, "n_candidates": len(ranked)}]
write_gold(catalog, schema, "novelty_report", report, drp_job_id)

dbutils.notebook.exit(f"novelty: {verdict} (placeholder) drp_job_id={drp_job_id}")
