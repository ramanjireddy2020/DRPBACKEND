# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3 — Drug Curation (CurateX)
# MAGIC Compound retrieval -> compound verification. Writes `gold.drug_candidates`.
# MAGIC Uses Mosaic AI + shared PubMed evidence.
# MAGIC
# MAGIC **Placeholder** implementation mirroring the platform architecture.

# COMMAND ----------
# MAGIC %run ./_common

# COMMAND ----------
p = get_params()
catalog, schema, criteria = p["catalog"], p["schema"], (p["input"] or "default criteria")
drp_job_id = p["drp_job_id"]
ensure_medallion(catalog, schema)
mlflow_log("curatex", {"criteria": criteria})

# COMMAND ----------
def compound_retrieval(criteria: str):
    """Compound retrieval — LLM compound generation + criteria matching."""
    step("Compound retrieval")
    mosaic_llm(f"Generate candidate compounds for: {criteria}")
    return [{"compound": "PLACEHOLDER_CMPD", "criteria": criteria, "confidence": 0.0}]


def compound_verification(compounds):
    """Compound verification — evidence + URL validation (Lambda) placeholder."""
    step("Compound verification")
    return compounds

# COMMAND ----------
compounds = compound_retrieval(criteria)
candidates = compound_verification(compounds)

write_gold(catalog, schema, "drug_candidates", candidates, drp_job_id)

dbutils.notebook.exit(f"curatex: OK (placeholder) drp_job_id={drp_job_id}")
