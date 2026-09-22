# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1 — Knowledge Graph (TxKG)
# MAGIC Disease -> target identification over BioKG. Reads Bronze/Silver,
# MAGIC writes `gold.kg_targets` + `gold.kg_relationships`. Invokes AI services.
# MAGIC
# MAGIC **Placeholder** implementation mirroring the platform architecture.

# COMMAND ----------
# MAGIC %run ./_common

# COMMAND ----------
p = get_params()
catalog, schema, disease = p["catalog"], p["schema"], (p["input"] or "Alzheimer disease")
drp_job_id = p["drp_job_id"]
ensure_medallion(catalog, schema)
mlflow_log("txkg", {"disease": disease, "drp_job_id": drp_job_id})

# COMMAND ----------
# MAGIC %md ## Architecture components (placeholders)

# COMMAND ----------
def multi_hop_bfs_traversal(disease: str):
    """Multi-hop BFS traversal over the knowledge graph (GraphFrames)."""
    step("Multi-hop BFS traversal (GraphFrames)")
    return []  # PLACEHOLDER: read silver.kg_edges, run GraphFrames BFS


def identify_targets(disease: str):
    """Disease input + target identification.

    PLACEHOLDER body, but the column contract is real: these are the exact fields
    `app/drp/runners._run_txkg_on_databricks` reads back to build the API's
    `Target` response, so the port must keep every one of them. `interpretation`,
    `method` and `recommendation` are per-run values repeated on each row — the
    API takes them from whichever row it reads first.
    """
    step("Disease input + target identification")
    return [
        {
            "uniprot_id": "PLACEHOLDER",
            "name": "PLACEHOLDER_TARGET",
            "gene_name": None,
            "full_name": None,
            "disease": disease,
            "disease_id": "",
            "score": 0.0,
            "corrected_score": 0.0,
            "hop_distance": None,
            "connection_types": [],
            "category": "Hidden/Novel",
            "confirmed": False,
            "sourcing_status": "unsourced",
            "novelty_label": "Unknown",
            "supporting_sources": [],
            "interpretation": "",
            "method": {},
            "recommendation": {},
        }
    ]


def tag_relationship_types(targets):
    """Relationship type tagging (CAUSES / ASSOCIATED_WITH / BIOMARKER_OF)."""
    step("Relationship type tagging (CAUSES/ASSOCIATED_WITH/BIOMARKER_OF)")
    return [
        {"uniprot_id": t["uniprot_id"], "disease": t["disease"], "relation": "ASSOCIATED_WITH"}
        for t in targets
    ]


def interpret_relationships(targets):
    """Understand disease-target relationships via Mosaic AI (Llama 3.3 70B)."""
    step("Understand disease-target relationships (LLM interpretation)")
    return mosaic_llm(f"Interpret disease-target relationships for {disease}")

# COMMAND ----------
multi_hop_bfs_traversal(disease)
targets = identify_targets(disease)
relationships = tag_relationship_types(targets)
interpret_relationships(targets)

write_gold(catalog, schema, "kg_targets", targets, drp_job_id)
write_gold(catalog, schema, "kg_relationships", relationships, drp_job_id)

dbutils.notebook.exit(f"txkg: OK (placeholder) drp_job_id={drp_job_id}")
