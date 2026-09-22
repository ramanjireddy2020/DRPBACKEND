"""
TxKG service — re-exports all service functions from the implementation module.

The full BioKG knowledge-graph logic lives in app/api/v1/endpoints/txkg_test.py.
This module provides a clean import path for other parts of the codebase.
"""
from DRP_Main.app.api.v1.endpoints.txkg_test import (
    load_data,
    load_cache,
    save_cache,
    find_disease,
    extract_targets,
    build_enhanced_subgraph,
    get_llm_interpretation,
    extract_metapaths_for_targets,
    fetch_pubmed_articles_for_target,
    fetch_uniprot_name_enhanced,
    fetch_multiple_uniprot_names_enhanced,
    # Global state references (read-only)
    df_links,
    id_to_name,
    id_to_type,
    disease_ids,
    diseases_in_kg,
    drug_ids,
    protein_ids,
    TARGET_CACHE,
    METAPATH_CACHE,
)

__all__ = [
    "load_data",
    "load_cache",
    "save_cache",
    "find_disease",
    "extract_targets",
    "build_enhanced_subgraph",
    "get_llm_interpretation",
    "extract_metapaths_for_targets",
    "fetch_pubmed_articles_for_target",
    "fetch_uniprot_name_enhanced",
    "fetch_multiple_uniprot_names_enhanced",
]
