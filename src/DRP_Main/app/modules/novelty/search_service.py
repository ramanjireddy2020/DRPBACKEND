# """
# Novelty search service — re-exports functions from the implementation module.

# The full Gemini + SerpAPI logic lives in app/api/v1/endpoints/novelty_search.py.
# This module provides a clean import path for other parts of the codebase.
# """
# from app.api.v1.endpoints.novelty_search import (
#     fetch_all_sources,
#     find_cached_context,
#     clean_text,
#     clean_patent_id,
#     search_pubmed,
#     search_wikipedia,
#     search_web,
#     search_patents,
#     search_patents_fallback,
#     search_results_cache,
#     gemini_client,
#     MODEL_NAME,
# )

# __all__ = [
#     "fetch_all_sources",
#     "find_cached_context",
#     "clean_text",
#     "clean_patent_id",
#     "search_pubmed",
#     "search_wikipedia",
#     "search_web",
#     "search_patents",
#     "search_patents_fallback",
# ]
