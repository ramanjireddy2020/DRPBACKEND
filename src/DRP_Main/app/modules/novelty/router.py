"""
Novelty Search router.

Two surfaces, following the LitMineX / CurateX precedent:

  * `novsearch_router` — the spec-aligned chain (USPTO PatentsView + Databricks
    Model Serving + Databricks Vector Search + SaulLM). This is what the agent
    and `/v1/agents/novsearch/*` run.
  * `novelty_search_agent.router` — the legacy Google-Patents / SerpAPI / Qdrant /
    Gemini agent (`/agent/*`), kept because it is a different contract with its
    own usage-tracking and history endpoints.
"""
from fastapi import APIRouter

from DRP_Main.app.modules.novelty.novsearch_router import router as novsearch_router

router = APIRouter()
router.include_router(novsearch_router)

try:  # The legacy agent constructs Gemini/Qdrant clients at import time.
    from DRP_Main.app.api.v1.endpoints.novelty_search_agent import router as legacy_router

    router.include_router(legacy_router)
except Exception as exc:  # noqa: BLE001 — a missing legacy dep must not break the module
    import logging

    logging.getLogger(__name__).warning(
        "Legacy novelty agent unavailable (%s); serving the spec-aligned chain only", exc
    )
