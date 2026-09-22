"""
Central API v1 router — aggregates all module routers in pipeline order.
"""
import logging

from fastapi import APIRouter

api_router = APIRouter()
logger = logging.getLogger(__name__)


def _include(module_path: str, attr: str = "router", **kwargs) -> None:
    """Import a module's router and include it, skipping (with a warning) any
    module that cannot be imported in the current environment — e.g. a missing
    API key, data file, or optional dependency. This keeps the application
    bootable (health/docs + whatever modules load cleanly) instead of letting a
    single module's import error crash the whole server at startup."""
    try:
        mod = __import__(module_path, fromlist=[attr])
        api_router.include_router(getattr(mod, attr), **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping router %s — not available: %s", module_path, exc)


# Orchestration — links the app to the per-module Databricks Jobs (run-now).
_include("DRP_Main.app.modules.orchestration.router", prefix="/jobs", tags=["0. Orchestration"])

# Agentic Architecture - LangGraph supervisor endpoint for conversational agent flow
_include("DRP_Main.app.api.v1.agent_routes", prefix="/agentic", tags=["Agent Interface"])

# Pipeline order — numbered so Swagger shows them in sequence.
# Note: txkg_router paths already include "/txkg/" (e.g. /txkg/health) so no prefix is added.
_include("DRP_Main.app.modules.txkg.router",          tags=["1. TxKG — Target Identification"])
_include("DRP_Main.app.modules.literature.router",    prefix="/literature-mining",    tags=["2. Literature Mining"])
_include("DRP_Main.app.modules.drug_curation.router", prefix="/drug-curation",        tags=["3. Drug Curation"])
_include("DRP_Main.app.modules.screening.router",     prefix="/screening",            tags=["4. Screening Suite"])
_include("DRP_Main.app.modules.novelty.router",       prefix="/novelty-search-agent", tags=["5. Novelty Search Agent"])

# Backward-compatible aliases — old URLs (no module prefix) used by the frontend.
_include("DRP_Main.app.modules.literature.router",    tags=["2. Literature Mining (legacy alias)"])
_include("DRP_Main.app.modules.drug_curation.router", tags=["3. Drug Curation (legacy alias)"])
_include("DRP_Main.app.modules.screening.router",     tags=["4. Screening Suite (legacy alias)"])
_include("DRP_Main.app.modules.novelty.router",       tags=["5. Novelty Search Agent (legacy alias)"])

# Auth / User / Items — supporting scaffolding; included only if they load cleanly.
_include("DRP_Main.app.api.v1.endpoints.auth",  prefix="/auth",  tags=["Auth"])
_include("DRP_Main.app.api.v1.endpoints.users", prefix="/users", tags=["Users"])
_include("DRP_Main.app.api.v1.endpoints.items", prefix="/items", tags=["Items"])