"""
Aggregate router for the DRP `/v1` API.

Mounted by `app/main.py` at the `/v1` prefix declared by the spec's server URL.
Runner registration happens by importing `runners` (its module-level `@register`
decorators populate the job registry).
"""
from fastapi import APIRouter

from DRP_Main.app.drp import runners  # noqa: F401 — registers job runners
from DRP_Main.app.drp.routers import (
    articles,
    auth,
    curatex,
    dashboard,
    files,
    kg,
    litminex,
    modules,
    novsearch,
    projects,
    screensuite,
    sessions,
    txkg,
)

drp_router = APIRouter()

# Order matters only where literal paths could be shadowed by path params;
# each module already declares its literal routes before its parameterised ones.
for module in (
    auth,
    dashboard,
    modules,
    sessions,
    txkg,
    kg,
    litminex,
    curatex,
    screensuite,
    novsearch,
    articles,
    projects,
    files,
):
    drp_router.include_router(module.router)
