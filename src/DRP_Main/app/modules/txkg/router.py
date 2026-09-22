"""
TxKG router.

Aggregates two route sets, both already carrying "/txkg/" in their paths (the central
router adds no prefix for this module):

- txkg_test.router      — the existing context-graph endpoints (diseases, subgraph,
                          predicted-targets, metapaths, articles, health).
- discovery_router      — the functional spec's three tools: RWRH propagation with
                          degree correction, Known/Hidden categorization, the sourcing
                          gate, per-candidate path reconstruction, novelty confirmation
                          and the single-delivery /txkg/discover chain.
"""
from fastapi import APIRouter

from DRP_Main.app.api.v1.endpoints.txkg_test import router as _legacy_router
from DRP_Main.app.modules.txkg.discovery_router import router as _discovery_router

router = APIRouter()
router.include_router(_legacy_router)
router.include_router(_discovery_router)
