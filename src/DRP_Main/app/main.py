"""
InnoDD API — main application entry point.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from DRP_Main.app.api.v1.router import api_router
from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
## InnoDD API — Pharmaceutical Research Platform

Pipeline modules (in order):

1. **TxKG** — Knowledge-graph based protein target identification
2. **Literature Mining** — PubMed search with LLM relevance scoring
3. **Drug Curation** — AI-powered drug compound discovery
4. **Screening Suite** — Molecular docking pipeline (AutoDock Vina)
5. **Novelty Search** — Patent search and novelty analysis

### Key Endpoints
- `GET /api/v1/txkg/predicted-targets` — Therapeutic targets
- `GET /api/v1/literature-mining/search-keywords/` — Literature search
- `POST /api/v1/drug-curation/fetch_compounds` — Drug curation
- `POST /api/v1/screening/process-protein-v2/` — Queue docking job
- `POST /api/v1/novelty-search-agent/agent/run` — Novelty search agent
    """,
    version=settings.PROJECT_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# `allow_origins=["*"]` together with `allow_credentials=True` is not a valid
# combination per the CORS spec — a browser refuses a wildcard on a credentialed
# request. Configure CORS_ORIGINS with the real frontend origins in any
# deployment that relies on cookies; the wildcard here is for token-only clients.
_cors_origins = settings.CORS_ORIGINS or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    # X-Drp-Token carries the app token where the Databricks Apps OAuth proxy
    # would otherwise consume Authorization (see drp/deps.py).
    allow_headers=["*", "Authorization", "Content-Type", "X-Drp-Token"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    logger.info("→ %s %s from %s", request.method, request.url.path, client_host)
    try:
        response = await call_next(request)
        logger.info("← %s %s %d", request.method, request.url.path, response.status_code)
        return response
    except Exception as e:
        logger.error("Error on %s %s: %s", request.method, request.url.path, e, exc_info=True)
        raise


app.include_router(api_router, prefix="/api/v1")

# ── DRP public API (/v1) ──────────────────────────────────────────────────────
# The frontend-facing contract (see the DRP OpenAPI spec): auth/onboarding,
# dashboard, sessions, the five agent modules, knowledge graph, articles,
# projects and files/export. Included defensively so a database or optional
# dependency problem degrades to "/v1 unavailable" instead of a boot failure.
try:
    from DRP_Main.app.drp.bootstrap import init_drp
    from DRP_Main.app.drp.jobs import capture_event_loop
    from DRP_Main.app.drp.router import drp_router

    init_drp()
    app.include_router(drp_router, prefix="/v1")

    @app.on_event("startup")
    async def _capture_drp_loop():
        # Agent jobs are scheduled from sync endpoints (threadpool workers), which
        # have no running loop of their own — hand them the server's.
        capture_event_loop()

    logger.info("DRP /v1 API mounted")
except Exception as exc:  # noqa: BLE001
    logger.error("DRP /v1 API not mounted: %s", exc, exc_info=True)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "version": settings.PROJECT_VERSION}


@app.exception_handler(404)
async def not_found_handler(request, exc):
    logger.warning("404 Not Found: %s", request.url.path)
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Endpoint {request.url.path} does not exist",
            "docs": "/docs",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error("500 on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Check the logs.",
            "path": str(request.url.path),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
