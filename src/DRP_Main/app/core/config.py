"""
Centralised application settings loaded from .env via pydantic-settings.
All secrets live in .env — never hardcoded in source.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Project ────────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "InnoDD API"
    PROJECT_VERSION: str = "2.3.0"

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./innodd.db"
    NOVELTY_DATABASE_URL: str = "sqlite:///./noveltysearch.db"

    # ── Auth ───────────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ── AWS Cognito (the frontend's identity provider) ────────────────────────
    # Set COGNITO_USER_POOL_ID to have /v1 accept Cognito-issued RS256 tokens.
    # Left empty, /v1 falls back to its own HS256 tokens from /v1/auth/login.
    COGNITO_REGION: str = "us-east-1"
    COGNITO_USER_POOL_ID: str = ""
    COGNITO_APP_CLIENT_ID: str = ""
    # Create a local `users` row the first time an unknown Cognito subject
    # presents a valid token. Off means every user must be provisioned already.
    COGNITO_AUTO_PROVISION: bool = True
    # Accept the access token from an `access_token` query parameter as well as
    # from a header. Needed only because the AWS Lambda proxy in front of this
    # API forwards query params and bodies but drops the caller's headers, so a
    # per-user token has no header to travel in. Query strings land in access
    # logs, so this is a stopgap: turn it off once the proxy forwards headers.
    DRP_ALLOW_QUERY_TOKEN: bool = False

    # Explicit browser origins allowed to call the API. Empty falls back to the
    # credential-less wildcard, which is fine for bearer-token clients only.
    CORS_ORIGINS: list[str] = []

    # ── LLM providers ─────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    # llama-3.3-70b-versatile was retired from Groq's lineup; gpt-oss-120b is the
    # current flagship-strength open model there.
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Databricks pay-per-token Foundation Model API — replaces Groq/Gemini for
    # TxKG and Novelty so no external LLM key or always-on serving endpoint is needed.
    DATABRICKS_LLM_ENDPOINT: str = "databricks-meta-llama-3-3-70b-instruct"

    # ── Search ─────────────────────────────────────────────────────────────────
    SERPAPI_API_KEY: str = ""

    # ── Langfuse observability ─────────────────────────────────────────────────
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

    # ── Azure OpenAI (optional) ────────────────────────────────────────────────
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_API_VERSION: str = "2024-05-01-preview"

    # ── Literature Mining settings ─────────────────────────────────────────────
    LIT_MAX_WORKERS: int = 10
    LIT_BATCH_SIZE: int = 10
    LIT_MESH_CACHE_DAYS: int = 30
    LIT_REQUEST_TIMEOUT: int = 15
    LIT_MAX_ABSTRACT_LENGTH: int = 500
    LIT_MESH_CACHE_DB: str = "data/mesh_cache.db"

    # ── LitMineX (spec-aligned literature mining pipeline) ────────────────────
    # Tool 1 — retrieval
    LITMINEX_RETRIEVAL_LIMIT: int = 100          # PMIDs pulled from esearch
    LITMINEX_EFETCH_BATCH: int = 50              # PMIDs per efetch call
    LITMINEX_FETCH_FULLTEXT: bool = True         # pull PMC OA sections when available
    LITMINEX_EXCLUDE_REVIEWS: bool = False       # optional publication-type filter
    # Tool 2 — NLP models (fall back to the rule-based path when unavailable)
    LITMINEX_NER_DISEASE_MODEL: str = "en_ner_bc5cdr_md"
    LITMINEX_NER_GENE_MODEL: str = "en_ner_bionlp13cg_md"
    LITMINEX_EMBEDDING_MODEL: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    LITMINEX_ENABLE_EMBEDDINGS: bool = True
    # Tool 3 — summarisation
    LITMINEX_SUMMARY_TOP_N: int = 10
    LITMINEX_SUMMARY_MIN_SCORE: float = 50.0
    # Use the batched mining agent (literature/mining_agent.py) as the primary
    # retrieval path: 2 PubMed requests per run instead of 2 per target, which is
    # what keeps it inside NCBI's unauthenticated rate limit. The spec chain needs
    # NER models, embeddings and an LLM scorer that are not installed on Databricks
    # Apps, so it stays as the fallback. Set false to restore the old order.
    LITMINEX_PREFER_AGENT: bool = True

    # ── Drug Curation settings ─────────────────────────────────────────────────
    DRUG_OUTPUT_DIR: str = "data/drug_curation"
    DRUG_MIN_RESULTS: int = 15
    DRUG_MAX_RESULTS: int = 50
    DRUG_PUBMED_MAX_RESULTS: int = 3
    DRUG_REQUEST_TIMEOUT: int = 10

    # ── CurateX (spec-aligned drug curation pipeline) ─────────────────────────
    CURATEX_REQUEST_TIMEOUT: int = 20        # per source HTTP call
    CURATEX_MAX_RETRIES: int = 2             # retries on 429/5xx before giving up
    CURATEX_MAX_KNOWN_LIGANDS: int = 300     # ChEMBL molecules pulled per target
    CURATEX_UNIVERSE_LIMIT: int = 500        # Tool 2 candidate pool size
    CURATEX_ENRICH_TOP_N: int = 60           # candidates given Open Targets + DailyMed
    CURATEX_EVIDENCE_TOP_N: int = 10         # candidates given Tool 3 literature validation
    CURATEX_MIN_CLINICAL_PHASE: int = 1      # 1 keeps investigational compounds in the pool

    # ── NovSearch (spec-aligned patent novelty pipeline, AWS/Databricks-native) ─
    # Databricks workspace — Model Serving + Vector Search both authenticate here.
    DATABRICKS_HOST: str = ""
    DATABRICKS_TOKEN: str = ""
    # Falls back to a named `.databrickscfg` profile (OAuth via `databricks auth
    # login`) when set — used by uc_store.py so local dev doesn't need a PAT on
    # workspaces where PAT creation is disabled by policy.
    DATABRICKS_CONFIG_PROFILE: str = ""
    # Tool 1 — Google Patents search via SerpAPI (SERPAPI_API_KEY above). Replaced
    # USPTO PatentsView (retired 2026-03-20) and its Open Data Portal successor
    # (requires an ID.me-verified MyUSPTO account, never obtained).
    NOVSEARCH_REQUEST_TIMEOUT: int = 30
    NOVSEARCH_SEARCH_FETCH_MULTIPLIER: int = 4   # candidates pulled per requested result
    NOVSEARCH_RRF_K: int = 60
    NOVSEARCH_TITLE_PIN_WEIGHT: float = 6.0
    # Tool 1 — cross-encoder rerank. No dedicated reranker endpoint exists on the
    # target workspace; retrieval_service.cross_encoder_rerank() already degrades
    # to the RRF order on any serving failure, so a wrong/missing name here is
    # tolerated by design rather than fixed with a same-purpose substitute.
    NOVSEARCH_CROSS_ENCODER_ENDPOINT: str = "ms-marco-minilm-l6-v2"
    # Tool 2 — embeddings. BGE-large-en-v1.5 isn't deployed on the target
    # workspace; GTE-large-en-v1.5 is a Databricks-hosted Foundation Model
    # endpoint there, also 1024-dim, OpenAI-embeddings-compatible.
    NOVSEARCH_EMBEDDING_ENDPOINT: str = "databricks-gte-large-en"
    NOVSEARCH_EMBEDDING_DIM: int = 1024
    NOVSEARCH_EMBED_BATCH: int = 32
    # Tool 2 — Databricks Vector Search (direct-access index, vectors supplied by us)
    NOVSEARCH_VS_ENDPOINT: str = "novsearch-vs"
    NOVSEARCH_VS_INDEX: str = "drug_repository.vectors.patent_chunks"
    NOVSEARCH_MAX_PATENTS: int = 20              # store cap; oldest evicted past this
    # Tool 3 — synthesis LLM, with a Groq fallback (core/llm.py, GROQ_API_KEY
    # above). SaulLM isn't deployed on the target workspace; Llama 3.3 70B is a
    # Databricks pay-per-token Foundation Model endpoint there (same one
    # TxKG/Novelty's legacy path already uses).
    NOVSEARCH_LLM_ENDPOINT: str = "databricks-meta-llama-3-3-70b-instruct"
    NOVSEARCH_LLM_NAME: str = "databricks-meta-llama-3-3-70b-instruct"
    NOVSEARCH_LLM_MAX_TOKENS: int = 2048
    # Tool 3 — retrieval breadth
    NOVSEARCH_TOP_K_CHUNKS: int = 8              # chunks pulled per patent for synthesis

    # ── ScreenSuite (docking) settings ────────────────────────────────────────
    # Two external executables, neither pip-installable. Both are resolved at
    # call time: an explicit path here wins, otherwise PATH is searched, and a
    # missing binary fails that one entity with a readable message rather than
    # breaking module import (see the screening module's docstring).
    SCREENING_VINA_PATH: str = ""            # AutoDock Vina binary (vina / vina.exe)
    SCREENING_OBABEL_PATH: str = ""          # Open Babel binary (obabel / obabel.exe)
    SCREENING_SUBPROCESS_TIMEOUT: int = 360  # per vina/obabel invocation, seconds
    # Where run output is written. Empty keeps the in-repo `app/data` layout;
    # point it at a UC volume path (/Volumes/...) on Databricks, where local
    # cluster storage is ephemeral.
    SCREENING_OUTPUT_ROOT: str = ""
    # Docking box. Boxing the whole protein is blind docking — poses are not
    # site-specific and affinities are not comparable across receptors — so a
    # ligand-derived site box is preferred and the whole-protein extent is only
    # the explicit fallback.
    SCREENING_BOX_MODE: str = "site"         # "site" | "protein"
    SCREENING_BOX_PADDING: float = 8.0       # Å added around a detected site
    SCREENING_BOX_MIN_SIZE: float = 20.0     # Å floor on any box edge
    SCREENING_BOX_MAX_SIZE: float = 40.0     # Å ceiling (vina slows sharply past this)
    SCREENING_NUM_MODES: int = 9
    SCREENING_EXHAUSTIVENESS: int = 8
    SCREENING_TOP_FRACTION: float = 0.05     # top-N% of ligands kept per protein
    # Cost guard. proteins × drugs is unbounded once runs are agent-triggered,
    # so a batch over this is refused before any compute is spent.
    SCREENING_MAX_COMBINATIONS: int = 200
    # Resolution step (RCSB / PubChem name → identifier lookups).
    SCREENING_RESOLVE_TIMEOUT: int = 20
    SCREENING_RESOLVE_MAX_MATCHES: int = 5   # shortlist size surfaced for confirmation

    # ── BioKG (TxKG) settings ─────────────────────────────────────────────────
    BIOKG_DATA_DIR: str = "/home/azureuser/innoddapi/app/modules/txkg/biokg_data"
    TXKG_MAX_HOP_DEPTH: int = 3

    # ── Unity Catalog persistence (bronze/silver/gold module output) ──────────
    # A SQL Warehouse ID is required to execute statements via the Databricks SDK's
    # Statement Execution API. Writes are skipped (logged, not fatal) when unset —
    # a missing warehouse must not break the agent pipeline.
    DRP_UC_CATALOG: str = "drug_repository"
    DRP_UC_WAREHOUSE_ID: str = ""
    DRP_UC_WRITES_ENABLED: bool = True

    # ── Agent job execution backend ─────────────────────────────────────────────
    # Where a /v1 agent job's work actually runs:
    #   "inprocess"  — the runner calls app/modules/* directly (default; the only
    #                  mode that works with no Databricks workspace attached).
    #   "databricks" — the runner triggers that module's Databricks job, polls it,
    #                  and reads the output back from Gold.
    # The API contract is identical either way, so this can be flipped per
    # deployment without the frontend noticing. Modules with no job linked fall
    # back to in-process so one unported module cannot take the whole API down.
    DRP_EXECUTION_BACKEND: str = "inprocess"
    # Per-module opt-in to the Databricks job, independent of the global switch
    # above: a comma-separated list of module keys (e.g. "ScreenSuite"). Lets one
    # module move to its real job while the rest stay in-process — flipping the
    # global switch would also delegate modules whose job notebooks are still
    # placeholders, and serve their PLACEHOLDER rows to the frontend.
    DRP_DATABRICKS_MODULES: str = ""
    # Schema holding the Gold tables the module jobs write. `drug_repository` is
    # laid out as a medallion (bronze/silver/gold/reference/vectors), so Gold
    # output belongs in `gold` — not the flat `drp` schema the placeholder job
    # notebooks defaulted to. The API sends this to every job it triggers, so
    # both sides stay on one schema.
    DRP_GOLD_SCHEMA: str = "gold"
    # How often to poll a triggered run, and how long before giving up on it.
    # Serverless compute still costs tens of seconds to start, so polling faster
    # than a few seconds only adds API calls.
    DRP_JOB_POLL_SECONDS: int = 5
    DRP_JOB_TIMEOUT_SECONDS: int = 1800

    # ── DRP public API (/v1 — see static/drp-api-docs.html spec) ──────────────
    DRP_REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    DRP_EXPORT_DIR: str = "data/drp_exports"
    DRP_UPLOAD_DIR: str = "data/drp_uploads"
    # Seed a login-able account on startup so the /v1 surface is usable before
    # a real identity provider is wired up. Disable in production.
    DRP_SEED_DEMO_USER: bool = True
    DRP_DEMO_EMAIL: str = "priya@drp.app"
    DRP_DEMO_PASSWORD: str = "drp-demo-password"
    DRP_DEMO_NAME: str = "Dr. Priya"
    DRP_DEMO_ROLE: str = "Chief Researcher"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
