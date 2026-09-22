# CLAUDE.md — DRP_Main (InnoDD API)

## Project Overview

DRP_Main is a **Databricks Asset Bundle** that packages the InnoDD API — a FastAPI-based
pharmaceutical research platform. It exposes a 5-module drug discovery pipeline via REST API,
running on port 8080 with a uvicorn server (locally, inside a Conda `pymol_env` environment).

**Tech stack:** Python 3.12, FastAPI, SQLAlchemy, Pydantic v2, pydantic-settings, Langfuse 2.x,
Groq, Google Gemini 2.5 Flash, SerpAPI, AutoDock Vina + Open Babel (external executables),
RDKit, PDBFixer/OpenMM, ProLIF, BioPython, NetworkX. Deployable to Databricks via the bundle
(`databricks.yml`).

---

## Repository Layout

This is a Databricks default-python bundle. The FastAPI application lives under `src/DRP_Main/`.

```
DRP_Main/
├── databricks.yml              # Bundle definition (targets: dev, prod)
├── resources/                  # Databricks job & pipeline configs
├── pyproject.toml              # Package metadata + dependencies (name: DRP_Main)
├── requirements.txt            # Pip dependency list (mirror of pyproject)
├── Dockerfile                  # Conda + AutoDock Vina + openbabel image
├── scripts/restart_ddapi.sh    # Local uvicorn restart helper
├── data/                       # Runtime data (SQLite DBs, outputs) — gitignored
├── static/                     # swagger.html, upload.html
├── tests/                      # pytest suite
└── src/
    ├── DRP_Main/
    │   ├── main.py             # CLI entry (Databricks job) + run_fastapi() (--api)
    │   ├── taxis.py            # Databricks sample (template leftover)
    │   └── app/                # ← the FastAPI application
    └── DRP_Main_etl/           # Databricks ETL sample (template leftover)
```

> **Import convention:** every module inside the app imports with the full package path,
> e.g. `from DRP_Main.app.core.config import settings` — NOT `from app.core.config`.
> The package root is `src/`, so `DRP_Main` must be importable (run from `src/` or install the package).

---

## Application Structure (`src/DRP_Main/app/`)

All pipeline logic lives in `app/modules/`. Each module is self-contained with its own
router, services, and schemas. Only the `modules/*` architecture is wired into the router.

```
src/DRP_Main/app/
├── core/
│   ├── config.py          # pydantic-settings, reads .env — all secrets here
│   ├── logging.py         # setup_logging() + get_logger() + init_langfuse()
│   └── security.py        # JWT / bcrypt
├── shared/
│   └── file_utils.py      # protein_output_directory(), convert_xml_to_json(), extract_interactions()
├── modules/
│   ├── txkg/              # Module 1 — BioKG knowledge graph target identification
│   ├── literature/        # Module 2 — PubMed search + MeSH + LLM relevance scoring
│   ├── drug_curation/     # Module 3 — CurateX spec chain (+ legacy Groq generator)
│   ├── screening/         # Module 4 — AutoDock Vina molecular docking pipeline
│   └── novelty/           # Module 5 — Patent novelty search (Gemini + SerpAPI)
├── api/v1/
│   ├── router.py          # Central router aggregating all modules
│   └── endpoints/
│       ├── txkg_test.py           # TxKG implementation (large; do not split) — used by modules/txkg
│       ├── novelty_search_agent.py# Novelty agent implementation — used by modules/novelty
│       ├── auth.py / users.py / items.py  # Auth/user scaffolding
│       └── .env                   # Module-level env overrides (DATABASE_URL for agent_data.db)
├── db/ models/ schemas/   # Auth/user infrastructure (SQLAlchemy models, Pydantic schemas)
├── services/
│   └── user_service.py    # Auth/user service (used by auth.py, users.py)
└── main.py                # FastAPI app factory
```

### The DRP public API (`app/drp/`, mounted at `/v1`)

`app/modules/*` is the **internal** pipeline surface (`/api/v1/*`). `app/drp/` is the
**frontend-facing** contract from the DRP OpenAPI spec (`drp-api-docs.html`), mounted at
`/v1` and covering all 47 spec paths: auth/onboarding, dashboard, sessions, the five agent
modules, knowledge graph, articles, projects and files/export.

```
src/DRP_Main/app/drp/
├── router.py       # aggregates routers/*, mounted at /v1 by app/main.py
├── routers/        # one file per spec tag (auth, dashboard, sessions, txkg, kg, …)
├── schemas.py      # camelCase wire models — exactly the spec's shapes
├── models.py       # SQLAlchemy tables, all prefixed drp_
├── jobs.py         # async job lifecycle: 202 {jobId} → status/result polling
├── runners.py      # adapters from a job onto app/modules/* (all imports lazy)
├── dispatch.py     # composer query → module → job params (@mention + keywords)
├── catalog.py      # module / therapeutic-area / quick-action catalogues
├── security.py     # JWT access tokens + opaque refresh tokens
├── deps.py         # bearer auth dependency, user profile accessor
└── bootstrap.py    # create_all + demo-user seed, called from app/main.py
```

Rules for this layer:
- **`/v1` never runs science inline.** A request creates a `DrpJob` and returns `202
  {jobId}`; `runners.py` executes it on the server event loop. `jobs.enqueue()` uses the
  loop captured by the app's startup hook, because sync (`def`) endpoints run in a
  threadpool with no running loop of their own.
- **Every pipeline import inside a runner is lazy.** TxKG needs the BioKG dataset,
  ScreenSuite needs PyMOL/Vina, NovSearch needs Gemini + Qdrant. A missing dependency must
  fail one job with a readable message, not break API import.
- **Jobs use their own session** (`jobs.JobSession`), never the request-scoped `get_db()`.
- Wire models are camelCase and live only in `drp/schemas.py`; do not leak module schemas
  (`ArticleResult`, `CompoundResult`, …) through `/v1`.
- Auth is bearer-only via `deps.current_user`. `drp/security.py` calls `bcrypt` directly —
  passlib 1.7.4 cannot read the version metadata of bcrypt ≥ 4.1.
- Login needs an account: `DRP_SEED_DEMO_USER` (default on) seeds `DRP_DEMO_EMAIL` /
  `DRP_DEMO_PASSWORD`. **Turn it off in production.**

**Pipeline order** (reflected in Swagger tag numbering):
1. TxKG → 2. Literature Mining → 3. Drug Curation → 4. Screening → 5. Novelty Search

> **History note:** an older monolithic architecture (`endpoints/{drug_curation,literature_mining,
> process_protein,novelty_search,...}.py` + `services/{auto_dock_vina,queue_service,...}.py`) was
> removed when the app was consolidated into `modules/`. Only `txkg_test.py` and
> `novelty_search_agent.py` were retained from the old `endpoints/` because the modules wrap them.
> Do not re-introduce logic under `endpoints/` or `services/` — add it to the relevant module.

---

## Running the App

### Locally (uvicorn)

```bash
# From the repo root, run from the src/ directory so DRP_Main is importable:
bash scripts/restart_ddapi.sh

# Or manually (must be in pymol_env conda env):
conda activate pymol_env
cd src
nohup uvicorn DRP_Main.app.main:app --host 0.0.0.0 --port 8080 --reload &

# Verify
curl http://localhost:8080/health
```

**Docking prerequisites.** The API boots and serves without any of them. Everything except the
Vina call itself comes from pip:

```bash
pip install -r requirements.txt       # includes openbabel-wheel, pdbfixer, openmm, prolif, meeko

# AutoDock Vina is a download, not a pip package — pick your platform's asset:
#   https://github.com/ccsb-scripps/AutoDock-Vina/releases
# then drop it at DRP_Main/bin/vina[.exe] (searched automatically), or set
# SCREENING_VINA_PATH to wherever it lives.

curl http://localhost:8080/api/v1/screening/health   # says what actually runs here
```

With the pip stack alone you get retrieval, receptor preparation, PDBQT conversion, the docking box,
ligand preparation and interaction profiling. Only the docking step needs the binary.

If `/health` reports `"reason": "blocked_by_policy"`, the binary is present and the OS refused to
run it — endpoint security. Reinstalling will not help; that path needs an allowlist entry, or run
it on a Linux host / the Databricks job.

The old `pymol_env` conda environment is no longer required — PyMOL and PLIP have been removed.

### On Databricks (bundle)

```bash
# Current workspace. The Terraform-free engine is required on these laptops.
DATABRICKS_BUNDLE_ENGINE=direct databricks bundle deploy --target bpuvvada --profile bpuvvada

# dev / prod targets point at the old workspace (dbc-9c7c244a-a883)
databricks bundle deploy --target dev     # or --target prod
databricks bundle run
```

---

## Key Architectural Decisions

### TxKG router — no prefix
`txkg_test.py` routes already include `/txkg/` in their paths (e.g. `@router.get("/txkg/health")`).
The central router includes `txkg_router` **without** any prefix to avoid double-prefixing
(`/txkg/txkg/health`). `modules/txkg/router.py` aggregates two route sets, both already
carrying `/txkg/`: the legacy `txkg_test.router` and `discovery_router` (below).

### TxKG discovery — the functional spec's three tools
`modules/txkg/discovery_service.py` implements the TxKG functional specification and is
kept separate from the legacy context-graph code in `txkg_test.py`, which it reuses only
for data loading (`df_links`, `id_to_type`, `ctx_adj`, `edge_lookup`, `PROTEIN_NAME_INDEX`).

| Spec | Where | Note |
|---|---|---|
| §0 node types | `load_annotations()` | `biokg.links.tsv` carries **no** process/function/tissue nodes — GO annotations, the MeSH disease tree and the Reactome pathway hierarchy live in `biokg.properties.*`. These are loaded into this module's own graph (never into `txkg_test`'s globals or pickle), taking the graph from 105k to 207k nodes and 7 to 10 layers. `PROTEIN_EXPRESSED_IN` and `GO_CC` are **off by default** — 1.3M low-specificity edges that would let tissue/compartment co-occurrence dominate propagation. Override with `TXKG_ANNOTATION_EDGES`. |
| §1 disease resolution | `resolve_disease()` | stricter than `txkg_test.find_disease()`, which falls through to bare substring containment and will anchor free text onto an unrelated disease. Returns one clarifying question rather than guessing. |
| §2 propagation | `propagate()` | RWRH power iteration over the **whole** graph. No hop cutoff — the restart probability supplies the distance decay. |
| §3 degree correction | `degree_corrected_scores()` | analytical hypergeometric test conditioned on degree (preferred, one calculation per candidate); adds a z-score against `build_null_model()`'s precomputed degree-matched null when the `.npz` is present. |
| §4 Known/Hidden | `has_curated_association()` | one direct fact: does a `PROTEIN_DISEASE_ASSOCIATION` edge exist. Not hop distance. |
| §5 sourcing gate | `_apply_sourcing_gate()` | resolves each path edge through `EDGE_PROVENANCE`; unsourced Hidden candidates are dropped or flagged `unconfirmed`, never presented as confirmed. |
| §6 path reconstruction | `reconstruct_paths()` | bounded BFS, run **only** for candidates that already survived scoring. Traverses `ctx_adj` **plus** the annotation layers, so processes appear as intermediates. |
| §6/§9 visualization | `render_candidate_subgraph_html()` | openable page under `/api/v1/graphs/discovery/{file}`; candidates coloured by §4 category, outlined grey when §5-unconfirmed, every edge's provenance in its tooltip. |
| §7–§8 novelty + interpretation | `novelty_label()`, `interpret_candidates()` | all NCBI eutils traffic goes through the shared per-event-loop throttle (`_eutils_gate`) — firing these concurrently otherwise returns 429s that silently degrade every novelty label to `Unknown`. Patents come from module 5's novelty agent when importable (latched off after one failure), else direct SerpAPI. Set `NCBI_API_KEY` / `SERPAPI_API_KEY`. |
| §9 output assembly | `build_ranked_table()` | one ranked table, every row labelled; Known/Hidden groups still available separately. |
| §10 recommendation | `build_recommendation()` | names the specific candidate and the factors that drove it. Rule order follows the spec: under-explored confirmed Hidden first, then both-groups-strong, then fall back to Known. |
| §11 supervising layer | `handle_message()`, `SESSION_STATE` | routes a message to the full chain, a scoped follow-up, or a recap; `mode` in the response says which. A follow-up reads already-computed scores/paths — only a fresh disease restarts at §1. |

`/v1/agents/txkg/query` (the frontend contract) runs this chain — `score` on the wire is the
corrected relevance score and `hopDistance` is `null`, because nothing is hop-bounded. The
`Target` wire model carries `category`, `confirmed`, `sourcingStatus`, `noveltyLabel` and
`supportingSources`; `/v1/agents/txkg/insights?tab=recommendations` serves §10's recommendation.

Two disk caches, both rebuilt automatically when their inputs change:
`biokg_data/rwrh_transition.pkl` (keyed on the annotation signature and λ) and
`biokg_data/annotation_edges.pkl`. Delete them or call `/txkg/discovery-clear-cache` to force a
rebuild. Query cost is ~1 s of propagation; the rest of `/txkg/discover` is PubMed and LLM latency.

Calibration tests for the spec's §13 notes live in `tests/test_txkg_discovery.py`. They skip
when the BioKG dataset is absent. Note `tests/conftest.py` requires Databricks credentials,
so the suite cannot run without workspace auth configured.

### LitMineX — the functional spec's three tools
`modules/literature/` carries two surfaces. `scoring_service.LiteratureService` is the
legacy keyword endpoint (`/search-keywords/`), kept because it is a different contract.
`litminex_service.LitMineXService` implements the LitMineX functional specification and is
what the agent and `/v1/agents/litminex/query` run.

| Spec | Where | Note |
|---|---|---|
| §2 input shapes | `RetrievalService.resolve_entities()` | Structured TxKG carry-over skips NER; a fresh query runs through it. If **either** concept is missing it raises `MissingEntityError` so the agent prompts once rather than guessing — one concept alone is not searchable, since Tool 1 exists to AND the two groups. `require_both=False` opts out for `/v1`, where the target picker allows a target with no disease; the scorer's co-occurrence gate then relaxes to the concepts actually searched. |
| §3.1 term extraction | `ner_service.py` | scispaCy `en_ner_bc5cdr_md` (disease) + `en_ner_bionlp13cg_md` (gene/protein), lazy-loaded like PyMOL. Missing models degrade to a rule-based tagger — the API never fails on their absence, but recall drops. |
| §3.2 MeSH expansion | `MeSHService.expand_term()` | Real NLM thesaurus via E-utilities `esearch`/`efetch` on `db=mesh`, SQLite-cached. Only the indented block under `Entry Terms:` is taken — the tree hierarchy that follows is not a synonym set and would drag the query out to terms like "Enzymes". |
| §3.3 query construction | `PubMedService.build_litminex_query()` | Every term field-tagged into **both** `[Title/Abstract]` and `[MeSH Terms]`, OR'd within a concept group, target group **AND** disease group. Capped at `MAX_TERMS_PER_GROUP`; `esearch` uses **POST**, because a fully expanded two-concept query overruns E-utilities' URI limit and returns 414. |
| §3.4 retrieval | `esearch_pmids()`, `fetch_records()`, `fetch_fulltext_sections()` | Batched efetch; PMC open-access full text bucketed into the sections the weight table names. Non-OA articles keep title + abstract and score at reduced positional resolution. |
| §4 scoring | `relevance_service.py` | The four signals, combined by the fixed formula in `relation_config.py` (relation 40 / position 25 / similarity 20 / primary 15). Each component takes the **best** evidence the article offers, so one strong primary result sentence is not diluted by weak co-occurrences. |
| §4.5 similarity | `embedding_service.py` | SapBERT via sentence-transformers, loaded once per process; falls back to token Jaccard when the weights are unavailable. The active backend is reported in the response and on `/health/`. |
| §5 summarisation | `summarization_service.py` | Answers are built **only** from tagged relation sentences, never raw abstracts, with every claim cited `[PMID:…]`. `_cited_pmids` discards citations to PMIDs outside the supplied evidence. |
| §6 QA | `LitMineXService.answer_followup()` | Re-runs Tool 3 alone against the cached ranked set, scoped by PMID; `deep=True` widens context to full section text. |
| §7 supervisor object | `LitMineXService._assemble()` | `module` / `target` / `disease` / `query_answer` / `article_table` / `session_state_update`. |

The weights live in `relation_config.py` so they can be spot-checked against known-good
papers in one place. `tests/test_litminex.py` pins them offline (no network, no LLM) and
asserts the AND'ed query shape — run it after touching any of this.

scispaCy and its two models are optional; see the note in `requirements.txt` for the model
wheel URLs. Without them the module runs, logs the downgrade, and reports
`"ner_backend": "rule-based"`.

### CurateX — the functional spec's three tools
`modules/drug_curation/` carries two surfaces. `curation_service.DrugCurationService`
is the legacy Groq compound generator (`/fetch_compounds`), kept because it is a
different contract — it asks an LLM to *invent* a compound list with a self-reported
confidence score. `curatex_service.CurateXService` implements the CurateX functional
specification and is what the agent and `/v1/agents/curatex/*` run: no LLM anywhere in
the chain, every number traced to a source record.

| Spec | Where | Note |
|---|---|---|
| §2.1 ligand discovery | `ligand_sources.py` | Six sources queried in parallel (ChEMBL activities, Open Targets, BindingDB, PubChem BioAssay, IUPHAR, DGIdb). Wide net on purpose — this set only defines the baseline ranges. DrugBank is excluded throughout (commercial licence). ChEMBL activities are filtered to IC50/Ki/EC50/Kd; the unfiltered table is dominated by percent-inhibition screening points. |
| §2.2 criteria extraction | `ligand_sources.py`, `dailymed_service.py` | ChEMBL + Open Targets + DailyMed only. **The Open Targets Platform v4 schema no longer has `knownDrugs`, `maximumClinicalTrialPhase`, `hasBeenWithdrawn` or `blackBoxWarning`** — they are `drugAndClinicalCandidates`, `maximumClinicalStage` (a string, mapped by `CLINICAL_STAGE_TO_PHASE`) and entries in `drugWarnings`. Introspect before changing a query; the API answers 400, not a partial result. |
| §3.1 target resolution | `identifier_service.resolve_target()` | One reviewed-human UniProt lookup yields accession + Ensembl gene ID + ChEMBL target ID from the entry's cross-references. TrEMBL is excluded — it returns several accessions per symbol and makes the xrefs ambiguous. |
| §3.2 disease resolution | `identifier_service.resolve_disease()` | MeSH Lookup → EFO. The spec assumes an internal MeSH→EFO table built by the BioKG loader; **no such table exists in this repo**, so the cross-walk runs against Open Targets' own `dbXRefs` (the same data EFO's ontology file would have been parsed for) and falls back to free-text disease search. Also returns the EFO descendant set. |
| §3.3 structure resolution | `identifier_service.resolve_structure()` | RDKit salt-strip → uncharge → canonical tautomer → InChIKey. Lazy import; without RDKit, name-only ligands cannot merge and each becomes its own row. |
| §4.1 criteria + weights | `criteria_config.py` | 20 scored rows (criterion 11 splits into 11a/11b), 19 criterion numbers, weights summing to 100, plus the unweighted exclusion filter and the display-only withdrawal year. Categorical lookup tables live here too. |
| §4.2 scoring | `scoring_service.py` | Min-max against the profile baseline, clipped, inverted for `LOWER_IS_BETTER`. The composite divides by the weight of the criteria that **had data** — a failed DailyMed parse drops out of both sums rather than scoring zero. A user weight of 0 is how "exclude this criterion" is expressed. |
| §5 Tool 1 | `profile_service.py` | Retrieval → extraction → InChIKey merge (Open Targets authoritative on conflict, displaced value kept in `secondary_values`) → DailyMed enrichment **post-dedup only** → baselines → exclusion setup. Ligands with no SPL are flagged `no label found`, never silently skipped. |
| §5 step 7 / §6 step 2 | `profile_service.build_exclusion()`, `candidate_service.apply_exclusion_filter()` | Matches over the EFO term **and its descendants**; excluded drugs are dropped before scoring, not scored-then-hidden. |
| §6 Tool 2 | `candidate_service.py` | Universe = ChEMBL molecules with any populated max phase, approved **and** investigational, ordered by max phase. Open Targets and DailyMed are one call *per candidate*, so the pool is pre-scored on ChEMBL fields and only `CURATEX_ENRICH_TOP_N` are enriched — the cap is reported in `metadata.enrichmentNote`, never silent. |
| §7 Tool 3 | `evidence_service.py` | PubMed per candidate, validated by requiring the drug **and** a target/disease term in the same record; unsupported candidates are flagged `score-only` and listed separately in the recommendation. Concurrency is capped at 3 — E-utilities 429s would otherwise degrade every candidate to `score-only`. |
| §8 output | `curatex_service._assemble()` | Candidate table + evidence links + recommendation + editable profile + coverage. `artifact` is explicitly `None`: table and text only, by design. |

Known gap to flag when reading results: **§6 step 1 defines the candidate universe with no
target-relevance term**, so a candidate need not bind the resolved target — ranking is
similarity to the ligand profile plus safety/regulatory quality. `search_candidate_universe`
takes `seed_records` if a target-scoped pool is ever wanted.

`tests/test_curatex.py` pins the weight table, the normalization/composite rules, the merge
precedence, the SPL parsing and the exclusion filter offline (no network, no LLM). Note
`tests/conftest.py` needs Databricks auth, so run it with `--noconftest` locally.

### NovSearch — the functional spec's three tools
`modules/novelty/` carries two surfaces. `api/v1/endpoints/novelty_search_agent.py` is the
legacy agent (`/agent/*`): SerpAPI → Google Patents, HTML scraping, HuggingFace embeddings,
Qdrant Cloud, one Gemini call, plus its own usage-tracking and history endpoints. It is kept
because it is a different contract. `novsearch_service.py` implements the NovSearch functional
specification on the platform's AWS/Databricks stack and is what the agent and
`/v1/agents/novsearch/*` run.

| Spec | Where | Note |
|---|---|---|
| §2 input shapes | `novsearch_service.normalize_input()` | Case A (`screensuite_carryover`) builds the query string from the resolved target/drug/disease and carries `candidate_id`/`docking_id` through. Case B (`user_direct`) uses the text as typed and runs the shared NER tagger only to isolate terms for title detection. Both converge on one normalized shape; the ids are **absent, not null-filled**, for a fresh query — and that absence is preserved all the way out (`response_model_exclude_none` on `/v1`). |
| §3 Tool 1 | `retrieval_service.py` | Title detection → Google Patents search (via SerpAPI) → BM25Okapi over title+abstract → RRF (k=60) of the API order and the BM25 order → title-pin similarity → MiniLM cross-encoder on Model Serving, min-maxed to 1-10. A cold or failing cross-encoder degrades to the RRF order rather than failing the assessment. `QueryCache` serves a repeat query from cache and fetches only the difference when more results are asked for. |
| §3 step 2 | `google_patents_service.py` | The spec names USPTO PatentsView (retired 2026-03-20); its USPTO Open Data Portal replacement was dropped too — ODP requires an ID.me-verified MyUSPTO account to issue a key, and one was never obtained, so NovSearch could never actually reach it. Retrieval now goes through **SerpAPI's `google_patents` engine** (same `SERPAPI_API_KEY` used elsewhere on the platform) for search — SerpAPI returns bibliographic fields (title, snippet, assignee, dates) directly in its own relevance order (`api_rank`). SerpAPI does **not** include claims text in search results, and a per-patent detail call for every indexed patent isn't worth metering, so full content (claims, background/summary/description) is scraped from `patents.google.com/patent/<id>/en` (public, unauthenticated) via `bs4`, with a regex fallback when the page's markup doesn't match the expected `itemprop`/class structure. `normalize_patent_id` / `display_patent_id` are near-identical here — Google's own ID form (`US10123456B2`) already **is** the display form, unlike ODP's bare-number storage key. HTML parsing is pinned offline against a fixture in `tests/test_novsearch.py`. |
| §4 Tool 2 | `indexing_service.py` | Dedup against the store **before** any fetch or embed → structured fetch → section-weighted chunking (independent claims 1.00 … background 0.70), claim chunks tagged by number → BGE-large-en-v1.5 (1024-dim, L2-normalised, `query: `/`passage: ` prefixes) on Model Serving → upsert into Databricks Vector Search → oldest-patent eviction past `NOVSEARCH_MAX_PATENTS`. A direct-access index, not delta-sync: indexing is driven by an API request, not a table write, so we supply the vectors. Row keys are a hash of `chunk_id`, so re-indexing overwrites rather than duplicating. |
| §5 Tool 3 | `synthesis_service.py` | One LLM call per report or QA turn. Retrieval is section-weighted, so an independent claim outranks an equally similar background paragraph. The report prompt requires the answer to address the §2 query specifically — not to summarise the patents — with every claim tied to a patent ID; it is split on `RECOMMENDED NEXT STEPS` into `agent_answer` / `recommendations`. QA modes follow `patent_ids`: one → `single_patent` (metadata questions answered from metadata chunks, claims questions prioritising claim-tagged chunks), several → `multiple_patent`, none → `multi_patent` over everything indexed. |
| §5 model | `databricks_clients.py` | SaulLM (or whatever's configured — Llama 3.3 70B on the deployed workspace) on Model Serving, falling back to Groq (`core/llm.py`, already used elsewhere on the platform) on any failure or rate limit. `model_used` reports which one **actually** answered, not which was configured. `databricks.vector_search` is a lazy import. |
| §6 supervisor object | `novsearch_service._assemble()` | `module` / `input_source` / `query` / `report` / `patents_table` / `session_state_update`, with `candidate_id` + `docking_id` only for a carry-over. |

Every model is self-hosted on Databricks Model Serving and reached through one
`/serving-endpoints/{name}/invocations` call, so there is a single auth path and timeout
policy. Set `DATABRICKS_HOST`, `DATABRICKS_TOKEN` and `PATENTSVIEW_API_KEY`; the endpoint and
index names default in `config.py` and are all overridable.

### ScreenSuite — the docking spec's resolution step + pipeline
`modules/screening/` is module 4. PyMOL and PLIP are **gone**: receptor prep is PDBFixer/OpenMM,
ligand and receptor conversion is the Open Babel **executable**, interaction profiling is ProLIF.

| Spec | Where | Note |
|---|---|---|
| §3 resolution | `resolution_service.py` | Protein names → RCSB full-text search (experimental structures preferred), falling back to a UniProt accession + AlphaFold model. Drug names → PubChem CIDs. Every resolved record carries **identifier + source**, because retrieval branches on source and a source-less identifier cannot be acted on. An identifier supplied by the caller is trusted and never re-looked-up. **ZINC is only honoured when supplied directly** — resolution never chooses it, since ZINC15's substance search is not a dependable lookup surface. |
| §3.4 confirmation | `resolution_service.resolve()` | Several matches → `stage="awaiting_structure_confirmation"` with a shortlist (identifier + source + title), nothing auto-selected. Mirrors CurateX's `AWAITING_TARGET` contract exactly. `allow_partial=True` still reports the unambiguous half. `confirm()` applies a choice and re-resolves. **Note: nothing in `supervisor/` branches on `stage` yet — that gap is shared with CurateX.** |
| §4 pipeline | `pipeline_service.screen_batch()` | Every protein × every drug in one submission. Ranking is **per protein, never pooled** — each receptor has its own box, so cross-receptor affinities are not comparable. Bounded by `SCREENING_MAX_COMBINATIONS` before any compute is spent. |
| §4.4 / §6 failures | `schemas.EntityFailure` | Download failure is a **hard gate**: the entity is excluded and reported, never treated as available downstream. A 200 response carrying HTML or under 200 bytes counts as a failure (RCSB/PubChem both do this for bad ids). Partial batches are the defined behaviour — successes plus a per-entity failure list, each tagged with its stage. |
| §6 interaction shape | `interaction_service._reshape_fingerprint()` | ProLIF's `to_dataframe()` is a MultiIndex frame keyed `(ligand, residue, interaction)`; returning those column labels puts a dataframe's shape on the wire instead of the findings. Output is grouped **by interaction type** with real rows, False cells dropped, ProLIF's cross-version type spellings normalised via `_TYPE_ALIASES`. |
| §6 collisions | `enums.safe_name()` / `run_directory()` | Output is run-scoped (`run_id`), not name-scoped. Two records sharing a name would otherwise overwrite each other's files **and** collide on the `<protein_name>.json` resume key, making one run return the other's results as a cache hit. |
| Docking box | `docking_service.compute_docking_box()` | **Not in the spec but it changes every number the spec ranks on.** The old code boxed the entire protein — blind docking, so poses are not site-specific and affinities are not comparable. The box now comes from the largest non-solvent HETATM group (the co-crystal ligand) plus padding, clamped to `SCREENING_BOX_MIN/MAX_SIZE`, taken from the **raw downloaded** structure because preparation strips exactly those records. Verified on 6VGL: a ~96× smaller search volume than blind. Apo/predicted models fall back to the whole-protein extent and say so in `box_mode`. |

#### External binaries — corrections to spec §8

The spec's §8 is partly out of date. What is actually true (verified 2026-09, win_amd64/py3.12):

| Spec claim | Reality |
|---|---|
| §8.1 "no pure-Python or pip-installable path to Vina exists on Windows — the Windows build is unsupported" | **A Windows build is published.** Every AutoDock Vina release carries `vina_<ver>_win.exe` alongside linux_x86_64, linux_aarch64, mac_x86_64 and mac_aarch64. `pip install vina` *is* still unavailable (it builds from source and needs Boost + a C++ toolchain), so the binary remains a download, not a dependency. |
| §8.2 "the Python bindings were a workaround for a binary-execution restriction; that direction is being reversed" | The restriction is real and still present: on a locked-down Windows workstation, endpoint security denies execution of both `vina.exe` and `obabel.exe` with `WinError 5 / Access is denied`, files present. So the bindings cannot simply be removed — they are the only conversion route that works there. |
| §8.2 "two external executables must be present" | **One** is genuinely mandatory (Vina). Open Babel ships as `openbabel-wheel`, giving both `obabel.exe` *and* working in-process bindings. |
| §8.4 "openmm is conda-only" (implied) | `openmm` 8.6.0 and `pdbfixer` 1.12.0 install from PyPI now. |

`docking_service.convert_to_pdbqt` therefore tries **three routes in order** and accumulates every
failure into one message: the `obabel` **executable** (what the platform images ship, and what §8.2
asks for) → Open Babel's **in-process binding** (same library, no execute permission needed) →
**Meeko** (ligands only). Nothing substitutes for Vina, so docking is the one stage that simply
stops when it is unavailable.

Binaries are resolved at call time — `SCREENING_VINA_PATH`/`SCREENING_OBABEL_PATH`, else `PATH`,
else `bin/` and a few repo-local spots — so a missing binary fails one run with a readable message
instead of breaking module import. `ExecutableBlockedError` distinguishes *present but refused by
policy* from *not installed*: reporting a blocked binary as "not found" sends people reinstalling a
dependency they already have. Every subprocess call is an argument list with
`subprocess.run(timeout=…)`; the old `timeout N obabel …` shell string was a POSIX-only builtin that
could not run on Windows at all.

`GET /api/v1/screening/health` probes each binary by **executing** it, not merely resolving it, and
reports `dockable`, `structure_preparation`, `interaction_profiling`, `blocked_by_policy` and a
one-line `blocker`. Check it before blaming a docking failure on the code.

#### Interaction profiling does not depend on Meeko metadata
`RDKitMolCreate.from_pdbqt_mol` rebuilds bond orders from the `REMARK SMILES` header — which **only
Meeko writes**. An obabel-prepared ligand yields a Vina pose with no such header, and Meeko then
returns an empty list, which silently broke profiling for exactly the preparation route §8.2
mandates. `_read_pose_molecules` tries Meeko first (best bond orders) and falls back to converting
the pose to SDF in-process for RDKit perception, retrying with valence checking relaxed if strict
sanitization rejects a geometry-perceived structure. A pose is never altered on the way in — no
added hydrogens, no recomputed charges.

#### Always lazy import
`pdbfixer`, `openmm`, `prolif`, `MDAnalysis` and `meeko` are optional. `api/v1/router.py` **skips
any router it cannot import**, so a module-level import of one of these makes the entire screening
surface vanish from the API on an environment that lacks it. Import them inside the function, and
give each an availability probe (`preparation_service._pdbfixer_available()`,
`interaction_service.prolif_available()`, `docking_service.tool_availability()`). Preparation
degrades to a documented text-level cleaner without PDBFixer (no added hydrogens — reported in the
result metadata, never passed off as a full prep); interaction profiling returns
`status="unavailable"` rather than raising.

### ScreenSuite on Databricks — how docking actually runs
Vina cannot run on the development laptops (endpoint security blocks it), so real docking runs in
the `drp_screensuite_job` notebook (`src/DRP_Main/jobs/screensuite_job.py`). Verified end to end on
**2026-09-15** on the `bpuvvada` target: JAK2 / 6VGL vs ruxolitinib, **−8.342 kcal/mol**, hinge
H-bond to LEU90 — the same contacts as the crystal pose. Run `run_2dbf0f6fdec5`, 272 s, written to
`drug_repository.gold.docking_results`.

| Fact | Why it matters |
|---|---|
| **Deploy to `bpuvvada` only** — profile `bpuvvada`, host `dbc-96626831-6c96` | The `dapi…` profiles point at the *old* workspace `dbc-9c7c244a-a883`. Check `databricks auth profiles` before any deploy. |
| Deploy with `DATABRICKS_BUNDLE_ENGINE=direct` | The default engine runs the Terraform binary, which endpoint security blocks here. |
| `bpuvvada` serverless is **x86_64, Python 3.10**; the old workspace was **aarch64, Python 3.11** | `vina` pip wheels exist for x86_64 only, so pip fails on ARM (falls back to source, dies on Boost). The notebook therefore downloads the official prebuilt binary matching `platform.machine()` — works on both. |
| Vina binary goes to **`/tmp`** | `/local_disk0` refuses writes on serverless. |
| The notebook imports the app from the **synced bundle files** (`sys.path` ← `<bundle root>/files/src`, derived from the notebook's own path) | No wheel path pinned to one user's folder; the job always runs the code deployed with it. |
| The app wheel's dependencies are **never** installed on the cluster | They include `databricks-connect`, which collides with the runtime's own PySpark. The notebook installs only what screening imports, one group per `%pip` cell — a combined `%pip` reports failure without naming the package. |
| **`langfuse==2.60.10` must be pinned** in the notebook | Unpinned installs pull langfuse 3.x, which removed `langfuse.decorators` → `ModuleNotFoundError` at import. |
| Egress to RCSB, PubChem, PyPI and GitHub works | Structure retrieval and the Vina download both happen at runtime. |
| Output defaults to `/tmp/screensuite_runs` (widget `output_root`) | Ephemeral — point it at a UC volume for anything that must survive the run. Gold rows survive either way. |

`pyproject.toml` excludes data files from the wheel (`[tool.hatch.build.targets.wheel] exclude`):
without it `biokg_data.zip` (186 MB) was packaged on every build.

#### The frontend path — `/v1/agents/screensuite/screen` → the job
Verified end to end from the live app on 2026-09-15: `202 {jobId}` → `completed` in 290 s →
`GET /v1/agents/screensuite/{jobId}/hits` returned `Ruxolitinib / 6VGL, −8.402 kcal/mol`. The `/v1`
request and response shapes are unchanged.

| Piece | Where | Note |
|---|---|---|
| Per-module opt-in | `DRP_DATABRICKS_MODULES` (config.py), `execution.should_delegate()` | A module listed here goes to its job even while `DRP_EXECUTION_BACKEND=inprocess`. **Keep the global switch `inprocess`**: flipping it also delegates TxKG, whose job notebook is still a placeholder, and would serve PLACEHOLDER rows to the frontend. |
| App config | **`app.yaml`** (repo root) | Set the opt-in here. `resources/drp_api_app.app.yml` mirrors it, but `bundle deploy` cannot update the app (the Apps API rejects its update mask), so only `app.yaml` via `databricks apps deploy` takes effect. |
| Permission | job ACL, granted by CLI | The app's service principal needs **CAN_MANAGE_RUN** on the ScreenSuite job, or `run_now` fails. Not defined in the bundle. |
| Runner | `runners._run_screensuite_on_databricks()` | Sends **only** the job's declared `query` parameter — `run_now` rejects undeclared ones, so `target`/`compounds` must not leak. A PDB-ID-shaped `target` ("6VGL") is sent as an identifier; a compound may carry `identifier`/`source`. No compounds → refused before any compute. |
| Results | Gold `docking_results` → `ScreeningHit` | `ligand`→`compound`, `affinity_kcal_mol`→`affinityKcalPerMol`, `pose_file`→`outputFile`. Rows whose `status` starts `Failed[` are reported as `failures`, not hits. |
| Ambiguity | notebook `resolution_failure_rows()` | A batch job has nobody to ask, so an ambiguous gene name (most are, on RCSB) is **written to Gold with its candidate list** instead of failing the run; the runner raises it as a readable error telling the user to resend with one PDB ID. Failure-row values are `""`, never `None` — an all-null column makes Spark's write crash. |

Deploying a change here takes both commands:

```bash
DATABRICKS_BUNDLE_ENGINE=direct databricks bundle deploy --target bpuvvada --profile bpuvvada
databricks apps deploy innodd-api \
  --source-code-path /Workspace/Users/bpuvvada@innominds.com/.bundle/DRP_Main/bpuvvada/files \
  --profile bpuvvada
```

`tests/test_screensuite_delegation.py` pins the opt-in, the job parameters, the Gold → wire mapping and
failure surfacing, offline.

### Screening queue — class-based singleton
`app/modules/screening/queue_service.py` uses a `QueueService` class with instance-level state
(`_queue`, `_is_paused`, `_current_task`, `_lock`). A module-level singleton
`queue_service = QueueService()` is imported by the router. Never use bare module-level globals for
queue state.

### Secrets — always via settings
All API keys and secrets are read from `.env` through `app/core/config.py` (`Settings` class,
pydantic-settings). Never hardcode keys in source. Access via
`from DRP_Main.app.core.config import settings; settings.GROQ_API_KEY`.

### Langfuse — version 2.x only
The `langfuse.decorators` submodule (`observe`, `langfuse_context`) is only available in
`langfuse<3`. The project pins to `langfuse==2.60.10`. Do not upgrade to langfuse 3.x — the
decorator API was removed.

---

## Adding a New Endpoint

1. Add the handler to the appropriate `src/DRP_Main/app/modules/<module>/router.py`
2. Add any new Pydantic models to `src/DRP_Main/app/modules/<module>/schemas.py`
3. Add business logic to the relevant `*_service.py` in the same module
4. Decorate service functions with `@observe(name="...")` from `langfuse.decorators`
5. No changes to `app/main.py` or `app/api/v1/router.py` needed

## Adding a New Module

1. Create `src/DRP_Main/app/modules/<name>/` with `__init__.py`, `router.py`, `schemas.py`, `*_service.py`
2. Add to `src/DRP_Main/app/api/v1/router.py`:
   ```python
   from DRP_Main.app.modules.<name>.router import router as <name>_router
   api_router.include_router(<name>_router, prefix="/<name>", tags=["N. <Name>"])
   ```

---

## Environment Variables (`.env` at repo root)

| Variable | Used by |
|---|---|
| `GROQ_API_KEY` | Literature scoring, Drug curation |
| `GROQ_MODEL` | Default: `llama-3.3-70b-versatile` |
| `GOOGLE_API_KEY` | Novelty search (Gemini) |
| `GEMINI_MODEL` | Default: `gemini-2.5-flash` |
| `SERPAPI_API_KEY` | Legacy novelty search, TxKG patent lookups, NovSearch patent retrieval (`google_patents_service.py`, SerpAPI's `google_patents` engine — replaced USPTO ODP, which needed an ID.me-verified MyUSPTO account never obtained) |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | NovSearch Model Serving (BGE-large, MiniLM, SaulLM) and Vector Search |
| `NOVSEARCH_EMBEDDING_ENDPOINT` / `NOVSEARCH_CROSS_ENCODER_ENDPOINT` / `NOVSEARCH_LLM_ENDPOINT` | Model Serving endpoint names |
| `NOVSEARCH_VS_ENDPOINT` / `NOVSEARCH_VS_INDEX` | Databricks Vector Search endpoint and direct-access index |
| `NOVSEARCH_MAX_PATENTS` | Patent cap in the vector store; oldest evicted past it (default 20) |
| `LANGFUSE_SECRET_KEY` | Observability tracing |
| `LANGFUSE_PUBLIC_KEY` | Observability tracing |
| `LANGFUSE_HOST` | Default: `https://us.cloud.langfuse.com` |
| `DATABASE_URL` | Main SQLite DB (default: `sqlite:///./data/agent_data.db`) |
| `AZURE_OPENAI_ENDPOINT` | Optional Azure OpenAI |
| `AZURE_OPENAI_API_KEY` | Optional Azure OpenAI |
| `SCREENING_VINA_PATH` / `SCREENING_OBABEL_PATH` | Paths to the two external executables. Empty searches `PATH`. Required for docking to run at all |
| `SCREENING_OUTPUT_ROOT` | Where run output is written. Empty keeps the in-repo `app/data` layout; point it at a UC volume on Databricks, where cluster storage is ephemeral |
| `SCREENING_BOX_MODE` | `site` (co-crystal-ligand box, default) or `protein` (blind docking over the whole structure) |
| `SCREENING_BOX_PADDING` / `SCREENING_BOX_MIN_SIZE` / `SCREENING_BOX_MAX_SIZE` | Å padding around a detected site, and the clamp on any box edge (defaults 8 / 20 / 40) |
| `SCREENING_NUM_MODES` / `SCREENING_EXHAUSTIVENESS` | Vina poses per ligand and search effort (defaults 9 / 8) |
| `SCREENING_TOP_FRACTION` | Top fraction of ligands kept per protein (default 0.05) |
| `SCREENING_MAX_COMBINATIONS` | Cost guard — a batch over this many protein × drug pairs is refused before compute (default 200) |
| `SCREENING_SUBPROCESS_TIMEOUT` | Per `vina`/`obabel` invocation, seconds (default 360) |
| `SCREENING_RESOLVE_TIMEOUT` / `SCREENING_RESOLVE_MAX_MATCHES` | RCSB/PubChem lookup timeout and confirmation-shortlist size |
| `BIOKG_DATA_DIR` | TxKG knowledge graph data path |
| `NCBI_API_KEY` | Optional — raises the PubMed/MeSH rate limit for TxKG novelty lookups and LitMineX retrieval |
| `LITMINEX_RETRIEVAL_LIMIT` | PMIDs pulled per query (default 100) |
| `LITMINEX_EXCLUDE_REVIEWS` | Optional publication-type filter — primary evidence only (default off) |
| `LITMINEX_FETCH_FULLTEXT` | Pull PMC open-access sections for positional weighting (default on) |
| `LITMINEX_NER_DISEASE_MODEL` / `LITMINEX_NER_GENE_MODEL` | scispaCy models for Tool 2 entity tagging |
| `LITMINEX_EMBEDDING_MODEL` / `LITMINEX_ENABLE_EMBEDDINGS` | SapBERT model for §4.5 semantic similarity |
| `LITMINEX_SUMMARY_TOP_N` / `LITMINEX_SUMMARY_MIN_SCORE` | Evidence selection for Tool 3 (defaults 10 / 50.0) |
| `CURATEX_REQUEST_TIMEOUT` / `CURATEX_MAX_RETRIES` | Per-call timeout and 429/5xx retry budget for the CurateX source clients |
| `CURATEX_MAX_KNOWN_LIGANDS` | ChEMBL molecules pulled per target for the profile baseline (default 300) |
| `CURATEX_UNIVERSE_LIMIT` / `CURATEX_MIN_CLINICAL_PHASE` | Tool 2 candidate pool size and maturity floor (1 keeps investigational compounds) |
| `CURATEX_ENRICH_TOP_N` / `CURATEX_EVIDENCE_TOP_N` | How many candidates get Open Targets + DailyMed extraction, and Tool 3 literature validation |
| `TXKG_RWR_RESTART` / `TXKG_RWR_LAMBDA` | RWRH restart and inter-layer jump probabilities (default 0.30 / 0.50) |
| `TXKG_HIDDEN_THRESHOLD` | Minimum corrected score (-log10 p) for a Hidden candidate (default 3.0) |
| `TXKG_ANNOTATION_EDGES` | Which `biokg.properties.*` annotation edges join the graph (see the TxKG discovery section) |
| `DRP_SEED_DEMO_USER` | Seed a login-able `/v1` account on startup (disable in prod) |
| `DRP_DEMO_EMAIL` / `DRP_DEMO_PASSWORD` | Credentials for that seeded account |
| `DRP_EXPORT_DIR` / `DRP_UPLOAD_DIR` | Where `/v1` exports and uploads are written |
| `DRP_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh-token lifetime for `/v1/auth/refresh` |

Copy `.env.example` to `.env` and fill in values before running.

---

## Data Directories

All runtime data lives under `data/` (gitignored):

| Path | Contents |
|---|---|
| `data/processed_proteins_output/<name>/` | Per-protein docking pipeline output |
| `data/drug_structures/` | Downloaded ligand SDF/PDBQT files |
| `data/protein_structures/` | Downloaded receptor PDB files |
| `data/vina_output/` | AutoDock Vina result PDBQT files |
| `data/mesh_cache.db` | SQLite cache for MeSH term lookups |
| `data/noveltysearch.db` | SQLite for novelty search session storage |
| `data/agent_data.db` | SQLite for auth / items |

Static input files (not generated) live in `src/DRP_Main/app/sample_ip_files/`:
- `literature_protein_and_drug_data.xlsx`
- `relevance_score_max_counts.json`
- `greek_symbols.json`
- `literature_search_protein_history.json`

---

## File Locations for Common Tasks

| Task | File |
|---|---|
| Edit the LitMineX chain / supervisor object (§7) | `src/DRP_Main/app/modules/literature/litminex_service.py` |
| Edit LitMineX Tool 1 (expansion + retrieval) | `src/DRP_Main/app/modules/literature/retrieval_service.py` |
| Edit LitMineX Tool 2 (relevance scoring) | `src/DRP_Main/app/modules/literature/relevance_service.py` |
| Change relation phrases / positional or score weights | `src/DRP_Main/app/modules/literature/relation_config.py` |
| Edit LitMineX Tool 3 (summarisation + QA) | `src/DRP_Main/app/modules/literature/summarization_service.py` |
| Edit biomedical NER (scispaCy + fallback) | `src/DRP_Main/app/modules/literature/ner_service.py` |
| Add/edit legacy keyword-search logic | `src/DRP_Main/app/modules/literature/scoring_service.py` |
| Add/edit MeSH term expansion | `src/DRP_Main/app/modules/literature/mesh_service.py` |
| Edit the CurateX chain / supervisor object (§8) | `src/DRP_Main/app/modules/drug_curation/curatex_service.py` |
| Edit CurateX Tool 1 (ligand retrieval, merge, profile) | `src/DRP_Main/app/modules/drug_curation/profile_service.py` |
| Edit CurateX Tool 2 (universe, exclusion, scoring) | `src/DRP_Main/app/modules/drug_curation/candidate_service.py` |
| Edit CurateX Tool 3 (evidence, recommendation) | `src/DRP_Main/app/modules/drug_curation/evidence_service.py` |
| Change criteria, default weights or categorical score maps | `src/DRP_Main/app/modules/drug_curation/criteria_config.py` |
| Change normalization / composite scoring | `src/DRP_Main/app/modules/drug_curation/scoring_service.py` |
| Add/edit a source API client (ChEMBL, Open Targets, …) | `src/DRP_Main/app/modules/drug_curation/ligand_sources.py` |
| Add/edit DailyMed SPL parsing | `src/DRP_Main/app/modules/drug_curation/dailymed_service.py` |
| Add/edit identifier resolution (UniProt / MeSH-EFO / InChIKey) | `src/DRP_Main/app/modules/drug_curation/identifier_service.py` |
| Add/edit legacy Groq compound generation | `src/DRP_Main/app/modules/drug_curation/curation_service.py` |
| Add/edit docking pipeline / batch entry point | `src/DRP_Main/app/modules/screening/pipeline_service.py` |
| Edit ScreenSuite name/ID resolution + confirmation checkpoint | `src/DRP_Main/app/modules/screening/resolution_service.py` |
| Change the docking box, Vina/Open Babel invocation, or ranking | `src/DRP_Main/app/modules/screening/docking_service.py` |
| Change the interaction-report shape (by interaction type) | `src/DRP_Main/app/modules/screening/interaction_service.py` |
| Change receptor preparation (PDBFixer / fallback cleaner) | `src/DRP_Main/app/modules/screening/preparation_service.py` |
| Change structure retrieval / the download gate | `src/DRP_Main/app/modules/screening/download_service.py` |
| Edit queue behavior | `src/DRP_Main/app/modules/screening/queue_service.py` |
| Edit TxKG spec logic (RWRH, degree correction, Known/Hidden, sourcing gate) | `src/DRP_Main/app/modules/txkg/discovery_service.py` |
| Add/edit a TxKG discovery endpoint | `src/DRP_Main/app/modules/txkg/discovery_router.py` |
| Edit legacy TxKG context-graph logic / data loading | `src/DRP_Main/app/api/v1/endpoints/txkg_test.py` |
| Edit the NovSearch chain / supervisor object (§2, §6) | `src/DRP_Main/app/modules/novelty/novsearch_service.py` |
| Edit NovSearch Tool 1 (search, BM25, RRF, rerank, cache) | `src/DRP_Main/app/modules/novelty/retrieval_service.py` |
| Edit NovSearch Tool 2 (dedup, chunking, embed, vector store) | `src/DRP_Main/app/modules/novelty/indexing_service.py` |
| Edit NovSearch Tool 3 (report synthesis + QA modes) | `src/DRP_Main/app/modules/novelty/synthesis_service.py` |
| Edit the Google Patents client (SerpAPI search + scraped full-text/claims parsing) | `src/DRP_Main/app/modules/novelty/google_patents_service.py` |
| Edit Model Serving clients (embeddings, cross-encoder, SaulLM/Groq fallback) | `src/DRP_Main/app/modules/novelty/databricks_clients.py` |
| Edit legacy novelty search logic (Google Patents / Gemini) | `src/DRP_Main/app/api/v1/endpoints/novelty_search_agent.py` |
| Change data directory paths | `src/DRP_Main/app/modules/screening/enums.py` |
| Change shared file utilities | `src/DRP_Main/app/shared/file_utils.py` |
| Add global config/env var | `src/DRP_Main/app/core/config.py` |
| Add/change a `/v1` endpoint | `src/DRP_Main/app/drp/routers/<tag>.py` + `drp/schemas.py` |
| Wire a `/v1` agent onto pipeline logic | `src/DRP_Main/app/drp/runners.py` |
| Change composer query → module routing | `src/DRP_Main/app/drp/dispatch.py` |

---

## Logging

Logs write to `src/DRP_Main/app/logs/api.log`. Root-level `*.log` files and `nohup.out` are runtime
artifacts from uvicorn — they are gitignored.

---

## Tests

```bash
uv run pytest tests/
# or, with the package importable:
cd src && python -m pytest ../tests/
```

Test files are in `tests/`. Sample protein names for API testing: `tests/sample_proteins.csv`.

---

## Git Workflow

- Main branch: `main`
- Never commit `.env`, `data/`, `*.log`, `*.db` — all are gitignored
- Never commit the `.venv/` or `.databricks/` directories
