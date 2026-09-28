"""
Job runners for the DRP `/v1` agent endpoints.

Each runner adapts an existing pipeline module to the DRP contract. Every import
of a pipeline module is **lazy** (inside the runner body): TxKG needs the BioKG
dataset, ScreenSuite needs PyMOL/Vina and NovSearch needs Gemini + a vector
store, none of which are guaranteed in every environment. A missing dependency
therefore fails one job with a readable message instead of breaking import of the
whole API.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.drp import execution, gold
from DRP_Main.app.drp.jobs import JobContext, JobSession, register
from DRP_Main.app.drp.models import DrpArticle
from DRP_Main.app.shared import uc_store
from DRP_Main.app.shared.volume_sync import ensure_local_dataset


def _ensure_biokg() -> None:
    """
    Make the BioKG dataset locally readable before anything imports TxKG.

    On Databricks Apps `BIOKG_DATA_DIR` points at a Unity Catalog volume, which
    the container cannot read as a filesystem; this mirrors it to local disk and
    rewrites the env var. A no-op everywhere else, and after the first call.
    Deliberately invoked before the lazy `txkg_test` / `discovery_service`
    imports, because those read the env var at module import time.

    Blocking: prefer `_load_txkg()` from inside a runner — see its docstring.
    """
    ensure_local_dataset("BIOKG_DATA_DIR", "biokg_data")


async def _load_txkg():
    """
    Mirror the BioKG dataset and import the TxKG module without stalling the API.

    Runners are coroutines on the server's own event loop, so anything blocking
    they call freezes every other request in the process — not just their own.
    The first TxKG run is the worst case: mirroring ~340 MB out of the Unity
    Catalog volume and building the graph indexes takes minutes, during which the
    whole API stops answering, health checks included.

    Both steps therefore run in a worker thread. They are also cached after the
    first call — `ensure_local_dataset` is a no-op once mirrored, and the import
    is a dict lookup — so later runs pay almost nothing for going through here.
    """
    def _prepare():
        _ensure_biokg()
        from DRP_Main.app.modules.txkg import discovery_service as ds

        return ds

    return await asyncio.to_thread(_prepare)


logger = get_logger(__name__)


# ── Unity Catalog persistence (bronze/silver/gold) ─────────────────────────────
# Best-effort side effects: a write failure here must never fail the job, since the
# job's in-memory/API result is already valid by the time these are called.

def _persist_agent_invocation(ctx: JobContext, agent_name: str, recommendation_text: str = "") -> None:
    """Record one row in orchestration.agent_invocation for this job."""
    try:
        uc_store.insert_rows(
            "orchestration",
            "agent_invocation",
            [
                {
                    "invocation_id": ctx.job_id,
                    "session_id": ctx.session_id,
                    "config_id": None,
                    "agent_name": agent_name,
                    "invoked_ts": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": None,
                    "recommendation_text": recommendation_text[:4000] if recommendation_text else None,
                    "status": "completed",
                }
            ],
        )
    except Exception:
        logger.warning("uc_store: failed to persist agent_invocation for job %s", ctx.job_id, exc_info=True)


def _persist_txkg_targets(
    ctx: JobContext, candidates: List[Dict[str, Any]], targets: List[Dict[str, Any]]
) -> None:
    """Persist TxKG's ranked candidates to reference.target + bronze.kg_target."""
    try:
        target_rows = [
            {
                "target_id": c["uniprot_id"],
                "uniprot_id": c["uniprot_id"],
                "target_name": c.get("name") or c["uniprot_id"],
                "organism": None,
                "created_ts": datetime.now(timezone.utc).isoformat(),
            }
            for c in candidates
            if c.get("uniprot_id")
        ]
        uc_store.upsert_rows("reference", "target", ["target_id"], target_rows)

        _persist_agent_invocation(ctx, "TxKG")

        kg_target_rows = [
            {
                "kg_target_id": str(uuid.uuid4()),
                "invocation_id": ctx.job_id,
                "target_id": c["uniprot_id"],
                "score": t.get("correctedScore"),
                "recommendations": t.get("category"),
                "relationship_type": t.get("sourcingStatus"),
                "edge_weight": t.get("score"),
            }
            for c, t in zip(candidates, targets)
            if c.get("uniprot_id")
        ]
        uc_store.insert_rows("bronze", "kg_target", kg_target_rows)
    except Exception:
        logger.warning("uc_store: failed to persist TxKG results for job %s", ctx.job_id, exc_info=True)


class RunnerError(RuntimeError):
    """A runner failed for a reportable reason (bad input, module unavailable).

    `user_facing` tells the job layer to surface the message as written, without
    the exception class name in front of it — these messages are advice for the
    researcher, not diagnostics.
    """

    user_facing = True


# ── Shared helpers ────────────────────────────────────────────────────────────
_STOP_PHRASES = (
    "find protein targets associated with",
    "find protein targets for",
    "find targets associated with",
    "find targets for",
    "identify targets for",
    "protein targets for",
    "targets for",
    "for drug repurposing",
    "drug repurposing",
    "knowledge graph for",
    "build a subgraph for",
    "subgraph for",
    "assess novelty for",
    "novelty for",
    "mine literature for",
    "literature for",
    "screen compounds for",
    "curate compounds for",
    # Instruction phrasings that reached UniProt verbatim as a protein name, e.g.
    # "@curatex create drug profile for JAK2" was looked up whole and failed with
    # "No reviewed human UniProt entry matched 'Create drug profile for JAK2'".
    "create a drug profile for",
    "create drug profile for",
    "generate a drug profile for",
    "generate drug profile for",
    "build a drug profile for",
    "build drug profile for",
    "drug profile for",
    "target product profile for",
    "create a profile for",
    "create profile for",
    "profile for",
    "curate drugs for",
    "curate for",
    "find compounds for",
    "find drugs for",
    "dock compounds against",
    "dock against",
)


_MENTION = re.compile(r"@[A-Za-z][\w-]*")


def _strip_query_noise(query: str) -> str:
    """Reduce a natural-language composer query to its disease/entity phrase."""
    text = _MENTION.sub("", (query or "")).strip().rstrip("?.!")
    lowered = text.lower()
    for phrase in _STOP_PHRASES:
        idx = lowered.find(phrase)
        if idx == -1:
            continue
        text = (text[:idx] + " " + text[idx + len(phrase) :]).strip()
        lowered = text.lower()
    return " ".join(text.split()).strip(" ,-")


_DISEASE_STOPWORDS = {"disease", "diseases", "disorder", "disorders", "syndrome", "mellitus"}

#: Everyday disease words mapped onto the vocabulary BioKG actually uses (MeSH).
#: Without this, "breast cancer" cannot reach "Breast Neoplasms" — the word
#: "cancer" appears nowhere in that name — and settles for whatever name does
#: contain both words, e.g. "Hereditary Breast and Ovarian Cancer Syndrome".
_DISEASE_SYNONYMS = {
    "cancer": "neoplasms",
    "cancers": "neoplasms",
    "carcinoma": "neoplasms",
    "tumor": "neoplasms",
    "tumour": "neoplasms",
    "tumors": "neoplasms",
    "tumours": "neoplasms",
    "stroke": "infarction",
    "heartburn": "pyrosis",
    "kidney": "renal",
    # MeSH uses the adjectival form in several organ names.
    "prostate": "prostatic",
    "breast": "breast",
    "liver": "hepatic",
    "stomach": "gastric",
}


def _match_disease_by_tokens(phrase: str) -> Optional[Tuple[str, str]]:
    """
    Match on token containment before falling back to fuzzy matching.

    BioKG names diseases in MeSH inverted form ('Diabetes Mellitus, Type 2'), so a
    user phrase like 'Type 2 Diabetes' scores poorly under `difflib` and can lose
    to an unrelated short name. Requiring every significant token to appear resolves
    that correctly.

    Two rules exist to stop a wrong disease being chosen silently — the failure
    that surfaced as "Propagating over the full knowledge graph from Colic..." for
    a query about something else:

    * **Whole words, not substrings.** `"type" in name` also matches "pheno*type*"
      and `"colic" in name` matches "*colic*ky", so a token could be satisfied by an
      unrelated word it happens to sit inside.
    * **Best overlap, not shortest name.** Preferring the shortest match hands the
      result to the most generic disease that happens to contain the tokens. A name
      is now ranked by how much of it the query actually accounts for, so
      "Colic" only wins when the query really is about colic.
    """
    _ensure_biokg()
    from DRP_Main.app.api.v1.endpoints.txkg_test import (
        disease_name_to_id,
        diseases_in_kg,
        id_to_name,
    )

    tokens = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if t not in _DISEASE_STOPWORDS]
    if not tokens:
        return None

    # Try the words as typed first, then their MeSH equivalents, so an exact
    # colloquial match still wins over a translated one where both exist.
    token_sets = [set(tokens)]
    translated = {_DISEASE_SYNONYMS.get(t, t) for t in tokens}
    if translated != set(tokens):
        token_sets.append(translated)

    # Both passes are scored and the better match wins; stopping at the first pass
    # that matched anything would keep "breast cancer" on "Hereditary Breast and
    # Ovarian Cancer Syndrome" (coverage 0.5) instead of reaching the translated
    # "Breast Neoplasms" (coverage 1.0). `rank` breaks ties toward the words as typed.
    best: Optional[Tuple[float, int, int, str, str]] = None
    for rank, token_set in enumerate(token_sets):
        for name, disease_id in disease_name_to_id.items():
            if disease_id not in diseases_in_kg:
                continue
            name_tokens = {
                t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in _DISEASE_STOPWORDS
            }
            if not name_tokens or not token_set.issubset(name_tokens):
                continue
            # Share of the matched name the query accounts for: 1.0 when the query
            # names the disease exactly, lower when the name carries extra qualifiers.
            coverage = len(token_set & name_tokens) / len(name_tokens)
            candidate = (-coverage, rank, len(name), name, disease_id)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    return best[4], id_to_name.get(best[4], best[3])


def _resolve_disease(query: str) -> Tuple[str, str]:
    """Map free text onto a BioKG disease, trying the cleaned phrase first."""
    _ensure_biokg()
    from DRP_Main.app.modules.txkg.service import find_disease

    candidates = [c for c in (_strip_query_noise(query), (query or "").strip()) if c]
    for candidate in candidates:
        matched = _match_disease_by_tokens(candidate)
        if matched:
            return matched
    for candidate in candidates:
        disease_id, disease_name = find_disease(candidate)
        if disease_id:
            return disease_id, disease_name
    raise RunnerError(
        f"No disease in the knowledge graph matched '{query}'. "
        "Try a disease name as it appears in /v1/agents/txkg (e.g. 'Type 2 Diabetes')."
    )


_SPEC_NODE_TYPES = {
    "disease": "Comorbidity",
    "gene/protein": "Protein",
    "protein": "Protein",
    "pathway": "Pathway",
    "biological_process": "Pathway",
    "molecular_function": "Pathway",
    "cellular_component": "Pathway",
    "complex": "Pathway",
    "tissue": "Pathway",
    "cell": "Pathway",
    "genetic_disorder": "Comorbidity",
    "disease_category": "Comorbidity",
    "drug": "Compound",
    "compound": "Compound",
}

GRAPH_LEGEND = {
    "Disease Hub": "#0225AA",
    "Protein": "#1E88E5",
    "Pathway": "#065B52",
    "Compound": "#00897B",
    "Comorbidity": "#E61919",
}


def _spec_node_type(raw_type: str, is_disease_hub: bool) -> str:
    if is_disease_hub:
        return "Disease Hub"
    return _SPEC_NODE_TYPES.get((raw_type or "").lower(), "Pathway")


def _to_spec_graph(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a txkg subgraph payload into the spec's nodes/edges/legend shape."""
    # `id` stays the accession — it is the node's identity and the edges key on
    # it — but `label` is what the graph draws, so a KEGG or Reactome id there
    # showed "hsa04630" where "JAK-STAT signaling pathway" belongs. BioKG's own
    # "name" falls back to the accession rather than to nothing, so it cannot be
    # trusted as "already resolved" — `resolve` decides that.
    from DRP_Main.app.modules.txkg.node_labels import resolve as resolve_label

    nodes = [
        {
            "id": node["id"],
            "label": resolve_label(node["id"], node.get("name")),
            "type": _spec_node_type(node.get("type", ""), bool(node.get("is_disease"))),
        }
        for node in subgraph.get("nodes", [])
    ]
    edges = [
        {"source": link["source"], "target": link["target"], "label": link.get("label", "")}
        for link in subgraph.get("links", [])
    ]
    return {"nodes": nodes, "edges": edges, "legend": GRAPH_LEGEND}


def _spec_targets(raw_targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "uniprotId": t.get("uniprot_id", ""),
            "name": t.get("name", ""),
            "score": float(t.get("score", 0) or 0),
            "geneName": t.get("gene_name"),
            "fullName": t.get("full_name"),
            "hopDistance": t.get("hop_distance"),
            "connectionTypes": t.get("connection_types", []) or [],
            "fromKnowledgeGraph": bool(t.get("from_knowledge_graph", True)),
            "customAdded": bool(t.get("custom_added", False)),
        }
        for t in raw_targets
    ]


# ── TxKG ──────────────────────────────────────────────────────────────────────
def _readable_path(path: Dict[str, Any]) -> str:
    """
    Render one connecting path with the entities it actually traverses.

    `metapath` is the *type* template — "disease→gene/protein→biological_process" —
    which tells a researcher nothing about their disease. `_describe_path` already
    records `node_names` and `edge_labels` alongside it, so the readable form costs
    nothing to produce:

        Thrombocytosis —Associated With→ JAK2 —Participates In→ megakaryocyte differentiation
    """
    names = path.get("node_names") or path.get("nodes") or []
    labels = path.get("edge_labels") or []
    if not names:
        return ""
    out = [str(names[0])]
    for i, name in enumerate(names[1:]):
        label = labels[i] if i < len(labels) else ""
        out.append(f"—{label}→ {name}" if label else f"→ {name}")
    return " ".join(out)


def _discovery_targets(candidates: List[Dict[str, Any]],
                       disease_name: str = "") -> List[Dict[str, Any]]:
    """
    Map the discovery service's candidate records onto the wire `Target` shape.

    `score` is normalised to 0-100 so the UI can present it as a percentage. The
    underlying corrected score is `-log10 p` from the degree-corrected random walk,
    which is unbounded — values above 100 were reaching the UI and reading as a
    broken percentage. Normalisation is relative to the strongest candidate in the
    same run, so the top target is 100 and everything else is its share of that;
    ranking is unchanged because the transform is monotonic.

    The raw value stays available as `correctedScore` for anyone who needs the
    absolute figure.
    """
    raw_scores = [float(c.get("corrected_score") or 0.0) for c in candidates]
    top = max(raw_scores, default=0.0)

    def _percent(value: float) -> float:
        # A run with no signal at all must not divide by zero.
        return round(100.0 * value / top, 2) if top > 0 else 0.0

    return [
        {
            "uniprotId": c["uniprot_id"],
            "name": c.get("name") or c["uniprot_id"],
            # 0-100 relative relevance. `hopDistance` is deliberately absent:
            # nothing is hop-bounded.
            "score": _percent(float(c.get("corrected_score") or 0.0)),
            "correctedScore": float(c.get("corrected_score") or 0.0),
            "geneName": c.get("gene_name"),
            "fullName": c.get("full_name"),
            "hopDistance": None,
            "connectionTypes": sorted(
                {p["metapath"] for p in c.get("paths", []) if p.get("metapath")}
            ),
            # The same paths with real entity names rather than node types, so the
            # UI can show "Thrombocytosis —Associated With→ JAK2" instead of
            # "disease→gene/protein". Shortest, best-sourced paths come first
            # (discovery_service sorts them), so the first few are the useful ones.
            "connectionPaths": [
                readable
                for readable in (_readable_path(p) for p in c.get("paths", [])[:5])
                if readable
            ],
            "fromKnowledgeGraph": True,
            "customAdded": False,
            "category": c.get("category"),
            "confirmed": c.get("confirmed", True),
            "sourcingStatus": c.get("sourcing_status"),
            "noveltyLabel": c.get("novelty_label", "Unknown"),
            "literatureHits": c.get("literature_hits"),
            "patentHits": c.get("patent_hits"),
            # The curated databases backing this candidate's connecting path (§5), so the
            # insights "Sources" tab can cite them.
            "supportingSources": c.get("supporting_sources")
            or sorted(
                {
                    src
                    for path in c.get("paths", [])
                    for src in (path.get("edge_sources") or [])
                    if src
                }
            ),
            # Same sources, resolved to the record that backs the claim so the
            # Sources panel can link to it instead of a database home page.
            "supportingSourceLinks": _source_links(
                c.get("supporting_sources")
                or sorted(
                    {
                        src
                        for path in c.get("paths", [])
                        for src in (path.get("edge_sources") or [])
                        if src
                    }
                ),
                c["uniprot_id"],
                c.get("gene_name") or "",
                disease_name,
            ),
        }
        for c in candidates
    ]


#: Source phrase → a deep link into that database for a specific protein.
#: `supportingSources` carries prose like "IntAct curated protein interactions",
#: which is all the UI had to work with, so every "Sources" link could only open
#: a database home page. These resolve to the record that actually backs the
#: claim. Matched on a keyword because one phrase names several databases
#: ("Reactome / KEGG / SMPDB pathway membership").
_SOURCE_LINKS: List[Tuple[str, str, str]] = [
    ("ctd",      "CTD",      "https://ctdbase.org/detail.go?type=gene&acc={gene}"),
    ("medgen",   "MedGen",   "https://www.ncbi.nlm.nih.gov/medgen/?term={disease}"),
    ("intact",   "IntAct",   "https://www.ebi.ac.uk/intact/search?query={uniprot}"),
    ("reactome", "Reactome", "https://reactome.org/content/query?q={uniprot}&species=Homo+sapiens"),
    ("kegg",     "KEGG",     "https://www.genome.jp/dbget-bin/www_bfind_sub?dbkey=genes&keywords={gene}"),
    ("smpdb",    "SMPDB",    "https://smpdb.ca/search?query={uniprot}"),
    ("uniprot",  "UniProt",  "https://www.uniprot.org/uniprotkb/{uniprot}/entry"),
    ("go ",      "QuickGO",  "https://www.ebi.ac.uk/QuickGO/annotations?geneProductId={uniprot}"),
    ("disgenet", "DisGeNET", "https://www.disgenet.org/search?q={gene}"),
]


def _source_links(sources: Sequence[str], uniprot: str, gene: str,
                  disease: str) -> List[Dict[str, str]]:
    """Resolve source phrases to `{name, url}` records for the Sources panel."""
    out: List[Dict[str, str]] = []
    seen = set()
    for phrase in sources or []:
        low = (phrase or "").lower()
        for needle, name, template in _SOURCE_LINKS:
            if needle not in low or name in seen:
                continue
            # A template needs the identifier it interpolates; without one the
            # link would land on a search page for an empty string, which is the
            # home-page behaviour this replaces.
            if "{uniprot}" in template and not uniprot:
                continue
            if "{gene}" in template and not gene:
                continue
            if "{disease}" in template and not disease:
                continue
            seen.add(name)
            out.append({
                "name": name,
                "url": template.format(
                    uniprot=quote(uniprot or ""),
                    gene=quote(gene or ""),
                    disease=quote(disease or ""),
                ),
                "describes": phrase,
            })
    return out


#: "'jak2' matches several structures — resend with one identifier. Candidates:
#: 7F7W (JAK2-JH2), 8C09 (Crystal structure of JAK2 JH2-I559F), ..."
_AMBIGUOUS_RE = re.compile(r"matches several structures.*?Candidates:\s*(.+)$", re.S)
_CANDIDATE_RE = re.compile(r"\b([0-9][A-Za-z0-9]{3})\s*\(([^)]*)\)")


def _structure_choices(failures: Sequence[str]) -> List[Dict[str, str]]:
    """
    PDB entries offered by an ambiguous-structure failure, as `{id, title}`.

    Parsed from the message rather than plumbed through as data because the
    resolution step runs inside the Databricks job and only its status string
    crosses back.
    """
    out: List[Dict[str, str]] = []
    seen = set()
    for failure in failures or []:
        match = _AMBIGUOUS_RE.search(str(failure))
        if not match:
            continue
        for pdb_id, title in _CANDIDATE_RE.findall(match.group(1)):
            key = pdb_id.upper()
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": key, "title": title.strip()})
    return out


async def _run_txkg_on_databricks(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Run TxKG as a Databricks job and rebuild the contract response from Gold.

    The notebook writes one `kg_targets` row per ranked candidate and one
    `kg_relationships` row per connecting edge, both tagged with this job's id.
    Everything the API returns is derived from those rows, so the response is
    identical in shape to the in-process path — see `run_txkg_query`.
    """
    query = params.get("query", "")
    limit = int(params.get("limit", 10))

    await execution.run_module("TxKG", params, ctx)

    ctx.progress("Reading ranked targets from the knowledge graph output...")
    rows = gold.read_job_rows("kg_targets", ctx.job_id)
    if not rows:
        raise RunnerError(
            f"The TxKG job completed but wrote no targets for '{query}'. "
            "The disease may not be present in the knowledge graph."
        )

    rows.sort(key=lambda r: gold.as_float(r, "score", 0.0) or 0.0, reverse=True)
    # Same 0-100 normalisation as the in-process path, so both backends agree.
    top_raw = max((gold.as_float(r, "score", 0.0) or 0.0) for r in rows) if rows else 0.0
    targets = [
        {
            "uniprotId": row.get("uniprot_id") or "",
            "name": row.get("name") or row.get("uniprot_id") or "",
            "score": round(100.0 * (gold.as_float(row, "score", 0.0) or 0.0) / top_raw, 2)
            if top_raw > 0 else 0.0,
            "correctedScore": gold.as_float(row, "corrected_score")
            or gold.as_float(row, "score", 0.0),
            "geneName": row.get("gene_name"),
            "fullName": row.get("full_name"),
            "hopDistance": gold.as_int(row, "hop_distance"),
            "connectionTypes": gold.decode(row, "connection_types", []) or [],
            "fromKnowledgeGraph": True,
            "customAdded": False,
            "category": row.get("category"),
            "confirmed": (row.get("confirmed") or "").lower() in ("true", "1", "yes"),
            "sourcingStatus": row.get("sourcing_status"),
            "noveltyLabel": row.get("novelty_label") or "Unknown",
            "supportingSources": gold.decode(row, "supporting_sources", []) or [],
        }
        for row in rows[:limit]
    ]

    # The disease and the LLM interpretation are per-run, not per-target: the
    # notebook repeats them on every row, so any row carries them.
    first = rows[0]
    disease_name = first.get("disease") or query
    known = sum(1 for t in targets if t["category"] == "Known/Direct")
    hidden = len(targets) - known

    _persist_agent_invocation(ctx, "TxKG")

    return {
        "disease": disease_name,
        "diseaseId": first.get("disease_id") or "",
        "count": len(targets),
        "targets": targets,
        "interpretation": first.get("interpretation") or "",
        "knowledgeGraphTargets": len(targets),
        "customTargets": 0,
        "hopDistribution": {},
        "maxHopsSearched": None,
        "recommendation": gold.decode(first, "recommendation", {}) or {},
        "method": gold.decode(first, "method", {}) or {},
        "counts": {"known": known, "hidden": hidden},
        "summary": f"{known} known and {hidden} hidden target(s) identified for {disease_name}",
    }


@register("txkg.query")
async def run_txkg_query(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Ranked target discovery for the disease named in a natural-language query.

    Runs the TxKG functional-spec chain (`modules/txkg/discovery_service`): full-graph
    RWRH propagation, degree correction, Known/Hidden categorization against the curated
    disease-association edge, the sourcing gate, per-candidate path reconstruction,
    novelty confirmation and a grounded interpretation.

    Delegates to the TxKG Databricks job instead when this deployment is configured
    for it; both paths return the same shape, so nothing downstream changes.
    """
    if execution.should_delegate("TxKG"):
        return await _run_txkg_on_databricks(params, ctx)

    query = params.get("query", "")
    limit = int(params.get("limit", 10))

    ctx.progress("Loading the knowledge graph...")
    ds = await _load_txkg()

    ctx.progress("Resolving the disease against the knowledge graph...")
    # `resolve_disease` walks the graph's entity index synchronously.
    resolved = await asyncio.to_thread(ds.resolve_disease, query)
    if not resolved.ok:
        # Fall back to the runner's token-based resolver, which handles phrasings like
        # "Find protein targets associated with Type 2 Diabetes" that the graph's own
        # disease index does not contain verbatim.
        disease_id, disease_name = await asyncio.to_thread(_resolve_disease, query)
    else:
        disease_id, disease_name = resolved.id, resolved.name

    # Name both sides of the match. The old line reported only the resolved disease
    # ("Propagating over the full knowledge graph from Colic..."), so a researcher
    # who asked about something else had no way to see that the wrong disease had
    # been picked, or that the graph's name for it differs from theirs.
    asked = _strip_query_noise(query) or query
    if asked.strip().lower() != (disease_name or "").strip().lower():
        ctx.progress(f"Matched '{asked}' to '{disease_name}' in the knowledge graph")

    ctx.progress(f"Propagating over the full knowledge graph from {disease_name}...")
    payload = await ds.run_discovery(
        disease_name,
        top_known=limit,
        top_hidden=limit,
        interpret_top=min(5, limit),
        # Bind to the conversation session so a follow-up ("why is this one Hidden?")
        # is answered from these results instead of rerunning the chain.
        session_id=ctx.session_id or f"job:{ctx.job_id}",
    )
    if not payload.get("resolved"):
        raise RunnerError(payload.get("clarification") or f"Could not resolve '{query}'.")

    ctx.progress("Categorizing Known vs Hidden targets and checking sourcing...")
    known = payload["candidates"]["known"]
    hidden = payload["candidates"]["hidden"]
    ordered = sorted(known + hidden, key=lambda c: -c["corrected_score"])
    targets = _discovery_targets(ordered, disease_name)
    counts = payload["counts"]

    _persist_txkg_targets(ctx, ordered, targets)

    return {
        "disease": payload["disease"]["name"],
        "diseaseId": payload["disease"]["id"],
        "count": len(targets),
        "targets": targets,
        "interpretation": payload["interpretation"],
        "knowledgeGraphTargets": len(targets),
        "customTargets": 0,
        "hopDistribution": {},
        "maxHopsSearched": None,
        # Spec-specific additions, surfaced alongside the existing contract.
        "table": payload["table"],
        "recommendation": payload["recommendation"],
        "subgraph": _to_spec_graph(payload["subgraph"]),
        "subgraphHtmlUrl": payload["subgraph"].get("html_url"),
        "method": payload["method"],
        "counts": counts,
        "summary": (
            f"{counts['known']} known and {counts['hidden']} hidden target(s) identified "
            f"({counts['hidden_confirmed']} of the hidden confirmed by sourcing)"
        ),
    }


# ── Knowledge graph ───────────────────────────────────────────────────────────
@register("subgraph.generate")
async def run_subgraph_generate(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """Build a disease sub-graph and reduce it to the spec's node/edge model."""
    await _load_txkg()
    from DRP_Main.app.api.v1.endpoints.txkg_test import get_subgraph

    disease = params.get("disease") or ""
    ctx.progress(f"Generating knowledge sub-graph for {disease}...")
    disease_id, disease_name = await asyncio.to_thread(_resolve_disease, disease)

    payload = await get_subgraph(
        disease=disease_name,
        max_nodes=int(params.get("maxNodes", 100)),
        max_hops=int(params.get("maxHops", 3)),
    )
    subgraph = payload.get("subgraph", {}) or {}
    stats = subgraph.get("statistics", {}) or {}
    graph = _to_spec_graph(subgraph)

    selected = params.get("selectedTarget")
    pathway_connections = sum(
        count
        for node_type, count in (stats.get("entity_counts") or {}).items()
        if node_type in ("pathway", "biological_process", "molecular_function", "cellular_component")
    )

    ctx.progress("Summarising graph statistics...")
    return {
        "disease": disease_name,
        "diseaseId": disease_id,
        "selectedTarget": selected,
        "graph": graph,
        "stats": {
            "relationshipsFound": stats.get("total_edges", len(graph["edges"])),
            "drugCandidates": stats.get("drug_count", 0),
            "pathwayConnections": pathway_connections,
        },
        "htmlUrl": payload.get("html_url"),
        "topProteins": subgraph.get("top_proteins", []),
        "nativeStatistics": stats,
    }


@register("metapath.analyze")
async def run_metapath_analyze(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """Meta-path reasoning over the disease of a previously generated graph."""
    await _load_txkg()
    from DRP_Main.app.api.v1.endpoints.txkg_test import (
        CONTEXT_TYPE_WEIGHTS,
        get_metapaths_for_disease,
    )

    disease = params.get("disease") or ""
    if not disease:
        raise RunnerError("Meta-path analysis needs a source graph job with a resolved disease.")

    ctx.progress(f"Running meta-path analysis for {disease}...")
    payload = await get_metapaths_for_disease(
        disease=disease,
        limit=int(params.get("limit", 10)),
        max_hops=int(params.get("maxHops", 3)),
    )

    metapaths = payload.get("metapaths", []) or []
    summary = payload.get("summary", {}) or {}
    # A path carries parallel lists: nodes[i] --edges[i]--> nodes[i+1], plus the
    # BioKG type of each node in node_types.
    node_ids, edge_keys, pathway_nodes = set(), set(), set()
    for entry in metapaths:
        for path in entry.get("paths", []) or []:
            nodes = path.get("nodes", []) or []
            relations = path.get("edges", []) or []
            node_types = path.get("node_types", []) or []
            node_ids.update(nodes)
            for index, relation in enumerate(relations):
                if index + 1 < len(nodes):
                    edge_keys.add((nodes[index], nodes[index + 1], relation))
            for node, node_type in zip(nodes, node_types):
                if str(node_type).lower() == "pathway":
                    pathway_nodes.add(node)

    ctx.progress("Scoring target predictions...")
    scores = [
        {
            "target": entry.get("target_name", ""),
            "uniprotId": entry.get("target_id", ""),
            "score": float(entry.get("score", 0) or 0),
            "contextScore": float(entry.get("context_score", 0) or 0),
            "totalPaths": entry.get("total_paths", 0),
        }
        for entry in metapaths
    ]
    traversals = [
        {
            "target": entry.get("target_name", ""),
            "uniprotId": entry.get("target_id", ""),
            "totalPaths": entry.get("total_paths", 0),
            "pathsByHop": {str(k): v for k, v in (entry.get("paths_by_hop") or {}).items()},
            "paths": entry.get("paths", []) or [],
        }
        for entry in metapaths
    ]

    return {
        "disease": payload.get("disease", disease),
        "summary": {
            "paths": summary.get("total_paths", sum(s["totalPaths"] for s in scores)),
            "targets": len(metapaths),
            "pathways": len(pathway_nodes),
            "nodes": len(node_ids),
            "edges": len(edge_keys),
            "clusters": len({s["uniprotId"] for s in scores if s["totalPaths"]}),
        },
        "scores": scores,
        "traversals": traversals,
        # What the two numbers on a meta-path row actually mean. They were being
        # read off the screen with no stated definition, so "score 0.3" could not
        # be told apart from a probability or a percentage.
        "scoreDefinitions": {
            "score": "Aggregate meta-path support for this target: the sum of its "
                     "paths weighted by contextScore, so many weak PPI hops never "
                     "outrank a single pathway-mediated connection. Higher is "
                     "stronger support; it is not a probability and has no fixed "
                     "maximum.",
            "contextScore": "Biological plausibility of the intermediate node that "
                            "links disease to target, from 1.0 down to 0.2. A shared "
                            "pathway (1.0) is stronger evidence than a shared "
                            "biological process (0.9), which beats a bare "
                            "protein-protein interaction (0.3) — guilt by "
                            "association only.",
            "contextWeights": dict(CONTEXT_TYPE_WEIGHTS),
            "totalPaths": "How many distinct connecting paths were reconstructed "
                          "for this target within the hop budget.",
        },
    }


@register("subgraph.explore")
async def run_subgraph_explore(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """Expand the graph one hop around a selected node."""
    await _load_txkg()
    from DRP_Main.app.api.v1.endpoints.txkg_test import (
        entity_neighbors,
        id_to_name,
        id_to_type,
    )

    node_id = params.get("nodeId") or ""
    if not node_id:
        raise RunnerError("nodeId is required")

    # Accept either a BioKG id or a display name.
    resolved = node_id
    if node_id not in entity_neighbors:
        matches = [nid for nid, name in id_to_name.items() if str(name).lower() == node_id.lower()]
        if matches:
            resolved = matches[0]
    neighbors = entity_neighbors.get(resolved)
    if not neighbors:
        raise RunnerError(f"Node '{node_id}' is not present in the knowledge graph")

    ctx.progress(f"Expanding connections around {node_id}...")
    outgoing = neighbors.get("targets", set()) or set()
    incoming = neighbors.get("sources", set()) or set()
    limit = int(params.get("limit", 60))

    nodes = [
        {
            "id": resolved,
            "label": str(id_to_name.get(resolved, resolved)),
            "type": _spec_node_type(id_to_type.get(resolved, ""), False),
        }
    ]
    edges = []
    for neighbor in list(outgoing)[:limit]:
        nodes.append(
            {
                "id": neighbor,
                "label": str(id_to_name.get(neighbor, neighbor)),
                "type": _spec_node_type(id_to_type.get(neighbor, ""), False),
            }
        )
        edges.append({"source": resolved, "target": neighbor, "label": ""})
    for neighbor in list(incoming)[: max(0, limit - len(edges))]:
        nodes.append(
            {
                "id": neighbor,
                "label": str(id_to_name.get(neighbor, neighbor)),
                "type": _spec_node_type(id_to_type.get(neighbor, ""), False),
            }
        )
        edges.append({"source": neighbor, "target": resolved, "label": ""})

    seen, deduped = set(), []
    for node in nodes:
        if node["id"] not in seen:
            seen.add(node["id"])
            deduped.append(node)

    return {"nodes": deduped, "edges": edges, "legend": GRAPH_LEGEND}


# ── LitMinex ──────────────────────────────────────────────────────────────────
@register("litminex.query")
async def run_litminex_query(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Literature mining across the selected targets, with articles persisted.

    Runs the LitMineX functional spec's three tools (query expansion + retrieval →
    relevance scoring → summarisation) and maps the result onto the `/v1` wire
    contract: `confidenceScore` is the spec's 0-100 relevance score, and the job
    result carries the cited query answer that the insights panel serves.
    """
    from DRP_Main.app.modules.literature.litminex_service import (
        LitMineXService,
        MissingEntityError,
    )

    target_ids: List[str] = [t for t in (params.get("targetIds") or []) if t]
    free_text = params.get("query", "") or ""
    if not target_ids:
        fallback = _strip_query_noise(free_text)
        target_ids = [fallback] if fallback else []
    if not target_ids:
        raise RunnerError("At least one target is required for literature mining")

    max_results = int(params.get("maxResults", 20))
    service = LitMineXService()
    disease = _strip_query_noise(free_text) or None

    # The batched mining agent runs first. It issues one esearch plus one efetch for
    # the whole keyword set — 2 PubMed requests instead of 2 per target — which is
    # what keeps this inside NCBI's unauthenticated rate limit without an API key.
    # The spec chain below needs NER models, embeddings and an LLM scorer, none of
    # which are installed on Databricks Apps, so it returns nothing there; trying it
    # first only spent time before failing over.
    if settings.LITMINEX_PREFER_AGENT:
        ctx.progress("Searching PubMed for the selected targets...")
        agent_rows = await _mine_articles_fallback(target_ids, disease, max_results)
        if agent_rows:
            ctx.progress(f"Scored {len(agent_rows)} articles; saving...")
            items = _persist_articles(ctx.job_id, agent_rows)
            _persist_litminex_evidence(ctx, target_ids, agent_rows)
            return {
                "totalArticles": len(items),
                "targetIds": target_ids,
                "items": items,
                # The agent scores and ranks articles but does not compose a cited
                # answer — that is the spec chain's summarisation tool.
                "queryAnswer": {"text": "", "citedPmids": []},
                "summary": f"{len(items)} articles ranked",
            }
        ctx.progress("No articles from the batched search; trying the full chain...")

    # One LitMineX run per selected target; the spec's chain is scoped to a single
    # target/disease pair, and the /v1 surface lets the user pick several.
    combined: List[Dict[str, Any]] = []
    answers: List[str] = []
    cited: List[str] = []
    for target in target_ids[:5]:
        ctx.progress(f"LitMineX: expanding and retrieving PubMed for {target}...")
        payload: Dict[str, Any] = {"target": target, "source": "txkg_carryover"}
        if disease and disease.lower() != target.lower():
            payload["disease"] = disease
        else:
            payload["query"] = free_text or f"What is known about {target}?"
        try:
            result = await asyncio.to_thread(
                service.run,
                payload,
                session_id=ctx.job_id,
                max_results=max_results,
                # The /v1 target picker lets a user mine literature for a target
                # without naming a disease; a job must not fail asking for one.
                require_both=False,
            )
        except MissingEntityError as exc:
            raise RunnerError(str(exc))

        summaries = {row["pmid"]: row.get("per_article_summary", "") for row in result["article_table"]}
        combined.extend(
            _litminex_to_article_dict(article, target, summaries)
            for article in result["ranked_articles"]
        )
        if result["query_answer"]["text"]:
            answers.append(result["query_answer"]["text"])
        cited.extend(result["query_answer"]["cited_pmids"])

    # Dedupe across targets, keeping each article's best score.
    best: Dict[str, Dict[str, Any]] = {}
    for article in combined:
        key = article["pmid"] or article["title"]
        if key not in best or article["score"] > best[key]["score"]:
            best[key] = article
    ranked = sorted(best.values(), key=lambda a: a["score"], reverse=True)[:max_results]

    if not ranked:
        # The spec chain needs MeSH expansion, embeddings and an LLM scorer; when
        # any of those is unavailable (or NCBI rate-limits the per-target queries)
        # it yields nothing, and the researcher gets an empty table with no
        # explanation. The mining agent is a simpler, batched path — one esearch
        # plus one efetch for the whole keyword set — with deterministic keyword
        # scoring when the LLM cannot be reached, so it degrades to *something*.
        ctx.progress("No articles from the spec chain; retrying with the mining agent...")
        ranked = await _mine_articles_fallback(target_ids, disease, max_results)

    ctx.progress(f"Scored {len(ranked)} articles; assembling the cited answer...")
    items = _persist_articles(ctx.job_id, ranked)
    _persist_litminex_evidence(ctx, target_ids, ranked)
    return {
        "totalArticles": len(items),
        "targetIds": target_ids,
        "items": items,
        "queryAnswer": {"text": "\n\n".join(answers), "citedPmids": _dedupe_str(cited)},
        "summary": f"{len(items)} articles ranked",
    }


# ── SaaS Pipeline ─────────────────────────────────────────────────────────────
@register("pipeline.run")
async def run_pipeline(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Run the five modules end to end, feeding each stage's output to the next.

    This backs the "SaaS Pipeline" entry in the module catalogue, which the API
    advertised long before anything implemented it — selecting it returned 422.

    A stage that fails does not abort the run. ScreenSuite needs PyMOL/Vina, which
    are not installable on Databricks Apps, so on that deployment it will always
    fail; aborting there would throw away four stages of valid work. Each stage
    records its own status and the chain continues with whatever it already has.
    """
    query = params.get("query", "")
    carry = int(params.get("carryTargets", 3))
    max_results = int(params.get("maxResults", 20))
    wanted = params.get("stages") or ["TxKG", "LitMineX", "CurateX", "ScreenSuite", "NovSearch"]

    stages: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}
    targets: List[Dict[str, Any]] = []
    disease = ""
    top_target = ""

    async def _stage(name: str, coro_factory) -> Optional[Dict[str, Any]]:
        """Run one stage, recording success or a readable failure either way."""
        if name not in wanted:
            stages.append({"module": name, "status": "Skipped", "summary": "not requested"})
            return None
        ctx.progress(f"Pipeline — running {name}...")
        try:
            out = await coro_factory()
        except Exception as exc:  # noqa: BLE001 — one stage must not lose the rest
            logger.warning("pipeline stage %s failed: %s", name, exc)
            stages.append(
                {"module": name, "status": "Failed", "summary": f"{type(exc).__name__}: {exc}"}
            )
            return None
        stages.append({"module": name, "status": "Completed", "summary": out.get("summary", "")})
        results[name] = out
        return out

    txkg = await _stage("TxKG", lambda: run_txkg_query({"query": query, "limit": params.get("limit", 10)}, ctx))
    if txkg:
        targets = txkg.get("targets", [])[:carry]
        disease = txkg.get("disease", "") or ""
        top_target = (targets[0].get("name") or targets[0].get("uniprotId")) if targets else ""

    # Literature search needs names, not accessions: PubMed text says "JAK2", never
    # "O60674", so carrying uniprotId forward returned zero articles every time.
    # Prefer geneName, then the display name, and only fall back to the accession.
    target_ids = [
        t.get("geneName") or t.get("name") or t.get("uniprotId")
        for t in targets
        if t.get("geneName") or t.get("name") or t.get("uniprotId")
    ]
    if target_ids:
        await _stage(
            "LitMineX",
            lambda: run_litminex_query(
                {"targetIds": target_ids, "query": query, "maxResults": max_results}, ctx
            ),
        )
    else:
        stages.append({"module": "LitMineX", "status": "Skipped", "summary": "no targets to carry"})

    curatex = None
    if top_target:
        curatex = await _stage(
            "CurateX",
            lambda: run_curatex_compounds(
                {"target": top_target, "disease": disease, "numResults": 20}, ctx
            ),
        )
    else:
        stages.append({"module": "CurateX", "status": "Skipped", "summary": "no target to profile"})

    compounds = (curatex or {}).get("compounds", [])[:5]
    if top_target:
        await _stage(
            "ScreenSuite",
            lambda: run_screensuite_screen(
                {"target": top_target, "compounds": [], "compoundLibrary": None}, ctx
            ),
        )
        drug = (compounds[0].get("name") or compounds[0].get("compound")) if compounds else None
        await _stage(
            "NovSearch",
            lambda: run_novsearch_assess(
                {"target": top_target, "disease": disease, "drug": drug, "numResults": 5}, ctx
            ),
        )

    completed = [s["module"] for s in stages if s["status"] == "Completed"]
    return {
        "query": query,
        "disease": disease,
        "topTarget": top_target,
        "stages": stages,
        "results": results,
        "targets": targets,
        "summary": (
            f"Pipeline finished {len(completed)}/{len(wanted)} stage(s): "
            + ", ".join(completed)
            if completed
            else "Pipeline produced no completed stages"
        ),
    }


async def _mine_articles_fallback(
    target_ids: List[str], disease: Optional[str], max_results: int
) -> List[Dict[str, Any]]:
    """
    Retrieve articles with the batched mining agent and map them onto the article
    shape `_persist_articles` stores.

    Targets are the mandatory protein keywords; the disease is optional context,
    matching the agent's scoring contract (protein up to 60, context up to 30,
    title matches +10).

    A target that arrived as a phrase ("JAK2 in thrombocytosis") is split first.
    Without this the whole phrase is sent as one keyword and PubMed is asked for
    the literal string `"JAK2 in thrombocytosis"[Title/Abstract]`, which matches
    nothing — the search silently returns zero rather than failing.
    """
    from DRP_Main.app.drp.dispatch import split_target_disease
    from DRP_Main.app.modules.literature import mining_agent

    proteins: List[str] = []
    context: List[str] = [disease] if disease else []
    for raw in target_ids[:5]:
        protein, extra = split_target_disease(raw)
        if protein:
            proteins.append(protein)
        if extra and extra.lower() not in {c.lower() for c in context}:
            context.append(extra)

    if not proteins:
        return []

    try:
        rows = await mining_agent.process_search(
            article_keywords=proteins,
            search_keywords=context,
            max_results=max_results,
        )
    except Exception as exc:  # noqa: BLE001 — a fallback must not raise
        logger.warning("literature mining agent fallback failed: %s", exc)
        return []

    # Key names must match `_litminex_to_article_dict` exactly — `_persist_articles`
    # reads snake_case (`found_keywords`, `pdf_file_path`). Writing camelCase here
    # is why the results table rendered an empty keywords column.
    return [
        {
            "pmid": row.get("pmid") or "",
            "pmcid": None,
            "title": row.get("title", ""),
            "authors": row.get("authors", ""),
            "year": row.get("year"),
            "abstract": row.get("preview", ""),
            "keywords": [],
            "found_keywords": row.get("found_keywords", []),
            "score": float(row.get("score", 0.0)),
            "preview": row.get("preview", ""),
            "pdf_file_path": row.get("pdf_file_path", ""),
            "pubmed_url": row.get("pubmed_url", ""),
        }
        for row in rows
    ]


def _litminex_to_article_dict(
    article: Dict[str, Any], target: str, summaries: Dict[str, str]
) -> Dict[str, Any]:
    """Map a spec-shaped ranked article onto the keys `_persist_articles` stores."""
    pmid = article.get("pmid", "")
    pub_date = article.get("pub_date") or ""
    matched = sorted(
        {
            mention
            for sentence in article.get("relation_sentences", [])
            for mention in sentence.get("target_mentions", []) + sentence.get("disease_mentions", [])
        }
    ) or [target]
    return {
        "pmid": pmid,
        "pmcid": article.get("pmcid"),
        "title": article.get("title", ""),
        "authors": ", ".join(article.get("authors", [])[:3]),
        "year": int(pub_date[:4]) if pub_date[:4].isdigit() else None,
        "abstract": article.get("abstract", ""),
        "keywords": [],
        "found_keywords": matched,
        "score": float(article.get("relevance_score", 0.0)),
        # The table view shows the evidence-built summary, not the raw abstract (§5).
        "preview": summaries.get(pmid) or article.get("abstract_preview", ""),
        "pdf_file_path": article.get("pdf_link", ""),
        "pubmed_url": article.get("pubmed_url", ""),
    }


def _dedupe_str(values: List[str]) -> List[str]:
    seen, out = set(), []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _persist_litminex_evidence(
    ctx: JobContext, target_ids: List[str], ranked: List[Dict[str, Any]]
) -> None:
    """Persist LitMineX's ranked articles to reference.target + bronze.lit_evidence."""
    try:
        if not target_ids:
            return
        uc_store.upsert_rows(
            "reference",
            "target",
            ["target_id"],
            [
                {
                    "target_id": t,
                    "uniprot_id": None,
                    "target_name": t,
                    "organism": None,
                    "created_ts": datetime.now(timezone.utc).isoformat(),
                }
                for t in target_ids
            ],
        )
        _persist_agent_invocation(ctx, "LitMineX")

        # Articles are deduped across all selected targets before this point, so the
        # per-article target association isn't retained; every evidence row anchors to
        # the first selected target rather than splitting evidence across several rows.
        primary_target = target_ids[0]
        evidence_rows = [
            {
                "evidence_id": str(uuid.uuid4()),
                "invocation_id": ctx.job_id,
                "target_id": primary_target,
                "pubmed_id": article.get("pmid"),
                "title": article.get("title"),
                "article_link": article.get("pubmed_url"),
                "publication_year": article.get("year"),
                "relevance_score": article.get("score"),
                "llm_score": None,
                "abstract_preview": (article.get("abstract") or "")[:2000],
                "relation_sentences": None,
                "query_text": None,
                "query_answer_summary": article.get("preview"),
                "selected_for_qa": False,
                "qa_history": None,
            }
            for article in ranked
        ]
        uc_store.insert_rows("bronze", "lit_evidence", evidence_rows)
    except Exception:
        logger.warning("uc_store: failed to persist LitMineX evidence for job %s", ctx.job_id, exc_info=True)


def _persist_articles(job_id: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Store literature hits so /articles/{articleId} can serve them later."""
    db = JobSession()
    stored: List[Dict[str, Any]] = []
    try:
        for raw in results:
            pmid = str(raw.get("pmid") or "")
            article_id = f"art_{pmid}" if pmid else f"art_{uuid.uuid4().hex[:12]}"
            pmcid = raw.get("pmcid")
            pmc_link = (
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                if pmcid
                else raw.get("pubmed_url", "")
            )
            existing = db.query(DrpArticle).filter(DrpArticle.id == article_id).first()
            if existing is None:
                existing = DrpArticle(id=article_id)
                db.add(existing)
            existing.job_id = job_id
            existing.pmid = pmid
            existing.title = raw.get("title", "")
            existing.authors = raw.get("authors", "") or ""
            existing.year = raw.get("year")
            existing.abstract = raw.get("abstract", "") or raw.get("preview", "")
            existing.keywords = raw.get("keywords", []) or []
            existing.found_keywords = raw.get("found_keywords", []) or []
            existing.confidence_score = float(raw.get("score", 0) or 0)
            existing.preview = raw.get("preview", "")
            existing.pmc_link = pmc_link
            existing.pdf_url = raw.get("pdf_file_path", "")
            stored.append(
                {
                    "id": article_id,
                    "title": existing.title,
                    "year": existing.year,
                    "confidenceScore": existing.confidence_score,
                    "foundKeywords": existing.found_keywords,
                }
            )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return stored


# ── CuraTeX ───────────────────────────────────────────────────────────────────
@register("curatex.target_profile")
async def run_curatex_target_profile(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Tool 1 (§5) — the editable weighted drug profile built from the target's
    known ligands, plus the exclusion flag when a disease is supplied.
    """
    from DRP_Main.app.modules.drug_curation.curatex_service import CurateXError, CurateXService

    target = (params.get("target") or "").strip()
    if not target:
        raise RunnerError("target is required")

    ctx.progress(f"Resolving {target} and pulling its known ligands...")
    try:
        payload = await asyncio.to_thread(
            CurateXService().build_profile,
            target=target,
            disease=params.get("disease"),
            weights=params.get("weights"),
            values=params.get("values"),
            session_id=ctx.session_id or f"job:{ctx.job_id}",
        )
    except CurateXError as exc:
        raise RunnerError(str(exc))

    profiles = payload.get("profiles") or []
    profile = profiles[0] if profiles else {}
    ctx.progress("Computing criterion baselines and the exclusion flag...")
    return {
        "target": target,
        "stage": payload.get("stage"),
        "profile": profile,
        "criteria": profile.get("criteria", []),
        "ligandCount": profile.get("ligandCount", 0),
        "exclusion": profile.get("exclusion", {}),
        "coverage": profile.get("coverage", {}),
        "warnings": profile.get("warnings", []),
        # The profile is editable before scoring — that is the whole point of
        # returning it as its own job rather than folding it into scoring.
        "editable": True,
        "sessionId": ctx.session_id or f"job:{ctx.job_id}",
        "summary": (
            f"Profile for {target} built from {profile.get('ligandCount', 0)} known ligand(s)"
        ),
    }


@register("curatex.compounds")
async def run_curatex_compounds(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    The full CurateX chain: profile → candidate matching and scoring → evidence.

    `compound` is a starting point, not a filter: it is resolved to a structure
    and reported alongside the ranked candidates so the user can see where their
    seed compound landed relative to the pool.
    """
    from DRP_Main.app.modules.drug_curation.curatex_service import CurateXError, CurateXService

    target = (params.get("target") or "").strip()
    compound = (params.get("compound") or "").strip()
    if not target:
        raise RunnerError("target is required — CurateX profiles a target's known ligands")

    ctx.progress(f"Building the ideal candidate profile for {target}...")
    try:
        result = await asyncio.to_thread(
            CurateXService().run,
            target=target,
            disease=params.get("disease"),
            weights=params.get("weights"),
            values=params.get("values"),
            top_n=int(params.get("numResults", 25)),
            session_id=ctx.session_id or f"job:{ctx.job_id}",
        )
    except CurateXError as exc:
        raise RunnerError(str(exc))

    table = result.get("candidateTable", [])
    metadata = result.get("metadata", {})
    ctx.progress(
        f"Scored {metadata.get('scoredCount', 0)} candidates; validating evidence..."
    )
    _persist_curatex_candidates(ctx, target, table)
    return {
        "target": target,
        "compound": compound or None,
        "disease": result.get("disease"),
        "totalCompounds": len(table),
        "compounds": table,
        "excluded": result.get("excludedCandidates", []),
        "scoreOnly": result.get("scoreOnlyCandidates", []),
        "recommendation": result.get("recommendation", ""),
        "profile": result.get("profile", {}),
        "searchMetadata": metadata,
        "warnings": result.get("warnings", []),
        "summary": (
            f"{len(table)} repurposing candidate(s) ranked for {target}; "
            f"{metadata.get('excludedCount', 0)} already-indicated drug(s) excluded"
        ),
    }


def _persist_curatex_candidates(
    ctx: JobContext, target: str, table: List[Dict[str, Any]]
) -> None:
    """Persist CurateX's ranked candidates to reference.target/compound + silver.drug_candidate."""
    try:
        if not table:
            return
        uc_store.upsert_rows(
            "reference",
            "target",
            ["target_id"],
            [
                {
                    "target_id": target,
                    "uniprot_id": None,
                    "target_name": target,
                    "organism": None,
                    "created_ts": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )
        uc_store.upsert_rows(
            "reference",
            "compound",
            ["compound_id"],
            [
                {
                    "compound_id": row.get("chemblId"),
                    "compound_name": row.get("name"),
                    "smiles": row.get("smiles"),
                    "molecular_formula": None,
                    "molecular_weight": None,
                    "source_db": "ChEMBL",
                    "created_ts": datetime.now(timezone.utc).isoformat(),
                }
                for row in table
                if row.get("chemblId")
            ],
        )
        _persist_agent_invocation(ctx, "CurateX")

        candidate_rows = [
            {
                "candidate_id": str(uuid.uuid4()),
                "invocation_id": ctx.job_id,
                "target_id": target,
                "compound_id": row.get("chemblId"),
                "property_profile": None,
                "profile_weights": None,
                "user_edited": False,
                "match_score": row.get("compositeScore"),
                "matched_criteria": None,
                "unmatched_criteria": None,
                "evidence_links": row.get("evidenceLinks"),
                "verification_status": row.get("evidenceStrength"),
                "excluded_existing_drug": False,
                "created_ts": datetime.now(timezone.utc).isoformat(),
                "updated_ts": datetime.now(timezone.utc).isoformat(),
            }
            for row in table
            if row.get("chemblId")
        ]
        uc_store.insert_rows("silver", "drug_candidate", candidate_rows)
    except Exception:
        logger.warning("uc_store: failed to persist CurateX candidates for job %s", ctx.job_id, exc_info=True)


# ── ScreenSuite ───────────────────────────────────────────────────────────────
#: A 4-character PDB code. A `target` in this shape is taken as an identifier,
#: so the job docks that exact structure instead of searching RCSB by name.
_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def _screensuite_job_query(target: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the ScreenSuite job's `query` payload from the `/v1` request.

    The `/v1` contract carries a free-text `target` and `compounds` as
    `[{drug_name, ...}]`. A compound may also carry `identifier`/`source` (e.g. a
    PubChem CID); when present the job uses it as-is and skips the name lookup.
    """
    protein: Dict[str, str] = {"name": target}
    if _PDB_ID.match(target):
        protein = {"name": target, "identifier": target.upper(), "source": "pdb"}

    drugs: List[Dict[str, str]] = []
    for compound in params.get("compounds") or []:
        if not isinstance(compound, dict):
            continue
        name = str(compound.get("drug_name") or compound.get("name") or "").strip()
        if not name:
            continue
        entry = {"name": name}
        identifier = str(
            compound.get("identifier") or compound.get("pubchem_id") or compound.get("pubchemId") or ""
        ).strip()
        if identifier:
            entry["identifier"] = identifier
            source = str(compound.get("source") or "").strip().lower()
            if source:
                entry["source"] = source
        drugs.append(entry)

    return {"proteins": [protein], "drugs": drugs}


async def _run_screensuite_on_databricks(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Run ScreenSuite as its Databricks job and rebuild the contract response from Gold.

    Docking cannot run inside the API process — AutoDock Vina is a native binary
    that neither the laptops (endpoint security) nor the App container can
    execute — so the real pipeline lives in `jobs/screensuite_job.py`. The
    notebook writes one `docking_results` row per ranked pose, plus a
    `Failed[...]` row for anything that never reached docking, all tagged with
    this job's id. The response keeps the in-process shape (`hits` of
    `ScreeningHit`), so `/v1/agents/screensuite/{jobId}/hits` is unchanged.
    """
    import json

    target = (params.get("target") or "").strip()
    query = _screensuite_job_query(target, params)
    if not query["drugs"]:
        raise RunnerError(
            "Name at least one compound to screen: compounds: "
            '[{"drug_name": "ruxolitinib"}]. A PubChem CID can be passed as '
            '"identifier" to skip the name lookup.'
        )

    ctx.progress(
        f"Docking {len(query['drugs'])} compound(s) against {target} on Databricks..."
    )
    # Only `query` is sent: run_now rejects any parameter the job does not
    # declare, and the notebook reads its input from `query`.
    await execution.run_module("ScreenSuite", {"query": json.dumps(query)}, ctx)

    ctx.progress("Reading docking results...")
    try:
        rows = await asyncio.to_thread(gold.read_job_rows, "docking_results", ctx.job_id)
    except gold.GoldUnavailable as exc:
        raise RunnerError(f"The ScreenSuite job finished but its results could not be read: {exc}")

    def _failed(row: Dict[str, Any]) -> bool:
        return (row.get("status") or "").startswith("Failed")

    ranked = [row for row in rows if not _failed(row)]
    failures = [row.get("status") for row in rows if _failed(row) and row.get("status")]

    if not ranked:
        reason = "; ".join(failures) or "the job wrote no results"
        # An ambiguous structure is a question, not a failure. The resolution
        # step deliberately stops before spending compute and hands back a
        # shortlist — docking the wrong JAK2 domain is far more expensive than
        # one round trip — but the runner was turning that into a failed job, so
        # the researcher saw a red error instead of the choice they were being
        # asked to make. Mirrors CurateX's `awaiting_target_confirmation`.
        candidates = _structure_choices(failures)
        if candidates:
            return {
                "stage": "awaiting_structure_confirmation",
                "target": target,
                "structureChoices": candidates,
                "hits": [],
                "totalHits": 0,
                "summary": (
                    f"'{target}' matches {len(candidates)} structures. "
                    "Choose one to screen against."
                ),
            }
        raise RunnerError(f"Screening produced no hits for '{target}'. {reason}")

    ranked.sort(key=lambda r: gold.as_float(r, "affinity_kcal_mol", 0.0) or 0.0)
    hits = [
        {
            "mode": gold.as_int(row, "mode"),
            "compound": row.get("ligand") or "",
            "protein": row.get("protein") or target,
            "affinityKcalPerMol": gold.as_float(row, "affinity_kcal_mol"),
            "outputFile": row.get("pose_file") or "",
        }
        for row in ranked
    ]
    _persist_screensuite_hits(ctx, target, hits)

    return {
        "target": target,
        "queued": False,
        "status": "partial" if failures else "success",
        "compoundLibrary": params.get("compoundLibrary"),
        "hits": hits,
        "failures": failures,
        "runId": ranked[0].get("run_id"),
        "summary": f"{len(hits)} screening hit(s) for {target}"
        + (f"; {len(failures)} failure(s)" if failures else ""),
    }


@register("screensuite.screen")
async def run_screensuite_screen(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Virtual screening against a receptor.

    Delegates to the ScreenSuite Databricks job when this deployment opts the
    module in (`DRP_DATABRICKS_MODULES` / `DRP_EXECUTION_BACKEND`) — the only
    place docking can actually run. In-process, docking needs a prepared receptor
    PDB plus ligand SDFs; when those are not supplied we return any previously
    computed hits for the target rather than silently producing nothing.
    """
    from DRP_Main.app.modules.screening.pipeline_service import (
        get_protein_status,
        get_top_5_percent,
    )

    target = (params.get("target") or "").strip()
    if not target:
        raise RunnerError("target is required")

    if execution.should_delegate("ScreenSuite"):
        return await _run_screensuite_on_databricks(params, ctx)

    pdb_path = params.get("pdbFilePath")
    compounds = params.get("compounds") or []

    if pdb_path and compounds:
        from DRP_Main.app.modules.screening.queue_service import queue_service
        from DRP_Main.app.modules.screening.schemas import DrugBase, ProteinBase

        ctx.progress(f"Queueing docking run for {target} ({len(compounds)} ligands)...")
        protein = ProteinBase(protein_name=target, pdb_file_path=pdb_path)
        drugs = [
            DrugBase(drug_name=c["drug_name"], sdf_file_path=c["sdf_file_path"]) for c in compounds
        ]
        queued = await asyncio.to_thread(queue_service.add_to_queue, protein, drugs)
        return {
            "target": target,
            "queued": True,
            "taskId": queued.get("task_id"),
            "compoundLibrary": params.get("compoundLibrary"),
            "hits": [],
            "summary": f"Docking queued for {target} — poll /v1/agents/screensuite/{ctx.job_id}/hits",
        }

    ctx.progress(f"Collecting existing screening results for {target}...")
    # Both lookups read files that exist only once a target has been screened. The
    # status read used to sit outside the guard, so an unknown target raised a raw
    # FileNotFoundError and the readable message below was never reached.
    try:
        status = await asyncio.to_thread(get_protein_status, target)
        rows = await asyncio.to_thread(get_top_5_percent, target)
    except FileNotFoundError:
        raise RunnerError(
            f"No screening results exist for '{target}'. Supply pdbFilePath and "
            "compounds ([{drug_name, sdf_file_path}]) to run a new docking job."
        )

    hits = [
        {
            "mode": _as_int(row.get("Mode")),
            "compound": row.get("ligand", ""),
            "protein": row.get("protein", target),
            "affinityKcalPerMol": _as_float(row.get("Affinity_kcal_per_mol")),
            "outputFile": row.get("out_pdbqt_file", ""),
        }
        for row in (rows or [])
    ]
    _persist_screensuite_hits(ctx, target, hits)
    return {
        "target": target,
        "queued": False,
        "status": status,
        "compoundLibrary": params.get("compoundLibrary"),
        "hits": hits,
        "summary": f"{len(hits)} screening hits for {target}",
    }


def _persist_screensuite_hits(ctx: JobContext, target: str, hits: List[Dict[str, Any]]) -> None:
    """Persist ScreenSuite docking hits to reference.target/compound + silver.docking_result."""
    try:
        if not hits:
            return
        uc_store.upsert_rows(
            "reference",
            "target",
            ["target_id"],
            [
                {
                    "target_id": target,
                    "uniprot_id": None,
                    "target_name": target,
                    "organism": None,
                    "created_ts": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )
        uc_store.upsert_rows(
            "reference",
            "compound",
            ["compound_id"],
            [
                {
                    "compound_id": h["compound"],
                    "compound_name": h["compound"],
                    "smiles": None,
                    "molecular_formula": None,
                    "molecular_weight": None,
                    "source_db": None,
                    "created_ts": datetime.now(timezone.utc).isoformat(),
                }
                for h in hits
                if h.get("compound")
            ],
        )
        _persist_agent_invocation(ctx, "ScreenSuite")

        docking_rows = [
            {
                "docking_id": str(uuid.uuid4()),
                "invocation_id": ctx.job_id,
                "candidate_id": None,
                "target_id": target,
                "compound_id": h.get("compound"),
                "mode": h.get("mode"),
                "binding_score_kcal_mol": h.get("affinityKcalPerMol"),
                "pose_rank": h.get("mode"),
                "protein_file_path": None,
                "ligand_file_path": None,
                "log_file_path": None,
                "residue_interactions": None,
                "hydrogen_bonds": None,
                "plip_report_path": None,
                "visualization_path": None,
                "overall_recommendation": None,
                "rerun_of_docking_id": None,
                "created_ts": datetime.now(timezone.utc).isoformat(),
            }
            for h in hits
            if h.get("compound")
        ]
        uc_store.insert_rows("silver", "docking_result", docking_rows)
    except Exception:
        logger.warning("uc_store: failed to persist ScreenSuite hits for job %s", ctx.job_id, exc_info=True)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── NovSearch ─────────────────────────────────────────────────────────────────
@register("novsearch.assess")
async def run_novsearch_assess(params: Dict[str, Any], ctx: JobContext) -> Dict[str, Any]:
    """
    Patent novelty assessment — the NovSearch functional-spec chain
    (`modules/novelty/novsearch_service`): USPTO PatentsView retrieval with BM25 +
    RRF + a hosted MiniLM cross-encoder, indexing into Databricks Vector Search
    with BGE-large embeddings, and one SaulLM synthesis call scoped to the query.

    Takes either input shape: a ScreenSuite carry-over (target / drug / disease,
    with the candidate and docking ids carried through so the report traces back
    to the screening result) or a fresh free-text query.
    """
    from DRP_Main.app.modules.novelty import novsearch_service as ns

    payload = {
        "target": (params.get("target") or "").strip() or None,
        "drug": (params.get("drug") or "").strip() or None,
        "disease": (params.get("disease") or "").strip() or None,
        "query": (params.get("query") or "").strip() or None,
        "candidate_id": (params.get("candidateId") or "").strip() or None,
        "docking_id": (params.get("dockingId") or "").strip() or None,
    }
    payload = {k: v for k, v in payload.items() if v}
    if not payload:
        raise RunnerError("Provide either a query, or a target/drug/disease combination.")

    num_results = max(1, min(20, int(params.get("numResults", 5))))

    try:
        result = await ns.run_novsearch(payload, num_results, progress=ctx.progress)
    except ns.AmbiguousQueryError as exc:
        raise RunnerError(str(exc)) from exc

    report = result.get("report", {})
    recommendations = report.get("recommendations", "")
    if isinstance(recommendations, str):
        recommendations = [r.strip("-• ") for r in recommendations.splitlines() if r.strip()]

    out: Dict[str, Any] = {
        "query": result.get("query", ""),
        "inputSource": result.get("input_source", "user_direct"),
        "target": (params.get("target") or "").strip(),
        "disease": (params.get("disease") or "").strip(),
        "assessment": report.get("agent_answer", ""),
        "recommendations": list(recommendations or []),
        "patents": result.get("patents_table", []),
        "patentsUsed": report.get("patents_used", []),
        "totalPatents": report.get("total_patents", 0),
        "totalChunks": report.get("total_chunks", 0),
        "modelUsed": report.get("model_used"),
        "sessionStateUpdate": result.get("session_state_update", {}),
        "summary": f"{len(result.get('patents_table', []))} patents analysed",
    }
    # Absent, not null, for a fresh query (§6).
    if result.get("candidate_id"):
        out["candidateId"] = result["candidate_id"]
    if result.get("docking_id"):
        out["dockingId"] = result["docking_id"]
    _persist_novsearch_report(ctx, out)
    return out


def _persist_novsearch_report(ctx: JobContext, out: Dict[str, Any]) -> None:
    """Persist NovSearch's report to reference.target/patent + gold.novelty_report/novelty_patent."""
    try:
        target = out.get("target")
        if target:
            uc_store.upsert_rows(
                "reference",
                "target",
                ["target_id"],
                [
                    {
                        "target_id": target,
                        "uniprot_id": None,
                        "target_name": target,
                        "organism": None,
                        "created_ts": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            )

        # `out["patents"]` is `novsearch_service`'s `patents_table` — snake_case keys
        # (patent_id, title, assignee, filing_date, relevance_score, rank), no link field.
        patents = out.get("patents") or []
        uc_store.upsert_rows(
            "reference",
            "patent",
            ["patent_id"],
            [
                {
                    "patent_id": p.get("patent_id"),
                    "patent_title": p.get("title"),
                    "patent_link": None,
                    "filing_date": p.get("filing_date"),
                    "assignee": p.get("assignee"),
                }
                for p in patents
                if p.get("patent_id")
            ],
        )

        _persist_agent_invocation(ctx, "NovSearch", recommendation_text="\n".join(out.get("recommendations") or []))

        novelty_id = str(uuid.uuid4())
        uc_store.insert_rows(
            "gold",
            "novelty_report",
            [
                {
                    "novelty_id": novelty_id,
                    "invocation_id": ctx.job_id,
                    "candidate_id": out.get("candidateId"),
                    "docking_id": out.get("dockingId"),
                    "target_id": target,
                    "query_text": out.get("query"),
                    "rerank_score": None,
                    "total_patents": out.get("totalPatents"),
                    "total_chunks": out.get("totalChunks"),
                    "agent_answer": out.get("assessment"),
                    "recommendations": "\n".join(out.get("recommendations") or []),
                    "qa_history": None,
                    "created_ts": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        link_rows = [
            {
                "link_id": str(uuid.uuid4()),
                "novelty_id": novelty_id,
                "patent_id": p.get("patent_id"),
                "rerank_score": p.get("relevance_score"),
                "cited_chunks": None,
            }
            for p in patents
            if p.get("patent_id")
        ]
        uc_store.insert_rows("gold", "novelty_patent", link_rows)
    except Exception:
        logger.warning("uc_store: failed to persist NovSearch report for job %s", ctx.job_id, exc_info=True)
