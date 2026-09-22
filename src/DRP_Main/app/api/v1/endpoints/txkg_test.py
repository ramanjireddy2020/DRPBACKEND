"""
TxGNN FastAPI Server - Context Graph-Based Target Identification
Attached as a router to the main FastAPI server.

Architecture (v1 structure + v2 intelligence):
- v1 code structure, endpoint layout, and organization preserved exactly
- v2 context graph traversal: pathway, GO BP/MF/CC, complex, genetic disorder, tissue, cell
- v2 local TSV protein name index (HUMAN only) — no UniProt API calls
- v2 context-aware scoring: pathway > genetic_disorder > GO_BP > GO_MF > complex > GO_CC > tissue > cell > PPI
- Context graph traversal shown in both subgraph visualisation and metapath extraction
- Multi-hop BFS up to 5 hops (configurable)
- Real PubMed article fetch, LLM interpretation via Groq
"""

from fastapi import FastAPI, APIRouter, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import networkx as nx
from collections import deque, defaultdict
from difflib import get_close_matches
import numpy as np
import warnings
import pickle
import os
import re
from functools import lru_cache
import json
import httpx
import xml.etree.ElementTree as ET
from urllib.parse import quote
import asyncio
from DRP_Main.app.core.llm import llm_client
from DRP_Main.app.core.config import settings
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

router = APIRouter()

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================


# BioKG Data Configuration
# Resolve relative to this file (not the current working directory) so the data is
# found regardless of where the server is launched from. Overridable via BIOKG_DATA_DIR.
BIOKG_DATA_DIR    = os.getenv(
    "BIOKG_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "biokg_data"),
)
LINKS_PATH        = f"{BIOKG_DATA_DIR}/biokg.links.tsv"
META_DISEASE_PATH = f"{BIOKG_DATA_DIR}/biokg.metadata.disease.tsv"
META_PROTEIN_PATH = f"{BIOKG_DATA_DIR}/biokg.metadata.protein.tsv"
META_DRUG_PATH    = f"{BIOKG_DATA_DIR}/biokg.metadata.drug.tsv"
META_PATHWAY_PATH = f"{BIOKG_DATA_DIR}/biokg.metadata.pathway.tsv"
CACHE_PATH        = f"{BIOKG_DATA_DIR}/cache_optimized3.pkl"
MAX_HOP_DEPTH     = 3

GRAPHS_DIR = f"{BIOKG_DATA_DIR}/graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Global data structures
df_links         = None
meta_disease     = None
meta_protein     = None
meta_drug        = None
id_to_name       = {}
id_to_type       = {}
id_to_raw_name   = {}
disease_name_to_id = {}
disease_ids      = set()
diseases_in_kg   = set()
drug_ids         = set()
protein_ids      = set()

# Precomputed indexes
disease_links_index = {}
drug_connections    = {}
entity_neighbors    = {}
edge_lookup         = {}   # (source, target) → edge_type — O(1) resolution
ctx_adj             = {}   # node → {neighbor: edge_type} — pre-filtered (no drugs/diseases)

# ============================================================================
# PERFORMANCE CACHES
# ============================================================================

# Global protein name index built from biokg.metadata.protein.tsv (HUMAN only)
# Structure: { uniprot_id: {'id', 'name', 'full_name', 'gene'} }
PROTEIN_NAME_INDEX: dict = {}

# Global cache for target extraction (per session)
TARGET_CACHE = {}

# Global cache for metapaths (per session)
METAPATH_CACHE = {}

# Context graph cache — { disease_id: nx.DiGraph }
CONTEXT_GRAPH_CACHE = {}

# ============================================================================
# CONTEXT GRAPH CONSTANTS
# ============================================================================

# Context entity types valid as traversal intermediates in BioKG.
CONTEXT_ENTITY_TYPES = {
    'gene/protein',
    'pathway',
    'biological_process',
    'cellular_component',
    'molecular_function',
    'complex',
    'genetic_disorder',
    'cell',
    'tissue',
    'other',
}

# Biological relevance weights for context scoring.
#   pathway (1.0)            — DISEASE_PATHWAY_ASSOCIATION + PROTEIN_PATHWAY_ASSOCIATION
#   genetic_disorder (0.95)  — DISEASE_GENETIC_DISORDER + RELATED_GENETIC_DISORDER
#   biological_process (0.9) — GO_BP on proteins + PATHWAY_GO_BP on pathways
#   molecular_function (0.85)— GO_MF on proteins + PATHWAY_GO_MF on pathways
#   complex (0.8)            — MEMBER_OF_COMPLEX (Reactome)
#   cellular_component (0.65)— GO_CC on proteins + PATHWAY_GO_CC on pathways
#   tissue (0.5)             — PROTEIN_EXPRESSED_IN (HPA)
#   cell (0.4)               — PART_OF_TISSUE (HPA/Cellosaurus)
#   gene/protein (0.3)       — PPI (guilt by association only)
#   other (0.2)              — fallback
CONTEXT_TYPE_WEIGHTS = {
    'pathway':             1.0,
    'genetic_disorder':    0.95,
    'biological_process':  0.9,
    'molecular_function':  0.85,
    'complex':             0.8,
    'cellular_component':  0.65,
    'tissue':              0.5,
    'cell':                0.4,
    'gene/protein':        0.3,
    'other':               0.2,
}


class Target(BaseModel):
    id: str
    name: str
    type: str
    score: float
    paths: int

class Disease(BaseModel):
    id: str
    name: str

# ============================================================================
# PUBMED INTEGRATION
# ============================================================================

async def fetch_pubmed_articles_for_target(disease_name: str, target_name: str, limit: int = 2) -> List[Dict]:
    """
    Fetch REAL PubMed articles for a disease-target pair.
    Returns only verified articles with real PMIDs.
    """
    try:
        target_clean = target_name.replace("UniProt:", "").strip()
        search_query = f'("{disease_name}"[Title/Abstract]) AND ("{target_clean}"[Title/Abstract])'
        print(f"      📡 PubMed Query: {search_query}")

        async with httpx.AsyncClient(timeout=15.0) as http_client:
            search_url    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed", "term": search_query,
                "retmax": limit * 2, "retmode": "json", "sort": "relevance"
            }
            search_response = await http_client.get(search_url, params=search_params)

            if search_response.status_code != 200:
                print(f"      ⚠️  PubMed search failed: {search_response.status_code}")
                return []

            pmids = search_response.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                print(f"      ⚠️  No results, trying broader search...")
                search_params["term"] = f'"{disease_name}" AND "{target_clean}"'
                resp2 = await http_client.get(search_url, params=search_params)
                if resp2.status_code == 200:
                    pmids = resp2.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                print(f"      ⚠️  No PubMed results for {disease_name} - {target_clean}")
                return []

            pmids = pmids[:limit]
            print(f"      ✅ Found {len(pmids)} PMIDs: {pmids}")

            fetch_response = await http_client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
            )
            if fetch_response.status_code != 200:
                return []

            articles = []
            try:
                root = ET.fromstring(fetch_response.content)
                for article_elem in root.findall(".//PubmedArticle"):
                    try:
                        pmid_elem  = article_elem.find(".//PMID")
                        pmid       = pmid_elem.text if pmid_elem is not None else None
                        if not pmid:
                            continue
                        title_elem = article_elem.find(".//ArticleTitle")
                        title      = title_elem.text if title_elem is not None else "No title available"
                        if title:
                            title = title.strip().rstrip('.')
                        articles.append({
                            "title": title, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "pmid": pmid, "source": "PubMed"
                        })
                    except Exception as e:
                        print(f"      ⚠️  Error parsing article: {e}")
                        continue
            except ET.ParseError as e:
                print(f"      ⚠️  XML parsing error: {e}")
                return []

            print(f"      ✅ Parsed {len(articles)} articles successfully")
            return articles

    except Exception as e:
        print(f"      ❌ Error fetching PubMed articles: {e}")
        return []


# ============================================================================
# PROTEIN NAME INDEX — replaces ALL UniProt API fetching
# ============================================================================

def build_protein_name_index() -> None:
    """
    Parse biokg.metadata.protein.tsv (long format: uniprot_id | attribute | value)
    and build PROTEIN_NAME_INDEX for HUMAN proteins only.

    TSV columns (tab-separated):
        col0 = UniProt accession  (e.g. Q2M2I8)
        col1 = attribute name     (e.g. NAME, FULL_NAME, SHORT_NAME, SPECIES …)
        col2 = value

    Name priority per protein:
        FULL_NAME if len <= 100  (stored as full_name)
        SHORT_NAME               (stored as short_name)
        NAME                     (gene symbol, stored as gene)

    Display name: FULL_NAME (≤100 chars) → SHORT_NAME → NAME (gene symbol) → uniprot_id
    """
    global PROTEIN_NAME_INDEX

    print("\n📚 Building protein name index from TSV (HUMAN only)…")
    raw: dict = {}

    try:
        with open(META_PROTEIN_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                uid  = parts[0].strip()
                attr = parts[1].strip().upper()
                val  = parts[2].strip()

                # Skip header rows
                if uid.upper() in ("P32234", "ID", "ACCESSION", "UNIPROT_ID", "PROTEIN_ID"):
                    continue
                if attr in ("NAME", "ATTRIBUTE", "PROPERTY") and val.upper() in ("NAME", "VALUE", "128UP"):
                    continue

                if uid not in raw:
                    raw[uid] = {"full_name": None, "short_name": None, "gene": None, "is_human": False}

                entry = raw[uid]

                if attr == "SPECIES" and val.upper() == "HUMAN":
                    entry["is_human"] = True
                elif attr == "FULL_NAME" and entry["full_name"] is None:
                    val = val.split("{")[0].strip()
                    if val and len(val) <= 100:
                        entry["full_name"] = val
                elif attr == "SHORT_NAME" and entry["short_name"] is None:
                    val = val.split("{")[0].strip()
                    if val:
                        entry["short_name"] = val
                elif attr == "NAME" and entry["gene"] is None:
                    entry["gene"] = val

    except FileNotFoundError:
        print(f"   ❌ Protein TSV not found at {META_PROTEIN_PATH}. Falling back to accession IDs.")
        return
    except Exception as exc:
        print(f"   ❌ Error reading protein TSV: {exc}")
        return

    human_count = 0
    for uid, entry in raw.items():
        if not entry["is_human"]:
            continue
        full_name  = entry["full_name"]
        short_name = entry["short_name"]
        gene       = entry["gene"]

        if full_name:
            display = full_name
        elif short_name:
            display = short_name
        elif gene:
            display = gene
        else:
            display = uid

        PROTEIN_NAME_INDEX[uid] = {
            "id": uid, "name": display,
            "full_name": full_name, "gene": gene,
        }
        human_count += 1

    print(f"   ✅ Indexed {human_count:,} HUMAN proteins (out of {len(raw):,} total in TSV)")


# ============================================================================
# LOCAL PROTEIN NAME LOOKUP — replaces all UniProt API helpers
# ============================================================================

def get_protein_name_from_index(uniprot_id: str) -> dict:
    """
    Synchronous local lookup. Returns same dict shape as old UniProt functions:
        {'id', 'name', 'full_name', 'gene'}
    """
    if uniprot_id in PROTEIN_NAME_INDEX:
        return PROTEIN_NAME_INDEX[uniprot_id]
    return {"id": uniprot_id, "name": uniprot_id, "full_name": None, "gene": None}


# Async wrappers — identical signatures to original UniProt functions
# so every existing `await` call site works without modification.

async def fetch_uniprot_name(uniprot_id: str) -> str:
    """Returns best display name string. No network call."""
    info = get_protein_name_from_index(uniprot_id)
    return info.get("full_name") or info.get("gene") or uniprot_id


async def fetch_uniprot_name_enhanced(uniprot_id: str, http_client=None) -> dict:
    """Returns same dict shape as original. No network call."""
    return get_protein_name_from_index(uniprot_id)


async def fetch_multiple_uniprot_names(uniprot_ids: List[str]) -> Dict[str, str]:
    """Returns { uniprot_id: display_name_str }. No network call."""
    return {uid: get_protein_name_from_index(uid)["name"]
            for uid in uniprot_ids if uid and uid != "-"}


async def fetch_multiple_uniprot_names_enhanced(uniprot_ids: List[str]) -> Dict[str, Dict]:
    """
    Returns { uniprot_id: {'id', 'name', 'full_name', 'gene'} }. No network call.
    Accepts any UniProt accession (6–10 char, letter then digit).
    """
    results = {}
    valid_ids = [
        uid for uid in uniprot_ids
        if uid and uid != "-" and 6 <= len(uid) <= 10 and uid[0].isalpha() and uid[1].isdigit()
    ]

    in_index = 0
    for uid in valid_ids:
        results[uid] = get_protein_name_from_index(uid)
        if uid in PROTEIN_NAME_INDEX:
            in_index += 1

    resolved = sum(
        1 for uid in valid_ids
        if results.get(uid, {}).get("full_name") or results.get(uid, {}).get("gene")
    )
    print(
        f"   ✅ Resolved {resolved}/{len(valid_ids)} protein names "
        f"({in_index} from local index, {len(valid_ids) - in_index} fallback to accession)"
    )
    return results


# ============================================================================
# CONTEXT GRAPH BUILDER
# ============================================================================

def build_context_graph(disease_id: str, max_hops: int = 3) -> nx.DiGraph:
    """
    Build biological context graph using ctx_adj — the pre-filtered typed
    adjacency dict built once at startup. No df_links scans at query time.
    Per-type neighbor caps prevent hub-node explosion.

        pathway/complex/GO  → 150 neighbors max
        protein             → 80  neighbors max
        tissue/cell/other   → 40  neighbors max

    Cached per disease in CONTEXT_GRAPH_CACHE.
    """
    if disease_id in CONTEXT_GRAPH_CACHE:
        print(f"   ⚡ Using cached context graph for {disease_id}")
        return CONTEXT_GRAPH_CACHE[disease_id]

    print(f"\n   🌐 Building context graph: {disease_id} (max_hops={max_hops})")

    CAPS = {
        'disease':            99999,
        'pathway':            150,
        'complex':            150,
        'biological_process': 150,
        'molecular_function': 150,
        'cellular_component': 150,
        'genetic_disorder':   150,
        'gene/protein':       80,
        'tissue':             40,
        'cell':               40,
        'other':              40,
    }

    visited         = {disease_id}
    current_level   = [disease_id]
    collected_edges = []   # list of (src, tgt, rel)

    for hop in range(1, max_hops + 1):
        next_level = []
        for node in current_level:
            node_type = id_to_type.get(node, 'other')
            cap       = CAPS.get(node_type, 40)
            neighbors = ctx_adj.get(node, {})

            if len(neighbors) > cap:
                sorted_nbrs = sorted(
                    neighbors.items(),
                    key=lambda kv: CONTEXT_TYPE_WEIGHTS.get(id_to_type.get(kv[0], 'other'), 0.2),
                    reverse=True
                )[:cap]
            else:
                sorted_nbrs = list(neighbors.items())

            for neighbor, rel in sorted_nbrs:
                if neighbor in visited:
                    continue
                # Skip non-human proteins — only HUMAN proteins (in PROTEIN_NAME_INDEX) allowed
                if id_to_type.get(neighbor, 'other') == 'gene/protein' and neighbor not in PROTEIN_NAME_INDEX:
                    continue
                visited.add(neighbor)
                next_level.append(neighbor)
                collected_edges.append((node, neighbor, rel))

        current_level = next_level
        print(f"      hop {hop}: {len(visited):,} nodes, {len(collected_edges):,} edges")
        if not current_level:
            break

    G = nx.DiGraph()
    G.add_node(disease_id)
    for src, tgt, rel in collected_edges:
        G.add_edge(src, tgt, relation=rel)

    type_dist = defaultdict(int)
    for n in G.nodes():
        type_dist[id_to_type.get(n, 'other')] += 1
    print(f"   ✅ Context graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    for etype, cnt in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"      {etype}: {cnt:,}")

    CONTEXT_GRAPH_CACHE[disease_id] = G
    return G


def get_context_score(disease_id: str, protein_id: str, context_graph: nx.DiGraph) -> float:
    """
    Score a protein based on how richly it is connected to the disease
    through biological context nodes in the context graph.

    Score = sum of CONTEXT_TYPE_WEIGHTS for each unique context node type
            that bridges disease ↔ protein, capped at 40.
    """
    if not context_graph.has_node(protein_id):
        return 0.0

    disease_ctx = set(context_graph.successors(disease_id)) | set(context_graph.predecessors(disease_id))
    protein_ctx = set(context_graph.successors(protein_id)) | set(context_graph.predecessors(protein_id))
    shared_ctx  = (disease_ctx & protein_ctx) - {disease_id, protein_id}

    if not shared_ctx:
        return 0.0

    type_contributions = defaultdict(float)
    for ctx_node in shared_ctx:
        ctx_type = id_to_type.get(ctx_node, 'other')
        type_contributions[ctx_type] += CONTEXT_TYPE_WEIGHTS.get(ctx_type, 0.2)

    return min(round(sum(type_contributions.values()), 2), 40.0)


# ============================================================================
# CACHING SYSTEM
# ============================================================================

def save_cache():
    """Save processed data to cache"""
    cache_data = {
        'id_to_name':           id_to_name,
        'id_to_type':           id_to_type,
        'id_to_raw_name':       id_to_raw_name,
        'disease_name_to_id':   disease_name_to_id,
        'disease_ids':          disease_ids,
        'diseases_in_kg':       diseases_in_kg,
        'drug_ids':             drug_ids,
        'protein_ids':          protein_ids,
        'disease_links_index':  disease_links_index,
        'drug_connections':     drug_connections,
        'entity_neighbors':     entity_neighbors,
        'edge_lookup':          edge_lookup,
        'ctx_adj':              ctx_adj,
        'df_links':             df_links.to_dict('records')
    }
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Cache saved to {CACHE_PATH}")


def load_cache():
    """Load preprocessed data from cache"""
    global df_links, id_to_name, id_to_type, id_to_raw_name
    global disease_name_to_id, disease_ids, diseases_in_kg, drug_ids, protein_ids
    global disease_links_index, drug_connections, entity_neighbors, edge_lookup, ctx_adj
    global PROTEIN_NAME_INDEX, TARGET_CACHE, METAPATH_CACHE

    if not os.path.exists(CACHE_PATH):
        return False

    try:
        print("📦 Loading from cache...")
        with open(CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)

        id_to_name          = cache_data['id_to_name']
        id_to_type          = cache_data['id_to_type']
        id_to_raw_name      = cache_data['id_to_raw_name']
        disease_name_to_id  = cache_data['disease_name_to_id']
        disease_ids         = cache_data['disease_ids']
        diseases_in_kg      = cache_data['diseases_in_kg']
        drug_ids            = cache_data['drug_ids']
        protein_ids         = cache_data.get('protein_ids', set())
        disease_links_index = cache_data['disease_links_index']
        drug_connections    = cache_data['drug_connections']
        entity_neighbors    = cache_data['entity_neighbors']
        df_links            = pd.DataFrame(cache_data['df_links'])

        edge_lookup_loaded = cache_data.get('edge_lookup', {})
        if edge_lookup_loaded:
            edge_lookup.update(edge_lookup_loaded)
            print(f"   ✅ Edge lookup loaded: {len(edge_lookup):,} entries")
        else:
            print("   ⚠️  Edge lookup not in cache — will rebuild")

        ctx_adj_loaded = cache_data.get('ctx_adj', {})
        if ctx_adj_loaded:
            ctx_adj.update(ctx_adj_loaded)
            print(f"   ✅ Context adjacency loaded: {len(ctx_adj):,} nodes")
        else:
            print("   ⚠️  ctx_adj not in cache — will rebuild")

        # Always rebuild protein name index from TSV (fresh, never stale)
        build_protein_name_index()

        # Rebuild edge_lookup and ctx_adj if missing (old cache format)
        if not edge_lookup or not ctx_adj:
            print("   🔨 Rebuilding edge lookup and ctx_adj from df_links...")
            _rebuild_edge_and_ctx_adj()

        # Clear session caches on load
        TARGET_CACHE.clear()
        METAPATH_CACHE.clear()
        CONTEXT_GRAPH_CACHE.clear()
        print("   ♻️ Cleared session caches (data reloaded)")

        print(f"✅ Cache loaded successfully!")
        print(f"   📊 {len(id_to_name):,} entities")
        print(f"   📊 {len(df_links):,} triples")
        print(f"   📊 {len(disease_ids):,} diseases ({len(diseases_in_kg):,} in KG)")
        print(f"   📊 {len(drug_ids):,} drugs")
        print(f"   📊 {len(protein_ids):,} proteins")
        return True
    except Exception as e:
        print(f"⚠️ Cache load failed: {e}")
        return False


def _rebuild_edge_and_ctx_adj():
    """Rebuild edge_lookup and ctx_adj from df_links (used when cache is missing these)."""
    src_arr = df_links['source'].astype(str).values
    tgt_arr = df_links['target'].astype(str).values
    rel_arr = df_links['edge_type'].astype(str).values
    for s, t, r in zip(src_arr, tgt_arr, rel_arr):
        edge_lookup[(s, t)] = r
        edge_lookup[(t, s)] = r
        s_type = id_to_type.get(s, 'other')
        t_type = id_to_type.get(t, 'other')
        t_is_non_human_protein = (t_type == 'gene/protein' and t not in PROTEIN_NAME_INDEX)
        s_is_non_human_protein = (s_type == 'gene/protein' and s not in PROTEIN_NAME_INDEX)
        if t_type not in ('drug', 'disease') and not t_is_non_human_protein:
            if s not in ctx_adj: ctx_adj[s] = {}
            ctx_adj[s][t] = r
        if s_type not in ('drug', 'disease') and not s_is_non_human_protein:
            if t not in ctx_adj: ctx_adj[t] = {}
            ctx_adj[t][s] = r
    print(f"   ✅ Rebuilt: {len(edge_lookup):,} edge entries, {len(ctx_adj):,} ctx nodes")


# ============================================================================
# DATA LOADING
# ============================================================================

def detect_id_name(df):
    """Auto-detect ID and name columns"""
    id_col   = next((c for c in df.columns if "id"     in c.lower() or "accession" in c.lower()), df.columns[0])
    name_col = next((c for c in df.columns if "name"   in c.lower() or "symbol"    in c.lower() or "label" in c.lower()), df.columns[-1])
    return id_col, name_col


def infer_entity_type(entity_id: str) -> str:
    """
    Infer entity type from ID prefix patterns as defined in BioKG schema.
    """
    e  = entity_id.strip()
    eu = e.upper()

    # Drug
    if eu.startswith('DB') and len(e) >= 6 and e[2:].isdigit():
        return 'drug'
    if eu.startswith(('CHEMBL', 'CHEBI:', 'PUBCHEM:', 'CID')):
        return 'drug'

    # Disease (MeSH)
    if eu.startswith('MESH:'):
        return 'disease'
    if eu.startswith('D') and len(e) >= 4 and e[1:].isdigit():
        return 'disease'
    if eu.startswith('C') and len(e) >= 4 and e[1:].isdigit():
        return 'disease'

    # Genetic disorder (OMIM)
    if eu.startswith(('MIM:', 'OMIM:')):
        return 'genetic_disorder'
    if e.isdigit() and len(e) == 6:
        return 'genetic_disorder'

    # GO terms (default to biological_process; overridden by edge type in build_entity_lookups)
    if eu.startswith('GO:') or eu.startswith('GO_'):
        return 'biological_process'

    # Pathway
    if eu.startswith('R-') and '-' in e[2:]:
        parts = e.split('-')
        if len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit():
            return 'complex'
        return 'pathway'
    if eu.startswith(('HSA', 'MAP')) and e[3:].isdigit():
        return 'pathway'
    if eu.startswith('SMP') and e[3:].isdigit():
        return 'pathway'
    if 'PATHWAY' in eu or eu.startswith(('REACT:', 'KEGG:', 'WP:')):
        return 'pathway'

    # Complex
    if eu.startswith('COMPLEX:') or eu.startswith('CPX-'):
        return 'complex'

    # Cell line
    if eu.startswith('CVCL_'):
        return 'cell'

    # UniProt protein (6–10 char, letter then digit)
    if ':' not in e and 6 <= len(e) <= 10 and e[0].isalpha() and e[1].isdigit():
        return 'gene/protein'

    return 'other'


def load_data():
    """Load BioKG data with improved name resolution"""
    global df_links, meta_disease, meta_protein, meta_drug
    global id_to_name, id_to_type, id_to_raw_name
    global disease_name_to_id, disease_ids, diseases_in_kg, drug_ids, protein_ids
    global PROTEIN_NAME_INDEX, TARGET_CACHE, METAPATH_CACHE

    print("📄 Loading BioKG data files...")

    df_links = pd.read_csv(
        LINKS_PATH, sep="\t",
        names=["source", "edge_type", "target"],
        dtype={'source': 'category', 'edge_type': 'category', 'target': 'category'}
    )

    meta_disease = pd.read_csv(META_DISEASE_PATH, sep="\t", low_memory=False, header=0)
    meta_protein = pd.read_csv(META_PROTEIN_PATH, sep="\t", low_memory=False, header=0)
    meta_drug    = pd.read_csv(META_DRUG_PATH,    sep="\t", low_memory=False, header=0)

    for df_meta, name in [(meta_disease, "disease"), (meta_protein, "protein"), (meta_drug, "drug")]:
        if df_meta.columns[0].startswith(('D','MONDO','DOID','MESH','P','Q','O','UNIPROT','DB','CHEMBL','DRUGBANK')):
            print(f"   ⚠️  Detected headers in first data row for {name}, fixing...")
            df_meta.columns = df_meta.iloc[0].tolist()
            if name == "disease":
                meta_disease = df_meta[1:].reset_index(drop=True)
            elif name == "protein":
                meta_protein = df_meta[1:].reset_index(drop=True)
            else:
                meta_drug = df_meta[1:].reset_index(drop=True)

    meta_disease.columns = [str(c).lower().strip() for c in meta_disease.columns]
    meta_protein.columns = [str(c).lower().strip() for c in meta_protein.columns]
    meta_drug.columns    = [str(c).lower().strip() for c in meta_drug.columns]

    print(f"✅ Loaded {len(df_links):,} triples")
    print(f"   📊 Diseases: {len(meta_disease):,}")
    print(f"   📊 Proteins: {len(meta_protein):,}")
    print(f"   📊 Drugs:    {len(meta_drug):,}")

    # Build protein name index from TSV BEFORE entity lookups
    build_protein_name_index()

    build_entity_lookups()
    build_fast_indexes()

    TARGET_CACHE.clear()
    METAPATH_CACHE.clear()
    CONTEXT_GRAPH_CACHE.clear()
    print("   ♻️ Cleared session caches (fresh data loaded)")


def build_entity_lookups():
    """Build comprehensive lookups for ALL entity types"""
    global id_to_name, id_to_type, id_to_raw_name
    global disease_name_to_id, disease_ids, diseases_in_kg, drug_ids, protein_ids

    print("\n🔨 Building entity lookups...")

    # DISEASES
    print(f"\n📊 Processing Diseases...")
    disease_id_col, disease_name_col = detect_id_name(meta_disease)
    print(f"   ✓ Detected columns: ID='{disease_id_col}', Name='{disease_name_col}'")

    for _, row in meta_disease.iterrows():
        d_id   = str(row[disease_id_col]).strip()
        d_name = str(row[disease_name_col]) if not pd.isna(row[disease_name_col]) else d_id
        if d_name.upper() in ['NAME','LABEL','TITLE','DISEASE','TYPE',''] or d_name == d_id:
            d_name = d_id
        id_to_raw_name[d_id]                              = d_name
        id_to_name[d_id]                                  = d_name
        id_to_type[d_id]                                  = 'disease'
        disease_name_to_id[d_name.lower()]                = d_id
        disease_name_to_id[d_name.lower().replace(',','')] = d_id
        disease_name_to_id[d_name.lower().replace('-',' ')] = d_id
        disease_ids.add(d_id)

    print(f"   ✓ Processed {len(disease_ids):,} diseases")

    # PROTEINS — use PROTEIN_NAME_INDEX (HUMAN only, names from TSV)
    print(f"\n📊 Processing Proteins (HUMAN only, names from local TSV)...")
    for uid, info in PROTEIN_NAME_INDEX.items():
        id_to_raw_name[uid] = info["gene"] or uid
        id_to_name[uid]     = info["name"]
        id_to_type[uid]     = "gene/protein"
        protein_ids.add(uid)
    print(f"   ✓ Processed {len(protein_ids):,} HUMAN proteins")

    # DRUGS
    print(f"\n📊 Processing Drugs...")
    drug_id_col, drug_name_col = detect_id_name(meta_drug)
    print(f"   ✓ Detected columns: ID='{drug_id_col}', Name='{drug_name_col}'")
    for _, row in meta_drug.iterrows():
        dr_id   = str(row[drug_id_col]).strip()
        dr_name = str(row[drug_name_col]) if not pd.isna(row[drug_name_col]) else dr_id
        id_to_raw_name[dr_id] = dr_name
        id_to_name[dr_id]     = dr_name
        id_to_type[dr_id]     = 'drug'
        drug_ids.add(dr_id)
    print(f"   ✓ Processed {len(drug_ids):,} drugs")

    # INFER TYPES for entities in graph not yet classified
    print(f"\n⚡ Inferring types for entities in graph...")
    all_entities = set(df_links['source'].cat.categories) | set(df_links['target'].cat.categories)

    # ── 1. Classify GO terms by exact edge type ──────────────────────────────
    go_type_map = {}
    edge_type_to_go = {
        'GO_BP':         'biological_process',
        'PATHWAY_GO_BP': 'biological_process',
        'GO_CC':         'cellular_component',
        'PATHWAY_GO_CC': 'cellular_component',
        'GO_MF':         'molecular_function',
        'PATHWAY_GO_MF': 'molecular_function',
    }
    go_edges = df_links[df_links['edge_type'].isin(edge_type_to_go.keys())]
    for _, row in go_edges.iterrows():
        go_type_map[str(row['target'])] = edge_type_to_go[str(row['edge_type'])]

    # ── 2. Classify Reactome complexes ───────────────────────────────────────
    complex_entities = set(df_links[df_links['edge_type'] == 'MEMBER_OF_COMPLEX']['target'].unique())

    # ── 3. Classify tissues and cells ────────────────────────────────────────
    tissue_entities = set(df_links[df_links['edge_type'] == 'PROTEIN_EXPRESSED_IN']['target'].unique())
    cell_entities   = set(df_links[df_links['edge_type'] == 'PART_OF_TISSUE']['source'].unique())
    tissue_entities.update(df_links[df_links['edge_type'] == 'PART_OF_TISSUE']['target'].unique())

    # ── 4. Classify genetic disorders ────────────────────────────────────────
    genetic_disorder_entities = set(df_links[df_links['edge_type'].isin(
        ['RELATED_GENETIC_DISORDER', 'DISEASE_GENETIC_DISORDER']
    )]['target'].unique())

    # ── 5. Load pathway names from metadata TSV ──────────────────────────────
    _bracket_re     = re.compile(r'[\(\[]')
    _lipid_prefixes = ('CL(','TG(','DG(','MG(','PC(','PE(','PS(','PI(','SM(','PG(',
                       'LPC(','LPE(','LPS(','FA(','CE(','Cer(','HexCer(')
    pathway_names_loaded = 0
    try:
        with open(META_PATHWAY_PATH, "r", encoding="utf-8") as fh:
            raw_lines = [l.rstrip("\n") for l in fh]

        def _accept_pathway_name(pid, name):
            if not name or not pid:
                return False
            if _bracket_re.search(name):
                return False
            if any(name.upper().startswith(p.upper()) for p in _lipid_prefixes):
                return False
            return True

        # Try tab-separated format first
        tsv_loaded = 0
        for line in raw_lines:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                pid_raw, attr, name = parts[0].strip(), parts[1].strip().upper(), parts[2].strip()
                if attr == "NAME" and _accept_pathway_name(pid_raw, name):
                    id_to_raw_name[pid_raw] = name
                    id_to_name[pid_raw]     = name
                    id_to_type[pid_raw]     = 'pathway'
                    tsv_loaded += 1

        if tsv_loaded > 0:
            pathway_names_loaded = tsv_loaded
            print(f"   ✅ Loaded {pathway_names_loaded:,} pathway names (tab-separated)")
        else:
            # Fallback: 3-line block format
            non_empty = [l for l in raw_lines if l.strip()]
            i = 0
            while i < len(non_empty):
                pid_raw = non_empty[i].strip()
                if i + 1 < len(non_empty) and non_empty[i+1].strip().upper() == "NAME":
                    if i + 2 < len(non_empty):
                        name = non_empty[i+2].strip()
                        i += 3
                        if i < len(non_empty) and non_empty[i].strip().isdigit():
                            i += 1
                        if _accept_pathway_name(pid_raw, name):
                            id_to_raw_name[pid_raw] = name
                            id_to_name[pid_raw]     = name
                            id_to_type[pid_raw]     = 'pathway'
                            pathway_names_loaded += 1
                    else:
                        i += 2
                else:
                    i += 1
            print(f"   ✅ Loaded {pathway_names_loaded:,} pathway names (block format)")
    except FileNotFoundError:
        print(f"   ⚠️  Pathway metadata not found at {META_PATHWAY_PATH}")
    except Exception as exc:
        print(f"   ⚠️  Could not load pathway names: {exc}")

    # ── 6. Classify all remaining untyped entities ────────────────────────────
    inferred_count = 0
    for entity_id in all_entities:
        if entity_id in id_to_type:
            continue
        if entity_id in go_type_map:
            inferred_type = go_type_map[entity_id]
        elif entity_id in complex_entities:
            inferred_type = 'complex'
        elif entity_id in genetic_disorder_entities:
            inferred_type = 'genetic_disorder'
        elif entity_id in tissue_entities:
            inferred_type = 'tissue'
        elif entity_id in cell_entities:
            inferred_type = 'cell'
        else:
            inferred_type = infer_entity_type(entity_id)

        id_to_type[entity_id] = inferred_type
        if entity_id not in id_to_raw_name:
            id_to_raw_name[entity_id] = entity_id
            id_to_name[entity_id]     = entity_id
        if inferred_type == 'drug':
            drug_ids.add(entity_id)
        elif inferred_type == 'gene/protein':
            protein_ids.add(entity_id)
        inferred_count += 1

    print(f"   ✓ Inferred types for {inferred_count:,} additional entities")

    diseases_in_kg.update(
        set(df_links[df_links['source'].isin(disease_ids)]['source'].unique()) |
        set(df_links[df_links['target'].isin(disease_ids)]['target'].unique())
    )

    type_counts = defaultdict(int)
    for entity_type in id_to_type.values():
        type_counts[entity_type] += 1

    print(f"\n✅ Built lookup for {len(id_to_name):,} entities")
    print(f"✅ Diseases with relationships: {len(diseases_in_kg):,}")
    print(f"\n📊 Entity Type Distribution:")
    for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {entity_type}: {count:,}")


def build_fast_indexes():
    """Build precomputed indexes including O(1) edge lookup and pre-filtered context adjacency"""
    global diseases_in_kg, disease_links_index, drug_connections, entity_neighbors, edge_lookup, ctx_adj

    print("\n⚡ Building fast lookup indexes...")

    sources_grouped = df_links.groupby('source', observed=True)['target'].apply(set).to_dict()
    targets_grouped = df_links.groupby('target', observed=True)['source'].apply(set).to_dict()
    all_entities    = set(df_links['source'].cat.categories) | set(df_links['target'].cat.categories)

    for entity_id in all_entities:
        entity_neighbors[entity_id] = {
            'targets': sources_grouped.get(entity_id, set()),
            'sources': targets_grouped.get(entity_id, set())
        }

    # Build O(1) edge type lookup (vectorized)
    print("   ⚡ Building edge type lookup (vectorized)...")
    src_arr = df_links['source'].astype(str).values
    tgt_arr = df_links['target'].astype(str).values
    rel_arr = df_links['edge_type'].astype(str).values
    for s, t, r in zip(src_arr, tgt_arr, rel_arr):
        edge_lookup[(s, t)] = r
        edge_lookup[(t, s)] = r
    print(f"   ✅ Edge lookup: {len(edge_lookup):,} entries")

    # Build pre-filtered typed adjacency dict for fast context-graph BFS
    print("   ⚡ Building context adjacency index (no drugs/diseases/non-human proteins)...")
    ctx_adj = {}
    for s, t, r in zip(src_arr, tgt_arr, rel_arr):
        s_type = id_to_type.get(s, 'other')
        t_type = id_to_type.get(t, 'other')
        # Exclude drugs and diseases as traversal nodes
        # Exclude non-human proteins (not in PROTEIN_NAME_INDEX)
        t_is_non_human_protein = (t_type == 'gene/protein' and t not in PROTEIN_NAME_INDEX)
        s_is_non_human_protein = (s_type == 'gene/protein' and s not in PROTEIN_NAME_INDEX)
        if t_type not in ('drug', 'disease') and not t_is_non_human_protein:
            if s not in ctx_adj: ctx_adj[s] = {}
            ctx_adj[s][t] = r
        if s_type not in ('drug', 'disease') and not s_is_non_human_protein:
            if t not in ctx_adj: ctx_adj[t] = {}
            ctx_adj[t][s] = r
    print(f"   ✅ Context adjacency index: {len(ctx_adj):,} nodes")

    for disease_id in disease_ids:
        if disease_id in entity_neighbors:
            neighbors = entity_neighbors[disease_id]['targets'] | entity_neighbors[disease_id]['sources']
            if neighbors:
                diseases_in_kg.add(disease_id)
                disease_links_index[disease_id] = neighbors

    for drug_id in drug_ids:
        if drug_id in entity_neighbors:
            drug_connections[drug_id] = entity_neighbors[drug_id]['targets'] | entity_neighbors[drug_id]['sources']

    print(f"✅ Indexed {len(diseases_in_kg):,} diseases with connections")
    print(f"✅ Indexed {len(drug_connections):,} drugs")
    print(f"✅ Indexed {len(entity_neighbors):,} entity neighborhoods")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def should_explore_node(node_type: str, hop_distance: int) -> bool:
    """Decide whether to explore a node based on its type and hop distance."""
    priority_types = {
        1: {'gene/protein', 'pathway', 'genetic_disorder', 'complex'},
        2: {'gene/protein', 'pathway', 'complex', 'biological_process',
            'molecular_function', 'cellular_component', 'tissue'},
        3: {'gene/protein'},
    }
    return node_type in priority_types.get(hop_distance, {'gene/protein'})

def get_max_neighbors_per_hop(hop: int) -> int:
    """Limit neighbors explored per hop to control explosion"""
    limits = {1: 100, 2: 50, 3: 25, 4: 10}
    return limits.get(hop, 5)


@lru_cache(maxsize=2000)
def find_disease(disease_query: str):
    """Find disease ID using fuzzy matching"""
    query_lower = disease_query.lower().strip()

    if query_lower in disease_name_to_id:
        d_id = disease_name_to_id[query_lower]
        if d_id in diseases_in_kg:
            return d_id, id_to_name[d_id]

    query_clean = query_lower.replace(',', '').replace('-', ' ')
    if query_clean in disease_name_to_id:
        d_id = disease_name_to_id[query_clean]
        if d_id in diseases_in_kg:
            return d_id, id_to_name[d_id]

    disease_names_with_kg = [name for name, did in disease_name_to_id.items()
                              if did in diseases_in_kg]
    matches = get_close_matches(query_lower, disease_names_with_kg, n=1, cutoff=0.4)
    if matches:
        d_id = disease_name_to_id[matches[0]]
        return d_id, id_to_name[d_id]

    for disease_name, disease_id in disease_name_to_id.items():
        if query_lower in disease_name or disease_name in query_lower:
            if disease_id in diseases_in_kg:
                return disease_id, id_to_name[disease_id]

    return None, None


async def extract_targets(disease_id: str, max_targets: int = 100, max_hops: int = 3):
    """
    Extract TOP protein targets using the full biological context graph.

    Context graph traverses ALL BioKG entity types (pathway, GO BP/MF/CC, complex,
    genetic disorder, tissue, cell, protein) except drugs and other diseases.

    Scoring per protein:
      base_score    — hop distance in context graph (closer = higher)
      direct_score  — direct disease-protein edges (1-hop bonus)
      context_score — shared biological context nodes weighted by type
      shared_score  — legacy shared-neighbour score (safety net for sparse diseases)
    """
    cache_key = f"{disease_id}_{max_targets}_{max_hops}"
    if cache_key in TARGET_CACHE:
        print(f"   ⚡ Using cached targets for {disease_id}")
        return TARGET_CACHE[cache_key]

    if disease_id not in disease_links_index:
        return []

    print(f"   🔍 Extracting targets via context graph for disease: {disease_id}")

    # ── Step 1: build (or retrieve cached) context graph ─────────────────────
    context_graph = build_context_graph(disease_id, max_hops=max_hops)

    # ── Step 2: collect HUMAN proteins from context graph ────────────────────
    proteins_by_hop = {i: set() for i in range(1, max_hops + 1)}

    try:
        hop_distances = dict(nx.single_source_shortest_path_length(
            context_graph.to_undirected(), disease_id, cutoff=max_hops
        ))
    except Exception:
        hop_distances = {}

    for node in context_graph.nodes():
        if id_to_type.get(node, 'other') == 'gene/protein' and node in PROTEIN_NAME_INDEX:
            dist = hop_distances.get(node, max_hops)
            hop  = min(dist, max_hops)
            if hop >= 1:
                proteins_by_hop[hop].add(node)

    for hop in range(1, max_hops + 1):
        print(f"   ✅ Hop {hop}: {len(proteins_by_hop[hop]):,} HUMAN proteins in context graph")

    all_protein_ids = []
    for hop in range(1, max_hops + 1):
        all_protein_ids.extend([(pid, hop) for pid in proteins_by_hop[hop]])

    if not all_protein_ids:
        return []

    # ── Step 3: resolve names from local index ────────────────────────────────
    protein_ids_only = [pid for pid, _ in all_protein_ids]
    print(f"   📖 Resolving names for {len(protein_ids_only)} HUMAN proteins from local index...")
    uniprot_data = await fetch_multiple_uniprot_names_enhanced(protein_ids_only)

    disease_edges_df  = df_links[(df_links['source'] == disease_id) | (df_links['target'] == disease_id)]
    disease_neighbors = disease_links_index.get(disease_id, set())

    # ── Step 4: score each protein ────────────────────────────────────────────
    targets = []
    for protein_id, hop_distance in all_protein_ids:
        protein_info = uniprot_data.get(protein_id, {})
        full_name    = protein_info.get('full_name')
        gene_name    = protein_info.get('gene') or protein_info.get('name')

        if full_name and len(full_name) <= 100:
            display_name = full_name
        elif gene_name and gene_name != protein_id and len(gene_name) > 1:
            display_name = gene_name
        else:
            display_name = protein_id

        # Base score: hop distance
        base_score = {1: 70, 2: 40, 3: 20}.get(hop_distance, 3)

        # Direct connection bonus (1-hop only)
        direct_score = 0
        if hop_distance == 1:
            direct_edges = len(disease_edges_df[
                (disease_edges_df['target'] == protein_id) |
                (disease_edges_df['source'] == protein_id)
            ])
            direct_score = min(direct_edges * 10, 20)

        # Context score: shared biological context nodes
        context_score = get_context_score(disease_id, protein_id, context_graph)

        # Legacy shared-neighbour score (safety net)
        protein_neighbors = entity_neighbors.get(protein_id, {})
        protein_all_nbrs  = protein_neighbors.get('targets', set()) | protein_neighbors.get('sources', set())
        filtered_shared   = {n for n in (disease_neighbors & protein_all_nbrs)
                             if id_to_type.get(n, 'other') not in ('drug', 'disease')}
        shared_score = min(len(filtered_shared) * max(1, 4 - hop_distance), 20)

        importance_score = min(base_score + direct_score + context_score + shared_score, 100)

        targets.append({
            'id':                  protein_id,
            'name':                display_name,
            'uniprot_id':          protein_id,
            'gene_name':           gene_name,
            'full_name':           full_name,
            'type':                'gene/protein',
            'score':               round(importance_score, 2),
            'hop':                 hop_distance,
            'context_score':       context_score,
            'from_knowledge_graph': True,
        })

    targets.sort(key=lambda x: x['score'], reverse=True)
    top_targets = targets[:max_targets]

    TARGET_CACHE[cache_key] = top_targets
    print(f"   🎯 Returning top {len(top_targets)} HUMAN targets (context-graph scored)")
    return top_targets


def calculate_node_importance(G):
    """Calculate node importance using centrality metrics"""
    importance = {}
    try:
        degree_cent  = nx.degree_centrality(G)
        between_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        try:
            close_cent = nx.closeness_centrality(G)
        except Exception:
            close_cent = {node: 0 for node in G.nodes()}

        for node in G.nodes():
            node_type   = id_to_type.get(node, 'other')
            type_weight = {
                'disease': 1.0, 'drug': 0.9, 'gene/protein': 0.8,
                'pathway': 0.7, 'biological_process': 0.6, 'other': 0.5
            }.get(node_type, 0.5)
            deg = degree_cent.get(node, 0)
            bet = between_cent.get(node, 0)
            clo = close_cent.get(node, 0)
            importance[node] = (deg * 0.4 + bet * 0.4 + clo * 0.2) * type_weight

        max_score = max(importance.values()) if importance else 1
        if max_score > 0:
            importance = {k: v / max_score for k, v in importance.items()}
    except Exception as e:
        print(f"   ⚠️ Centrality calculation failed: {e}, using degree only")
        degree_cent = dict(G.degree())
        max_degree  = max(degree_cent.values()) if degree_cent else 1
        importance  = {node: degree_cent.get(node, 0) / max_degree for node in G.nodes()}
    return importance


def extract_metapaths(disease_id, max_paths=200):
    """Extract Disease -> Intermediate -> Drug paths with attention scores"""
    disease_rels = df_links[(df_links['source'] == disease_id) | (df_links['target'] == disease_id)]
    metapaths    = []
    path_frequencies = defaultdict(int)

    for _, rel in disease_rels.head(max_paths).iterrows():
        intermediate_id   = rel['target'] if rel['source'] == disease_id else rel['source']
        intermediate_type = id_to_type.get(intermediate_id, 'other')
        second_hop        = df_links[(df_links['source'] == intermediate_id) | (df_links['target'] == intermediate_id)]

        for _, rel2 in second_hop.iterrows():
            target_id   = rel2['target'] if rel2['source'] == intermediate_id else rel2['source']
            target_type = id_to_type.get(target_id, 'other')
            if target_type == 'drug':
                path = {
                    'disease': disease_id,
                    'disease_name': id_to_name.get(disease_id, disease_id),
                    'intermediate': intermediate_id,
                    'intermediate_name': id_to_name.get(intermediate_id, intermediate_id),
                    'intermediate_type': intermediate_type,
                    'drug': target_id,
                    'drug_name': id_to_name.get(target_id, target_id),
                    'relation1': rel['edge_type'],
                    'relation2': rel2['edge_type'],
                    'metapath': f"Disease→{intermediate_type}→Drug"
                }
                metapaths.append(path)
                path_frequencies[path['metapath']] += 1

    max_freq = max(path_frequencies.values()) if path_frequencies else 1
    for path in metapaths:
        path['attention'] = path_frequencies[path['metapath']] / max_freq
    return metapaths, path_frequencies


async def build_enhanced_subgraph(disease_id: str, max_nodes: int = 100, max_hops: int = 3):
    """
    Build a rich visualisation subgraph using the biological context graph.

    Uses the pre-built context graph (all entity types, no drugs/diseases as
    intermediates) as the traversal backbone. Top predicted protein targets
    are always included and boosted visually. Context graph traversal is
    reflected in the node types and colours shown.
    """
    if disease_id not in disease_links_index:
        return {'nodes': [], 'links': [], 'statistics': {}, 'metapaths': []}

    print(f"   🔨 Building context-graph subgraph for: {disease_id} (max_nodes={max_nodes})")

    context_graph = build_context_graph(disease_id, max_hops=max_hops)
    top_targets   = await extract_targets(disease_id, max_targets=25, max_hops=max_hops)
    target_protein_ids = {t['id'] for t in top_targets}

    print(f"   🎯 Including {len(target_protein_ids)} top HUMAN targets in subgraph")

    G        = nx.DiGraph()
    hop_info = {disease_id: 0}
    node_importance_boost = {}

    G.add_node(disease_id)

    # Pre-compute hop distances for all context graph nodes in one BFS pass
    try:
        all_hop_distances = dict(nx.single_source_shortest_path_length(
            context_graph.to_undirected(), disease_id, cutoff=max_hops
        ))
    except Exception:
        all_hop_distances = {}
    target_hop_distances = {pid: all_hop_distances.get(pid, max_hops) for pid in target_protein_ids}

    nodes_per_hop_limit  = max(10, max_nodes // max_hops)
    visited              = {disease_id}
    current_level        = {disease_id}
    nodes_added_per_hop  = {i: 0 for i in range(max_hops + 1)}
    nodes_added_per_hop[0] = 1

    for hop in range(1, max_hops + 1):
        next_level     = set()
        hop_edges_added = 0

        print(f"   🔄 Subgraph hop {hop}...")

        for node in current_level:
            ctx_neighbors = (
                set(context_graph.successors(node)) |
                set(context_graph.predecessors(node))
            )

            # Sort: target proteins first, then by context type weight
            sorted_neighbors = sorted(
                ctx_neighbors,
                key=lambda n: (
                    0 if n in target_protein_ids else 1,
                    -CONTEXT_TYPE_WEIGHTS.get(id_to_type.get(n, 'other'), 0.2)
                )
            )

            for neighbor in sorted_neighbors:
                # Get edge from context graph
                if context_graph.has_edge(node, neighbor):
                    rel = context_graph[node][neighbor].get('relation', 'related_to')
                    G.add_edge(node, neighbor, relation=rel)
                elif context_graph.has_edge(neighbor, node):
                    rel = context_graph[neighbor][node].get('relation', 'related_to')
                    G.add_edge(neighbor, node, relation=rel)
                else:
                    continue

                hop_edges_added += 1

                if neighbor not in visited:
                    if neighbor in target_protein_ids or nodes_added_per_hop[hop] < nodes_per_hop_limit:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        if neighbor not in hop_info:
                            hop_info[neighbor] = target_hop_distances.get(neighbor, hop)
                        if neighbor in target_protein_ids:
                            node_importance_boost[neighbor] = 1.5
                        nodes_added_per_hop[hop] += 1

                if G.number_of_nodes() >= max_nodes:
                    break
            if G.number_of_nodes() >= max_nodes:
                break

        print(f"      ✅ Hop {hop}: {nodes_added_per_hop[hop]} nodes, {hop_edges_added} edges")
        current_level = next_level
        if G.number_of_nodes() >= max_nodes or not current_level:
            break

    if G.number_of_nodes() == 0:
        return {'nodes': [], 'links': [], 'statistics': {}, 'metapaths': []}

    print(f"   📊 Graph contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # ── Resolve protein names ─────────────────────────────────────────────────
    protein_ids_in_graph = [
        n for n in G.nodes()
        if id_to_type.get(n, 'other') == 'gene/protein' and n in PROTEIN_NAME_INDEX
    ]
    non_protein_entities = {
        n: (id_to_raw_name.get(n, n), id_to_type.get(n, 'other'))
        for n in G.nodes() if n not in protein_ids_in_graph
    }

    print(f"   🔬 {len(protein_ids_in_graph)} HUMAN proteins, {len(non_protein_entities)} context entities")

    uniprot_data = await fetch_multiple_uniprot_names_enhanced(protein_ids_in_graph)
    for uid, data in uniprot_data.items():
        full_name = data.get('full_name')
        gene_name = data.get('gene') or data.get('name')
        if full_name and len(full_name) <= 100:
            id_to_name[uid] = full_name
        elif gene_name and gene_name != uid and len(gene_name) > 1:
            id_to_name[uid] = gene_name
        else:
            id_to_name[uid] = id_to_raw_name.get(uid, uid)

    # Assign readable names to non-protein context entities
    print(f"   ⚡ Assigning readable names to {len(non_protein_entities)} context entities")
    for entity_id, (raw_name, entity_type) in non_protein_entities.items():
        existing = id_to_name.get(entity_id, entity_id)
        if existing != entity_id:
            continue  # already has a proper name from metadata load

        if raw_name != entity_id:
            clean = raw_name.replace('_', ' ').strip()
        elif entity_type == 'biological_process':
            clean = f"GO BP: {entity_id}"
        elif entity_type == 'molecular_function':
            clean = f"GO MF: {entity_id}"
        elif entity_type == 'cellular_component':
            clean = f"GO CC: {entity_id}"
        elif entity_type == 'complex':
            clean = f"Complex: {entity_id.split('-')[-1] if '-' in entity_id else entity_id}"
        elif entity_type == 'genetic_disorder':
            clean = f"OMIM: {entity_id.replace('MIM:', '').replace('OMIM:', '')}"
        elif entity_type == 'tissue':
            clean = entity_id.replace('_', ' ').title()
        elif entity_type == 'cell':
            clean = f"Cell: {entity_id}"
        else:
            clean = entity_id
        id_to_name[entity_id] = clean

    # ── Importance scoring ────────────────────────────────────────────────────
    print(f"   📈 Calculating node importance scores...")
    importance = calculate_node_importance(G)

    importance_values = list(importance.values())
    max_imp  = max(importance_values) if importance_values else 1
    min_imp  = min(importance_values) if importance_values else 0
    imp_range = max_imp - min_imp if max_imp > min_imp else 1

    normalized_importance = {}
    for node, imp in importance.items():
        norm_score    = ((imp - min_imp) / imp_range) * 100
        hop_distance  = hop_info.get(node, max_hops)
        hop_penalty   = max(1.0 - (hop_distance * 0.15), 0.3)
        boost         = node_importance_boost.get(node, 1.0)
        if node in target_protein_ids:
            boost = 1.5
        # Extra boost for biologically meaningful context types
        type_boost = 1.0 + (CONTEXT_TYPE_WEIGHTS.get(id_to_type.get(node, 'other'), 0.2) * 0.3)
        normalized_importance[node] = min(round(norm_score * hop_penalty * boost * type_boost, 2), 100)

    # ── Build output ──────────────────────────────────────────────────────────
    nodes = []
    entity_counts  = defaultdict(int)
    hop_node_counts = defaultdict(int)

    for node in G.nodes():
        name          = id_to_name.get(node, node)
        node_type     = id_to_type.get(node, 'other')
        imp_normalized = normalized_importance.get(node, 50.0)
        is_top_target  = node in target_protein_ids
        hop_distance   = hop_info.get(node, 0)

        node_size = 20 + int(imp_normalized * 0.3)
        if node == disease_id:
            node_size = 60
        elif is_top_target:
            node_size = max(node_size, 35)
        elif node_type in ('pathway', 'biological_process', 'molecular_function'):
            node_size = max(node_size, 25)

        nodes.append({
            'id':                node,
            'name':              name,
            'type':              node_type,
            'importance':        imp_normalized,
            'size':              node_size,
            'is_predicted_target': is_top_target,
            'hop_distance':      hop_distance,
            'is_disease':        node == disease_id,
        })
        entity_counts[node_type] += 1
        hop_node_counts[hop_distance] += 1

    links = []
    for source, target, data in G.edges(data=True):
        edge_label = data['relation'].replace('_', ' ').title()
        source_imp = normalized_importance.get(source, 50)
        target_imp = normalized_importance.get(target, 50)
        links.append({
            'source':    source,
            'target':    target,
            'type':      data['relation'],
            'label':     edge_label,
            'importance': round((source_imp + target_imp) / 2, 2),
        })

    proteins = sorted(
        [n for n in nodes if n.get('is_predicted_target')],
        key=lambda x: x['importance'], reverse=True
    )
    hop_distribution = {
        f"{h}-hop": hop_node_counts[h]
        for h in range(max_hops + 1) if hop_node_counts.get(h, 0) > 0
    }
    total_edges = len(links)
    avg_degree  = (2 * total_edges) / len(nodes) if nodes else 0

    print(f"   ✅ Subgraph built successfully!")
    print(f"      - Total: {len(nodes)} nodes, {len(links)} edges")
    print(f"      - Proteins: {len(proteins)} ({len([p for p in proteins if p.get('is_predicted_target')])} predicted)")
    print(f"      - Hop distribution: {hop_distribution}")

    return {
        'nodes': nodes,
        'links': links,
        'statistics': {
            'total_nodes':       len(nodes),
            'total_edges':       len(links),
            'entity_counts':     dict(entity_counts),
            'protein_count':     entity_counts.get('gene/protein', 0),
            'drug_count':        0,   # drugs excluded from context graph
            'max_hops':          max_hops,
            'hop_distribution':  hop_distribution,
            'nodes_per_hop':     dict(nodes_added_per_hop),
            'avg_degree':        round(avg_degree, 2),
            'graph_density':     round(total_edges / (len(nodes) * (len(nodes) - 1)), 4) if len(nodes) > 1 else 0,
            'predicted_targets': len(proteins),
            'context_types':     dict(entity_counts),
        },
        'top_proteins': [
            {'id': p['id'], 'name': p['name'], 'importance': p['importance'], 'hop_distance': p['hop_distance']}
            for p in proteins[:10]
        ],
        'top_drugs': [],   # excluded from context graph
        'metapaths': [],
        'disease_node': {
            'id':   disease_id,
            'name': id_to_name.get(disease_id, disease_id),
        },
    }


def generate_interactive_html(disease_id: str, disease_name: str, subgraph_data: dict) -> str:
    """
    Generate an interactive HTML file with vis.js network visualization.
    Includes context graph entity types in legend and stats.
    Returns the filename of the generated HTML file.
    """
    nodes = subgraph_data['nodes']
    links = subgraph_data['links']
    stats = subgraph_data['statistics']

    # Color mapping — full context graph entity types
    color_map = {
        'disease':'#0A2E52',
        'gene/protein':'#2D6A4F',
        'pathway':'#007B82',
        'biological_process':'#1E3A5F',
        'molecular_function':'#02A7B0',
        'cellular_component':'#5B6FA3',
        'complex':'#7B5EA7',
        'genetic_disorder':'#D32F2F',
        'tissue':'#7FB685',
        'cell':'#D7FCFE',
        'drug':'#43A047',
        'other':'#64748B',
    }

    nodes_json = []
    for node in nodes:
        node_type         = node['type']
        color             = color_map.get(node_type, '#A7A4E0')
        importance_display = round(node['importance'], 2)
        nodes_json.append({
            'id':    node['id'],
            'label': node['name'][:30] + '...' if len(node['name']) > 30 else node['name'],
            'title': f"{node['name']}\nType: {node_type}\nScore: {importance_display}\nID: {node['id']}",
            'color': color,
            'size':  node['size'],
            'font':  {'size': 12}
        })

    edges_json = []
    for link in links:
        edge_label = link['label'][:25] if len(link['label']) > 25 else link['label']
        edges_json.append({
            'from':   link['source'],
            'to':     link['target'],
            'title':  link['label'],
            'label':  edge_label,
            'arrows': 'to',
            'color':  {'color': "#636262", 'opacity': 0.6},
            'font':   {'size': 10}
        })

    # Context entity breakdown for stats panel
    ctx_type_labels = {
        'pathway':             'Pathways',
        'biological_process':  'GO Biological Process',
        'molecular_function':  'GO Molecular Function',
        'cellular_component':  'GO Cellular Component',
        'complex':             'Protein Complexes',
        'genetic_disorder':    'Genetic Disorders (OMIM)',
        'tissue':              'Tissues',
        'cell':                'Cell Lines',
        'gene/protein':        'Proteins',
        'disease':             'Disease',
        'other':               'Other',
    }
    ctx_rows = ''.join(
        f'<div><strong>{ctx_type_labels.get(k, k.replace("_"," ").title())}:</strong> {v}</div>'
        for k, v in stats.get('entity_counts', {}).items() if v > 0
    )

    stats_html = f"""
    <div style="padding: 15px; background: #f5f5f5; border-radius: 8px; margin-bottom: 15px;">
        <h3 style="margin: 0 0 10px 0;">Graph Statistics</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 13px; white-space: nowrap;">
            <div><strong>Nodes:</strong> {stats['total_nodes']}</div>
            <div><strong>Edges:</strong> {stats['total_edges']}</div>
            <div><strong>Proteins:</strong> {stats['protein_count']}</div>
            <div><strong>Predicted Targets:</strong> {stats.get('predicted_targets', 0)}</div>
            {ctx_rows}
        </div>
    </div>
    """

    legend_items = []
    for entity_type, count in stats['entity_counts'].items():
        color = color_map.get(entity_type, '#A7A4E0')
        label = ctx_type_labels.get(entity_type, entity_type.replace('_', ' ').title())
        legend_items.append(
            f'<div style="display: flex; align-items: center; margin: 5px 0;">'
            f'<div style="width: 20px; height: 20px; background: {color}; border-radius: 50%; margin-right: 8px;"></div>'
            f'<span>{label}: {count}</span></div>'
        )
    legend_html = f"""
    <div style="padding: 15px; background: #f5f5f5; border-radius: 8px;">
        <h3 style="margin: 0 0 10px 0;">Legend</h3>
        {''.join(legend_items)}
    </div>
    """

    top_entities_html = ""
    if subgraph_data.get('top_proteins'):
        proteins_list = '\n'.join([
            f'<li><strong>{p["name"]}</strong> (Score: {p["importance"]:.1f})</li>'
            for p in subgraph_data['top_proteins'][:10]
        ])
        top_entities_html = f"""
        <div style="padding: 15px; background: #E8F4FD; border-radius: 8px; margin-top: 15px;">
            <h3 style="margin: 0 0 10px 0;">Top Predicted Targets</h3>
            <ol style="margin: 0; padding-left: 20px; font-size: 13px;">{proteins_list}</ol>
        </div>
        """

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph - {disease_name}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #ffffff; }}
        #header {{ text-align: center; margin-bottom: 20px; }}
        #container {{ display: flex; gap: 20px; }}
        #network {{ flex: 1; height: 700px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }}
        #sidebar {{ width: 300px; overflow-y: auto; max-height: 700px; }}
        .controls {{ padding: 15px; background: #f5f5f5; border-radius: 8px; margin-bottom: 15px; }}
        button {{ padding: 8px 16px; margin: 5px; border: none; border-radius: 4px;
                  background: #007B82;
                  color: white; cursor: pointer; font-size: 14px; }}
        button:hover {{ background: white; color: #0A2E52; }}
    </style>
</head>
<body>
    <div id="header">
        <h2 style="color:#0A2E52">Knowledge Graph: {disease_name}</h2>
        <p style="color: #64748B;">Interactive Biological Context Network Visualization</p>
    </div>
    <div id="container">
        <div id="network"></div>
        <div id="sidebar">
            <div class="controls">
                <h3 style="margin: 0 0 10px 0;">Controls</h3>
                <button onclick="network.fit()">Fit to Screen</button>
                <button onclick="resetPhysics()">Reset Layout</button>
                <button onclick="togglePhysics()">Toggle Physics</button>
            </div>
            {stats_html}
            {legend_html}
            {top_entities_html}
        </div>
    </div>
    <script type="text/javascript">
        const nodes = new vis.DataSet({json.dumps(nodes_json)});
        const edges = new vis.DataSet({json.dumps(edges_json)});
        const options = {{
            nodes: {{ shape: 'dot', font: {{ size: 12, color: '#000000' }}, borderWidth: 2, borderWidthSelected: 3 }},
            edges: {{ width: 1, arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                      smooth: {{ type: 'continuous', roundness: 0.5 }} }},
            physics: {{
                enabled: true,
                forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01,
                                     springLength: 200, springConstant: 0.08, damping: 0.4, avoidOverlap: 0.5 }},
                maxVelocity: 50, solver: 'forceAtlas2Based', timestep: 0.35,
                stabilization: {{ enabled: true, iterations: 150, updateInterval: 25 }}
            }},
            interaction: {{ dragNodes: true, dragView: true, zoomView: true, hover: true, tooltipDelay: 200 }}
        }};
        const container = document.getElementById('network');
        const network   = new vis.Network(container, {{ nodes, edges }}, options);
        let physicsEnabled = true;
        function togglePhysics() {{ physicsEnabled = !physicsEnabled; network.setOptions({{ physics: {{ enabled: physicsEnabled }} }}); }}
        function resetPhysics() {{ network.setOptions({{ physics: {{ enabled: true }} }}); network.stabilize(); }}
        network.on('click', function(params) {{ if (params.nodes.length > 0) console.log('Selected:', nodes.get(params.nodes[0])); }});
        network.once('stabilizationIterationsDone', function() {{ network.fit({{ animation: {{ duration: 1000, easingFunction: 'easeInOutQuad' }} }}); }});
    </script>
</body>
</html>
    """

    filename = "kg_graph.html"
    filepath = os.path.join(GRAPHS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ Generated interactive HTML: {filepath}")
    return filename


# ============================================================================
# CUSTOM TARGET MANAGEMENT
# ============================================================================

custom_targets_cache = {}

def add_custom_target_to_cache(disease_id: str, uniprot_id: str, protein_name: str, score: float = 25.0):
    """Add a custom protein target to the cache"""
    if disease_id not in custom_targets_cache:
        custom_targets_cache[disease_id] = []
    existing = next((t for t in custom_targets_cache[disease_id] if t['uniprot_id'] == uniprot_id), None)
    if not existing:
        custom_target = {
            "uniprot_id": uniprot_id, "name": protein_name,
            "score": score, "paths": 0,
            "connection_types": ["Custom Addition"],
            "from_knowledge_graph": False, "custom_added": True
        }
        custom_targets_cache[disease_id].append(custom_target)
        return custom_target
    return existing

def get_custom_targets_for_disease(disease_id: str):
    return custom_targets_cache.get(disease_id, [])

def clear_custom_targets_for_disease(disease_id: str):
    if disease_id in custom_targets_cache:
        del custom_targets_cache[disease_id]


# ============================================================================
# LLM INTERPRETATION — Groq
# ============================================================================

async def get_llm_interpretation(disease_name: str, targets: List[Dict] = None):
    """Generate biomedical LLM interpretation of predicted targets via Groq"""
    try:
        content_parts = []
        if targets and len(targets) > 0:
            cleaned_targets = []
            for t in targets[:5]:
                name = t.get("name") or t.get("target") or t.get("uniprot_id") or "Unknown"
                name = str(name).replace("UniProt:", "").strip()
                cleaned_targets.append(f"- {name}")
            content_parts.append(f"Key therapeutic targets:\n" + "\n".join(cleaned_targets))

        if not content_parts:
            return (
                f"The knowledge graph for {disease_name} shows limited data. "
                f"This may reflect a rare or under-researched disease with few known therapeutic targets."
            )

        prompt = f"""
You are a biomedical research assistant analyzing knowledge-graph based predictions for {disease_name}.

{chr(10).join(content_parts)}

Provide a short, scientifically sound interpretation (4–5 sentences) covering:
1. What these predictions suggest about possible therapeutic mechanisms.
2. The biological pathways or target systems potentially involved.
3. Any notable research or translational implications.

Keep the language accessible to biomedical professionals.
"""

        def _call_llm():
            return llm_client.databricks(
                messages=[
                    {"role": "system", "content": "You are a biomedical AI expert specializing in drug discovery and therapeutic insights."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )

        interpretation = await asyncio.to_thread(_call_llm)
        return interpretation.strip()

    except Exception as e:
        error_msg    = str(e).lower()
        target_count = len(targets) if targets else 0
        target_names = []
        for t in (targets or [])[:5]:
            name = t.get("name") or t.get("target") or t.get("uniprot_id")
            if name:
                target_names.append(str(name).replace("UniProt:", "").strip())

        summary = (
            f"The knowledge graph analysis identified {target_count} protein target(s) "
            f"associated with {disease_name}."
        )
        if target_names:
            summary += f" Key protein targets include: {', '.join(target_names)}."
        if "timeout" in error_msg or "connection" in error_msg:
            return summary + " (Note: LLM interpretation unavailable due to temporary network timeout.)"
        if "api" in error_msg or "key" in error_msg:
            return summary + " (Note: Databricks serving endpoint misconfiguration detected.)"
        if "rate" in error_msg:
            return summary + " (Note: Databricks serving endpoint rate limit reached.)"
        return summary


# ============================================================================
# METAPATH EXTRACTION — context graph–aware
# ============================================================================

def extract_metapaths_for_targets(disease_id: str, target_protein_ids: set, max_hops: int = 3) -> Dict[str, List[Dict]]:
    """
    Extract multi-hop metapaths from disease to specific target proteins via context graph.

    Traversal uses ctx_adj (pre-filtered: no drugs/diseases as neighbors).
    Intermediates are ONLY biological context nodes (pathway, GO, complex,
    genetic disorder, tissue, cell, protein).

    Key fixes vs v1:
    - Per-path visited set only (no global visited_nodes_at_depth that blocked branches)
    - Target protein checked FIRST before capping intermediate neighbors
    - Raised limits: MAX_QUEUE_SIZE=5000, MAX_ITERATIONS=20000
    - Per-type neighbor caps raised for hub nodes (pathway/GO → 200)
    """
    cache_key = f"{disease_id}_{'-'.join(sorted(target_protein_ids))}_{max_hops}"
    if cache_key in METAPATH_CACHE:
        print(f"   ⚡ Using cached metapaths")
        return METAPATH_CACHE[cache_key]

    protein_metapaths = {}

    MAX_PATHS_PER_PROTEIN = 5
    MAX_QUEUE_SIZE        = 5000
    MAX_ITERATIONS        = 20000

    # Per-type neighbor caps — raised for biological hub nodes
    NEIGHBOR_CAPS = {
        'disease':            200,
        'pathway':            200,
        'complex':            200,
        'biological_process': 200,
        'molecular_function': 200,
        'cellular_component': 200,
        'genetic_disorder':   150,
        'gene/protein':       100,
        'tissue':             60,
        'cell':               60,
        'other':              60,
    }

    print(f"   ✅ Using global edge lookup ({len(edge_lookup):,} entries) for metapath extraction")

    ctx_graph = CONTEXT_GRAPH_CACHE.get(disease_id)

    def get_neighbors(node):
        if ctx_graph and ctx_graph.has_node(node):
            return set(ctx_graph.successors(node)) | set(ctx_graph.predecessors(node))
        return set(ctx_adj.get(node, {}).keys())

    targets_to_process = list(target_protein_ids)[:20]

    for protein_idx, protein_id in enumerate(targets_to_process):
        metapaths     = []
        visited_paths = set()

        queue      = deque()
        queue.append((disease_id, (disease_id,), ()))
        iterations = 0

        while queue and len(metapaths) < MAX_PATHS_PER_PROTEIN:
            iterations += 1
            if iterations > MAX_ITERATIONS:
                print(f"   ⚠️  Iteration limit for {protein_id}")
                break
            while len(queue) > MAX_QUEUE_SIZE:
                queue.pop()

            current, path_nodes, path_edges = queue.popleft()
            current_depth = len(path_nodes) - 1

            if current_depth >= max_hops:
                continue

            all_neighbors   = get_neighbors(current)
            path_nodes_set  = set(path_nodes)

            # Check target first — before any cap is applied
            target_present  = protein_id in all_neighbors
            other_neighbors = [n for n in all_neighbors if n != protein_id and n not in path_nodes_set]

            # Cap intermediate neighbors by type
            cap = NEIGHBOR_CAPS.get(id_to_type.get(current, 'other'), 60)
            if len(other_neighbors) > cap:
                other_neighbors.sort(
                    key=lambda n: CONTEXT_TYPE_WEIGHTS.get(id_to_type.get(n, 'other'), 0.2),
                    reverse=True
                )
                other_neighbors = other_neighbors[:cap]

            # Process target protein if reachable
            if target_present:
                edge_rel       = edge_lookup.get((current, protein_id),
                                  edge_lookup.get((protein_id, current), 'connected_to'))
                new_path_nodes = path_nodes + (protein_id,)
                new_path_edges = path_edges + (edge_rel,)
                hop_count      = len(new_path_edges)

                if hop_count > 1:
                    int_types = [id_to_type.get(n, 'other') for n in new_path_nodes[1:-1]]
                    valid = 'disease' not in int_types and 'drug' not in int_types
                else:
                    valid = True

                if valid and new_path_nodes not in visited_paths:
                    visited_paths.add(new_path_nodes)

                    intermediate      = None
                    intermediate_name = None
                    intermediate_type = None
                    if hop_count > 1:
                        intermediate      = new_path_nodes[1]
                        intermediate_name = id_to_name.get(intermediate, intermediate)[:40]
                        intermediate_type = id_to_type.get(intermediate, 'other')

                    metapaths.append({
                        'path_type':         f'{hop_count}-hop',
                        'nodes':             list(new_path_nodes),
                        'node_names':        [id_to_name.get(n, n)[:40] for n in new_path_nodes],
                        'node_types':        [id_to_type.get(n, 'other') for n in new_path_nodes],
                        'edges':             list(new_path_edges),
                        'edge_labels':       [e.replace('_', ' ').title() for e in new_path_edges],
                        'intermediate':      intermediate,
                        'intermediate_name': intermediate_name,
                        'intermediate_type': intermediate_type,
                        'hop_count':         hop_count,
                        'context_weight':    CONTEXT_TYPE_WEIGHTS.get(intermediate_type, 0.2)
                                             if intermediate_type else 1.0,
                    })
                    if len(metapaths) >= MAX_PATHS_PER_PROTEIN:
                        break

            # Enqueue intermediates for deeper search
            if current_depth < max_hops - 1:
                for neighbor in other_neighbors:
                    neighbor_type = id_to_type.get(neighbor, 'other')
                    if neighbor_type in ('disease', 'drug'):
                        continue
                    # Skip non-human proteins as intermediates
                    if neighbor_type == 'gene/protein' and neighbor not in PROTEIN_NAME_INDEX:
                        continue
                    edge_rel = edge_lookup.get((current, neighbor),
                               edge_lookup.get((neighbor, current), 'connected_to'))
                    queue.append((neighbor, path_nodes + (neighbor,), path_edges + (edge_rel,)))

        metapaths.sort(key=lambda p: (-p.get('context_weight', 0), p.get('hop_count', 99)))
        protein_metapaths[protein_id] = metapaths

        found = len(metapaths)
        if (protein_idx + 1) % 5 == 0 or found == 0:
            status = "✅" if found > 0 else "⚠️  no paths"
            print(f"   {status} [{protein_idx + 1}/{len(targets_to_process)}] {protein_id}: {found} paths")

    found_count = sum(1 for p in protein_metapaths.values() if p)
    print(f"   ✅ Completed: {found_count}/{len(targets_to_process)} targets have paths")

    METAPATH_CACHE[cache_key] = protein_metapaths
    return protein_metapaths


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def init_txkg_data():
    """Initialize TxKG data on module import"""
    global PROTEIN_NAME_INDEX, TARGET_CACHE, METAPATH_CACHE

    PROTEIN_NAME_INDEX.clear()
    TARGET_CACHE.clear()
    METAPATH_CACHE.clear()
    CONTEXT_GRAPH_CACHE.clear()
    print("♻️ Cleared in-memory caches on module load.")

    print("\n" + "=" * 70)
    print("🚀 TxKG Module Loading")
    print("=" * 70 + "\n")

    if load_cache():
        print("\n✅ TxKG data loaded from cache!")
    else:
        print("📦 Building TxKG indexes (first-time setup)...")
        load_data()
        save_cache()
        print("\n✅ TxKG data ready!")

try:
    init_txkg_data()
except Exception as e:
    print(f"⚠️ Error loading TxKG data: {e}")
    print("   Data will be loaded on first request")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/txkg")
async def root():
    """Root endpoint"""
    return {
        "message": "TxKG FastAPI Server",
        "version": "7.0.0",
        "status": "online",
        "llm_backend": "groq",
        "features": [
            "Context graph traversal (pathway/GO BP+MF+CC/complex/genetic disorder/tissue/cell)",
            "Local TSV-based protein names — HUMAN only, no UniProt API",
            "Node importance scoring with context type boost",
            "Context-aware metapath analysis (biological reasoning paths)",
            "Knowledge graph edge-based target extraction",
            "REAL PubMed URLs (no hallucination)"
        ],
        "endpoints": {
            "/txkg/diseases":          "Get list of diseases",
            "/txkg/predicted-targets": "Get therapeutic targets (KG-based)",
            "/txkg/subgraph":          "Get enhanced disease subgraph",
            "/txkg/metapaths":         "Get metapaths for predicted targets (reasoning paths)",
            "/txkg/disease-edges":     "Get all edges for a disease (DEBUG)",
            "/txkg/llm-interpretation":"Get AI interpretation",
            "/txkg/articles":          "Get related articles (REAL URLs only)",
            "/txkg/health":            "Health check"
        }
    }


@router.get("/txkg/diseases")
async def get_diseases(
    search: Optional[str] = Query(None, description="Search term"),
    limit:  int = Query(10000, ge=1, le=100000, description="Maximum number of diseases to return")
):
    """Get list of available diseases"""
    try:
        available = [
            {"id": did, "name": id_to_name.get(did, did)}
            for did in diseases_in_kg
        ]
        if search:
            search_lower = search.lower()
            available = [d for d in available if search_lower in d['name'].lower()]
        available.sort(key=lambda x: x['name'])
        return {
            "success": True,
            "count": len(available[:limit]),
            "total_available": len(available),
            "total_diseases_in_kg": len(diseases_in_kg),
            "diseases": available[:limit]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/txkg/predicted-targets")
async def get_predicted_targets(
    disease:         str = Query(...,  description="Disease name"),
    limit:           int = Query(10,   ge=1, le=500, description="Limit number of targets (default: 10)"),
    custom_proteins: Optional[str] = Query(None, description="Comma-separated UniProt IDs to add"),
    max_hops:        int = Query(3,    ge=1, le=5,   description="Maximum hop distance")
):
    """
    Get predicted therapeutic protein targets (HUMAN only, names from local TSV).
    Context graph traversal with pathway/GO/complex/genetic disorder/tissue/cell scoring.
    """
    try:
        disease_id, disease_name = find_disease(disease)
        if not disease_id:
            raise HTTPException(status_code=404, detail=f"Disease '{disease}' not found in the knowledge graph")

        print(f"\n{'=' * 70}")
        print(f"🎯 Extracting targets for: {disease_name}")
        print('=' * 70)

        all_targets = await extract_targets(disease_id, max_targets=500, max_hops=max_hops)

        if not all_targets and not custom_proteins:
            disease_edges = df_links[(df_links['source'] == disease_id) | (df_links['target'] == disease_id)]
            return {
                "success": True,
                "disease": disease_name, "disease_id": disease_id,
                "count": 0, "targets": [],
                "note": f"No human protein targets found. Disease has {len(disease_edges)} edges in KG.",
                "max_hops_searched": max_hops
            }

        display_targets = all_targets[:limit] if limit else all_targets

        hop_counts = defaultdict(int)
        for t in all_targets:
            hop_counts[t['hop']] += 1

        formatted_targets = []
        for t in display_targets:
            gene_name  = t.get("gene_name")
            full_name  = t.get("full_name")

            if full_name and full_name != t["uniprot_id"] and len(full_name) <= 100:
                final_display = full_name
            elif gene_name and gene_name != t["uniprot_id"]:
                final_display = gene_name
            else:
                final_display = t["uniprot_id"]

            formatted_targets.append({
                "uniprot_id":          t["uniprot_id"],
                "name":                final_display,
                "gene_name":           gene_name,
                "full_name":           full_name,
                "score":               round(t["score"], 2),
                "context_score":       t.get("context_score", 0),
                "hop_distance":        t["hop"],
                "paths":               t.get("paths", []),
                "connection_types":    t.get("connection_types", []),
                "from_knowledge_graph": True,
                "custom_added":        False
            })

        # Handle custom proteins
        if custom_proteins:
            custom_ids = [cid.strip() for cid in custom_proteins.split(',') if cid.strip()]
            if custom_ids:
                print(f"\n➕ Adding {len(custom_ids)} custom proteins...")
                valid_custom_ids = [
                    cid for cid in custom_ids
                    if cid.startswith(('P', 'Q', 'O')) and 6 <= len(cid) <= 10
                ]
                if valid_custom_ids:
                    custom_uniprot_data = await fetch_multiple_uniprot_names_enhanced(valid_custom_ids)
                    for custom_id in valid_custom_ids:
                        if any(t['uniprot_id'] == custom_id for t in formatted_targets):
                            print(f"   ⚠️  {custom_id} already in knowledge graph, skipping")
                            continue
                        uniprot_info = custom_uniprot_data.get(custom_id, {})
                        gene_name    = uniprot_info.get('gene')
                        full_name    = uniprot_info.get('full_name')
                        if full_name and full_name != custom_id and len(full_name) <= 100:
                            protein_name = full_name
                        elif gene_name and gene_name != custom_id:
                            protein_name = gene_name
                        else:
                            protein_name = custom_id

                        exists_in_graph = custom_id in entity_neighbors
                        if exists_in_graph:
                            protein_nbrs = entity_neighbors.get(custom_id, {})
                            disease_nbrs = disease_links_index.get(disease_id, set())
                            all_nbrs     = protein_nbrs.get('targets', set()) | protein_nbrs.get('sources', set())
                            shared       = disease_nbrs & all_nbrs
                            custom_score = min(len(shared) * 20, 100)
                            paths_count  = len(shared)
                            connection_types = ["Custom Addition", "In Knowledge Graph"]
                        else:
                            custom_score     = 25.0
                            paths_count      = 0
                            connection_types = ["Custom Addition", "Not in KG"]

                        formatted_targets.append({
                            "uniprot_id":           custom_id,
                            "name":                 protein_name,
                            "gene_name":            gene_name,
                            "full_name":            full_name,
                            "score":                custom_score,
                            "context_score":        0,
                            "hop_distance":         None,
                            "paths":                paths_count,
                            "connection_types":     connection_types,
                            "from_knowledge_graph": exists_in_graph,
                            "custom_added":         True
                        })
                        print(f"   ✅ Added {protein_name} - Score: {custom_score}")

        # LLM interpretation via Groq
        try:
            kg_targets_only = [t for t in formatted_targets if not t.get('custom_added', False)]
            llm_summary = await get_llm_interpretation(disease_name, targets=kg_targets_only[:5])
        except Exception as e:
            print(f"LLM interpretation failed: {e}")
            kg_count     = len([t for t in formatted_targets if not t.get('custom_added', False)])
            top_proteins = [t["name"] for t in formatted_targets if not t.get('custom_added', False)][:5]
            llm_summary  = (
                f"The knowledge graph analysis identified {kg_count} human protein targets "
                f"associated with {disease_name}. Top targets include: {', '.join(top_proteins)}."
            )

        kg_count     = len([t for t in formatted_targets if not t.get('custom_added', False)])
        custom_count = len([t for t in formatted_targets if t.get('custom_added', False)])

        print(f"\n✅ Returning {len(formatted_targets)} total targets ({kg_count} from KG, {custom_count} custom)")
        print(f"{'=' * 70}\n")

        return {
            "success": True,
            "disease": disease_name, "disease_id": disease_id,
            "count": len(formatted_targets),
            "targets": formatted_targets,
            "interpretation": llm_summary,
            "knowledge_graph_targets": kg_count,
            "custom_targets": custom_count,
            "naming_source": "local_tsv_human_only",
            "score_range": "0-100",
            "target_extraction": f"1 to {max_hops}-hop connections (context graph)",
            "max_hops_searched": max_hops,
            "hop_distribution": {
                f"{hop}-hop": count for hop, count in sorted(hop_counts.items())
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in predicted targets: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/txkg/subgraph")
async def get_subgraph(
    disease:   str = Query(..., description="Disease name"),
    max_nodes: int = Query(100, ge=10, le=500, description="Maximum nodes to include"),
    max_hops:  int = Query(3,   ge=1,  le=5,   description="Maximum hop depth")
):
    """
    Get enhanced disease subgraph via context graph traversal.
    Includes pathway, GO BP/MF/CC, complex, genetic disorder, tissue, cell nodes.
    Generates interactive HTML visualization at /api/v1/graphs/kg_graph.html.
    """
    try:
        disease_id, disease_name = find_disease(disease)
        if not disease_id:
            raise HTTPException(status_code=404, detail=f"Disease '{disease}' not found in knowledge graph")

        print(f"\n{'=' * 70}")
        print(f"🔍 Building Enhanced Subgraph: {disease_name}")
        print('=' * 70)

        subgraph_data = await build_enhanced_subgraph(disease_id, max_nodes=max_nodes, max_hops=max_hops)

        if not subgraph_data['nodes']:
            return {
                "success": False,
                "disease": disease_name, "disease_id": disease_id,
                "message": "No relationships found for this disease",
                "subgraph": subgraph_data, "html_url": None,
                "max_hops_searched": max_hops
            }

        html_filename = generate_interactive_html(disease_id, disease_name, subgraph_data)
        html_url      = f"/api/v1/graphs/{html_filename}"

        print(f"✅ Subgraph built: {subgraph_data['statistics']['total_nodes']} nodes, "
              f"{subgraph_data['statistics']['total_edges']} edges")
        print(f"✅ HTML available at: {html_url}")
        print(f"{'=' * 70}\n")

        return {
            "success": True,
            "disease": disease_name, "disease_id": disease_id,
            "subgraph": subgraph_data,
            "html_url": html_url,
            "max_hops_searched": max_hops,
            "color_legend": {
                "disease":            "#0225AA",
                "gene/protein":       "#1E88E5",
                "pathway":            "#065B52",
                "biological_process": "#AA44AA",
                "molecular_function": "#CC4488",
                "cellular_component": "#7B5EA7",
                "complex":            "#00897B",
                "genetic_disorder":   "#E61919",
                "tissue":             "#7CB87C",
                "cell":               "#AED581",
                "other":              "#A7A4E0"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in subgraph: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/txkg/llm-interpretation")
async def get_interpretation(disease: str = Query(..., description="Disease name")):
    """Get AI interpretation of predictions"""
    try:
        disease_id, disease_name = find_disease(disease)
        if not disease_id:
            raise HTTPException(status_code=404, detail=f"Disease '{disease}' not found in knowledge graph")

        all_targets = await extract_targets(disease_id, max_targets=100)
        targets     = all_targets[:5]

        if not targets:
            interpretation = f"Limited data available for {disease_name} in the knowledge graph."
        else:
            interpretation = await get_llm_interpretation(disease_name, targets)

        return {
            "success": True,
            "disease": disease_name, "disease_id": disease_id,
            "interpretation": interpretation,
            "targets_analyzed": len(targets)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/txkg/articles")
async def get_articles(
    disease: str = Query(...,  description="Disease name"),
    target:  Optional[str] = Query(None, description="Target protein name (optional)"),
    limit:   int = Query(2,    ge=1, le=10, description="Articles per target")
):
    """Get related research articles from PubMed"""
    try:
        from urllib.parse import unquote
        disease_decoded = unquote(disease)

        disease_id, disease_name = find_disease(disease_decoded)
        if not disease_id:
            raise HTTPException(status_code=404, detail=f"Disease '{disease_decoded}' not found")

        if target:
            target_decoded = unquote(target)
            articles = await fetch_pubmed_articles_for_target(disease_name, target_decoded, limit)
            return {
                "success": True,
                "disease": disease_name, "target": target_decoded,
                "count": len(articles), "articles": articles
            }

        targets = await extract_targets(disease_id, max_targets=100, max_hops=3)
        if not targets:
            return {
                "success": True,
                "disease": disease_name, "count": 0, "articles": [],
                "message": "No targets found for this disease"
            }

        top_targets = sorted(targets, key=lambda t: t["score"], reverse=True)[:5]
        print(f"\n📰 Fetching articles for {len(top_targets)} top targets...")

        all_articles = []
        for idx, target_data in enumerate(top_targets):
            target_id   = target_data["id"]
            target_name = target_data.get("name", "")
            gene_name   = target_data.get("gene_name")

            if gene_name and gene_name != target_id and len(gene_name) > 1:
                search_name = gene_name
            elif target_name and target_name != target_id and len(target_name) > 1:
                search_name = target_name.split('(')[0].strip() if '(' in target_name else target_name
            else:
                search_name = target_id

            search_name = str(search_name).replace("UniProt:", "").strip()
            print(f"   🔍 Target {idx + 1}: Searching PubMed for '{search_name}'...")

            articles = await fetch_pubmed_articles_for_target(disease_name, search_name, limit)
            if articles:
                print(f"      ✅ Found {len(articles)} articles")
                for article in articles:
                    article['target_id']    = target_id
                    article['target_name']  = search_name
                    article['target_score'] = target_data['score']
                all_articles.extend(articles)
            else:
                print(f"      ⚠️  No articles found for {search_name}")

        # Deduplicate by PMID
        seen_pmids     = set()
        unique_articles = []
        for article in all_articles:
            pmid = article.get('pmid')
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                unique_articles.append(article)

        print(f"\n✅ Total articles found: {len(unique_articles)}")

        return {
            "success": True,
            "disease": disease_name, "disease_id": disease_id,
            "targets_searched": len(top_targets),
            "count": len(unique_articles),
            "articles": unique_articles
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error fetching articles: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/txkg/health")
async def health_check():
    """Health check endpoint"""
    return {
        "success": True,
        "status": "healthy",
        "version": "7.0.0-context-graph",
        "llm_backend": "databricks",
        "llm_model": settings.DATABRICKS_LLM_ENDPOINT,
        "data_loaded": df_links is not None,
        "cache_stats": {
            "protein_name_index_size":  len(PROTEIN_NAME_INDEX),
            "target_cache_size":        len(TARGET_CACHE),
            "metapath_cache_size":      len(METAPATH_CACHE),
            "context_graph_cache_size": len(CONTEXT_GRAPH_CACHE),
        },
        "stats": {
            "total_entities":   len(id_to_name),
            "total_diseases":   len(disease_ids),
            "diseases_in_kg":   len(diseases_in_kg),
            "total_drugs":      len(drug_ids),
            "total_proteins":   len(protein_ids),
            "total_triples":    len(df_links) if df_links is not None else 0
        },
        "features": {
            "context_graph":           True,
            "context_graph_types":     list(CONTEXT_ENTITY_TYPES),
            "enhanced_subgraph":       True,
            "llm_entity_cleaning":     "disabled_for_speed",
            "node_importance_scoring": True,
            "metapath_analysis":       True,
            "verified_pubmed_urls":    True,
            "caching":                 "enabled",
            "multi_hop_support":       True,
            "max_hops":                "1-5 (configurable)",
            "protein_naming":          "local_tsv_human_only",
            "full_name_char_limit":    100,
            "context_type_weights":    CONTEXT_TYPE_WEIGHTS,
            "optimizations": [
                "Local TSV protein name index — HUMAN only, no UniProt API calls",
                "Full biological context graph (pathway/GO BP+MF+CC/complex/genetic disorder/tissue/cell)",
                "Context-aware target scoring (pathway > genetic_disorder > GO > complex > tissue > cell > PPI)",
                "Context graph caching per disease",
                "O(1) edge type lookup (edge_lookup dict)",
                "Pre-filtered context adjacency index (ctx_adj)",
                "Target extraction caching",
                "Metapath caching",
                "Multi-hop BFS traversal",
                "Context type weighting in metapath sorting",
                "Groq LLM backend (llama-3.3-70b-versatile)"
            ]
        }
    }


@router.post("/txkg/clear-cache")
async def clear_all_caches():
    """Clear all runtime session caches to free memory. Protein name index is preserved."""
    global TARGET_CACHE, METAPATH_CACHE, CONTEXT_GRAPH_CACHE

    protein_index_count  = len(PROTEIN_NAME_INDEX)
    target_count         = len(TARGET_CACHE)
    metapath_count       = len(METAPATH_CACHE)
    context_graph_count  = len(CONTEXT_GRAPH_CACHE)

    TARGET_CACHE.clear()
    METAPATH_CACHE.clear()
    CONTEXT_GRAPH_CACHE.clear()

    return {
        "success": True,
        "message": "Session caches cleared (protein name index preserved)",
        "cleared": {
            "target_entries":        target_count,
            "metapath_entries":      metapath_count,
            "context_graph_entries": context_graph_count,
        },
        "preserved": {
            "protein_name_index_entries": protein_index_count,
            "reason": "Static index from TSV — call /txkg/reload-protein-index to refresh"
        }
    }


@router.post("/txkg/reload-protein-index")
async def reload_protein_index():
    """Reload protein name index from TSV file (e.g. after TSV update)"""
    old_count = len(PROTEIN_NAME_INDEX)
    PROTEIN_NAME_INDEX.clear()
    TARGET_CACHE.clear()   # Invalidate since names may change
    build_protein_name_index()
    new_count = len(PROTEIN_NAME_INDEX)
    return {
        "success": True,
        "message": "Protein name index reloaded from TSV",
        "previous_count": old_count,
        "new_count": new_count
    }


@router.get("/txkg/disease-edges")
async def get_disease_edges(
    disease: str = Query(...,  description="Disease name"),
    limit:   int = Query(100,  ge=1, le=500, description="Maximum edges to return")
):
    """Get all edges connected to a disease — DEBUG endpoint"""
    try:
        disease_id, disease_name = find_disease(disease)
        if not disease_id:
            raise HTTPException(status_code=404, detail=f"Disease '{disease}' not found")

        disease_edges     = df_links[(df_links['source'] == disease_id) | (df_links['target'] == disease_id)].head(limit)
        edges             = []
        entity_type_counts = defaultdict(int)

        for _, row in disease_edges.iterrows():
            source_id      = row['source']
            target_id      = row['target']
            connected_id   = target_id if source_id == disease_id else source_id
            connected_type = id_to_type.get(connected_id, 'other')
            entity_type_counts[connected_type] += 1
            edges.append({
                "source":               source_id,
                "source_name":          id_to_name.get(source_id, source_id),
                "edge_type":            row['edge_type'],
                "target":               target_id,
                "target_name":          id_to_name.get(target_id, target_id),
                "connected_entity_type": connected_type
            })

        return {
            "success": True,
            "disease": disease_name, "disease_id": disease_id,
            "total_edges": len(edges),
            "edges": edges,
            "statistics": {"entity_types": dict(entity_type_counts)}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graphs/kg_graph.html")
async def serve_graph_html():
    """Serve the generated knowledge graph HTML file"""
    filepath = os.path.join(GRAPHS_DIR, "kg_graph.html")
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail="Graph not generated yet. Please call /api/v1/txkg/subgraph first."
        )
    return FileResponse(
        filepath, media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache", "Expires": "0",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )


@router.get("/txkg/graph-html-content")
async def get_graph_html_content():
    """Return the HTML content directly as text for embedding"""
    filepath = os.path.join(GRAPHS_DIR, "kg_graph.html")
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail="Graph not generated yet. Please call /api/v1/txkg/subgraph first."
        )
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return {
            "success": True,
            "html": html_content,
            "file_size": len(html_content),
            "file_path": os.path.abspath(filepath)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading HTML file: {str(e)}")


@router.get("/txkg/graph-status")
async def check_graph_status():
    """Check if graph HTML file exists"""
    filepath = os.path.join(GRAPHS_DIR, "kg_graph.html")
    exists   = os.path.exists(filepath)
    file_info = {}
    if exists:
        stat = os.stat(filepath)
        file_info = {
            "size":     stat.st_size,
            "modified": stat.st_mtime,
            "path":     os.path.abspath(filepath)
        }
    return {
        "success": True,
        "graph_exists": exists,
        "file_info": file_info,
        "graphs_directory": os.path.abspath(GRAPHS_DIR)
    }


@router.get("/txkg/metapaths")
async def get_metapaths_for_disease(
    disease:  str = Query(...,  description="Disease name"),
    limit:    int = Query(10,   ge=1, le=50, description="Number of targets to analyze"),
    max_hops: int = Query(3,    ge=1, le=5,  description="Maximum path length")
):
    """
    Get metapaths for predicted targets via context graph traversal.
    Returns biological reasoning paths from disease to top predicted targets.
    Paths traverse biological context nodes (pathway/GO/complex/genetic disorder/tissue/cell).
    Sorted by context type weight (pathway > genetic_disorder > GO > complex > …).
    """
    try:
        disease_id, disease_name = find_disease(disease)
        if not disease_id:
            raise HTTPException(status_code=404, detail=f"Disease '{disease}' not found")

        print(f"\n{'=' * 70}")
        print(f"🔍 Extracting metapaths for: {disease_name}")
        print('=' * 70)

        top_targets = await extract_targets(disease_id, max_targets=limit, max_hops=max_hops)

        if not top_targets:
            return {
                "success": True,
                "disease": disease_name, "disease_id": disease_id,
                "targets_analyzed": 0, "metapaths": [],
                "summary": {"total_paths": 0, "one_hop_paths": 0, "two_hop_paths": 0},
                "max_hops_searched": max_hops
            }

        target_protein_ids = {t['id'] for t in top_targets}
        print(f"   🔗 Extracting metapaths for {len(target_protein_ids)} targets...")
        metapaths_data = extract_metapaths_for_targets(disease_id, target_protein_ids, max_hops=max_hops)

        metapath_results = []
        hop_path_counts  = defaultdict(int)

        for target in top_targets:
            target_id = target['id']
            paths     = metapaths_data.get(target_id, [])

            paths_by_hop = defaultdict(int)
            for path in paths:
                hop_count = path.get('hop_count', len(path.get('edges', [])))
                paths_by_hop[hop_count] += 1
                hop_path_counts[hop_count] += 1

            metapath_results.append({
                'target_id':     target_id,
                'target_name':   target['name'],
                'score':         target['score'],
                'context_score': target.get('context_score', 0),
                'hop_type':      target.get('hop', 1),
                'total_paths':   len(paths),
                'paths_by_hop':  dict(paths_by_hop),
                'paths':         paths
            })

        metapath_results.sort(key=lambda x: x['score'], reverse=True)
        total_paths = sum(hop_path_counts.values())

        print(f"   ✅ Found {total_paths} total paths")
        for hop, count in sorted(hop_path_counts.items()):
            print(f"      - {hop}-hop: {count}")
        print(f"{'=' * 70}\n")

        return {
            "success": True,
            "disease": disease_name, "disease_id": disease_id,
            "targets_analyzed": len(metapath_results),
            "metapaths": metapath_results,
            "summary": {
                "total_paths":          total_paths,
                "paths_by_hop":         dict(hop_path_counts),
                "avg_paths_per_target": round(total_paths / len(metapath_results), 2) if metapath_results else 0
            },
            "max_hops_searched": max_hops,
            "hop_distribution":  {
                f"{hop}-hop": count for hop, count in sorted(hop_path_counts.items())
            },
            "context_type_weights": CONTEXT_TYPE_WEIGHTS  # expose weights to frontend
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in metapaths: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))