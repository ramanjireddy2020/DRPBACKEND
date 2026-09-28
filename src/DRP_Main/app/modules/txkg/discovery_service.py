"""
TxKG discovery service — implementation of the TxKG functional specification.

Implements the spec's three-tool structure on top of the BioKG data already loaded
by ``app/api/v1/endpoints/txkg_test.py`` (entity lookups, ``df_links``, ``ctx_adj``,
``edge_lookup``, ``PROTEIN_NAME_INDEX``).

    Tool 1 — resolve_disease()            §1  disease resolution
             reconstruct_paths()          §6  per-candidate bounded BFS + subgraph
    Tool 2 — score_candidates()           §2  full-graph RWRH propagation
                                          §3  degree correction
                                          §4  Known/Hidden categorization
                                          §5  sourcing gate
    Tool 3 — novelty_label()              §7  patent/literature novelty axis
             interpret_candidates()       §8  grounded LLM interpretation

    run_discovery()                       §9/§12 full chain + §10 recommendation
    SESSION_STATE / answer_followup()     §11 supervising layer

Nothing here is hop-bounded before scoring: RWRH runs over the whole graph and the
restart probability supplies the distance decay (§2). BFS is used only to redraw the
path for a candidate that already survived scoring (§6).
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from DRP_Main.app.api.v1.endpoints import txkg_test as kg
from DRP_Main.app.modules.txkg.kegg_pathways import KEGG_PATHWAY_NAMES

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

#: Restart probability for RWR. Higher = signal stays closer to the seed.
RESTART_PROB = float(os.getenv("TXKG_RWR_RESTART", "0.30"))

#: Inter-layer jump probability for the heterogeneous variant (RWRH). At a node with
#: both same-layer and cross-layer neighbours, this much of the outgoing probability
#: mass crosses into other node-type layers.
LAYER_JUMP_PROB = float(os.getenv("TXKG_RWR_LAMBDA", "0.50"))

RWR_MAX_ITER = int(os.getenv("TXKG_RWR_MAX_ITER", "100"))
RWR_TOL = float(os.getenv("TXKG_RWR_TOL", "1e-10"))

#: §3 — fraction of the graph treated as the "disease-relevant region" when testing,
#: per candidate, whether its connectivity into that region beats chance given its degree.
RELEVANT_REGION_FRACTION = float(os.getenv("TXKG_REGION_FRACTION", "0.005"))
RELEVANT_REGION_MIN = 200

#: §3 — node types that may constitute the disease-relevant region. Propagation itself
#: still runs over the whole graph including drugs (§2, no cutoff, no layer removed),
#: but the region a candidate is tested *against* is the biological one. Without this,
#: a disease's drug associations flood the region and every drug-metabolism hub
#: (CYP3A4, albumin, ABCB1) reads as disease-relevant through the drug layer — the exact
#: connectivity artifact the correction exists to neutralise. Degree stays uncorrected
#: for this: a hub's drug edges still count against its expected hit rate.
RELEVANT_REGION_TYPES = {
    "gene/protein",
    "pathway",
    "biological_process",
    "molecular_function",
    "cellular_component",
    "complex",
    "genetic_disorder",
    "disease",
    "disease_category",
    "tissue",
    "cell",
}

#: §4 — a candidate with no curated disease association needs at least this corrected
#: score (-log10 p) to be surfaced as Hidden. Anything weaker is dropped outright.
HIDDEN_SCORE_THRESHOLD = float(os.getenv("TXKG_HIDDEN_THRESHOLD", "3.0"))

#: How many top-ranked proteins by raw propagation get the (more expensive) degree
#: correction and downstream treatment. Not a hop cutoff — every protein is propagated
#: to and ranked first; this only bounds the reporting tail.
CANDIDATE_POOL_SIZE = int(os.getenv("TXKG_CANDIDATE_POOL", "500"))

#: §0 — the curated disease-association edge type. In BioKG this is the
#: DisGeNET/CTD-derived protein↔disease edge. A candidate is "already documented"
#: if and only if this exact edge exists.
CURATED_DISEASE_ASSOC_EDGE = "PROTEIN_DISEASE_ASSOCIATION"

#: §0 — provenance for every edge type: which curated source it was compiled from and
#: how much weight that source carries. BioKG's link file carries no per-edge reference
#: column, so provenance is resolved at the edge-type level against the source registry
#: published with the dataset (see biokg_data/README.md "Data Sources"). An edge type
#: absent from this registry is treated as unsourced by the §5 gate.
EDGE_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "PROTEIN_DISEASE_ASSOCIATION": {"source": "CTD / MedGen curated disease-gene associations", "confidence": 0.95},
    "PROTEIN_PATHWAY_ASSOCIATION": {"source": "Reactome / KEGG / SMPDB pathway membership", "confidence": 0.90},
    "DISEASE_PATHWAY_ASSOCIATION": {"source": "CTD / KEGG disease-pathway curation", "confidence": 0.85},
    "MEMBER_OF_COMPLEX": {"source": "Reactome complex composition", "confidence": 0.90},
    "COMPLEX_IN_PATHWAY": {"source": "Reactome complex-pathway assignment", "confidence": 0.90},
    "COMPLEX_TOP_LEVEL_PATHWAY": {"source": "Reactome top-level pathway hierarchy", "confidence": 0.85},
    "DISEASE_GENETIC_DISORDER": {"source": "MedGen / OMIM disease-disorder mapping", "confidence": 0.85},
    "RELATED_GENETIC_DISORDER": {"source": "MedGen / OMIM related-disorder mapping", "confidence": 0.80},
    "PPI": {"source": "IntAct curated protein interactions", "confidence": 0.70},
    "GO_BP": {"source": "UniProt GO biological process annotation", "confidence": 0.80},
    "GO_MF": {"source": "UniProt GO molecular function annotation", "confidence": 0.75},
    "GO_CC": {"source": "UniProt GO cellular component annotation", "confidence": 0.65},
    "PATHWAY_GO_BP": {"source": "Reactome pathway GO biological process", "confidence": 0.80},
    "PATHWAY_GO_MF": {"source": "Reactome pathway GO molecular function", "confidence": 0.75},
    "PATHWAY_GO_CC": {"source": "Reactome pathway GO cellular component", "confidence": 0.65},
    "PROTEIN_EXPRESSED_IN": {"source": "Human Protein Atlas expression", "confidence": 0.60},
    "PART_OF_TISSUE": {"source": "HPA / Cellosaurus tissue hierarchy", "confidence": 0.55},
    "DRUG_TARGET": {"source": "DrugBank drug-target", "confidence": 0.90},
    "DRUG_ENZYME": {"source": "DrugBank drug-enzyme", "confidence": 0.85},
    "DRUG_TRANSPORTER": {"source": "DrugBank drug-transporter", "confidence": 0.85},
    "DRUG_CARRIER": {"source": "DrugBank drug-carrier", "confidence": 0.85},
    "DRUG_DISEASE_ASSOCIATION": {"source": "CTD drug-disease association", "confidence": 0.80},
    "DRUG_PATHWAY_ASSOCIATION": {"source": "SMPDB drug-pathway association", "confidence": 0.80},
    "DPI": {"source": "DrugBank drug-protein interaction", "confidence": 0.80},
    "DDI": {"source": "DrugBank drug-drug interaction", "confidence": 0.75},
    "HAS_PARENT_PATHWAY": {"source": "Reactome pathway hierarchy", "confidence": 0.85},
    "DISEASE_SUPERGRP": {"source": "MeSH disease tree hierarchy", "confidence": 0.80},
}

# --------------------------------------------------------------------------------------
# Annotation layers — the node types §0 requires but biokg.links.tsv does not carry
# --------------------------------------------------------------------------------------
#
# §0 lists biological process and phenotype among the graph's node types, and §6 promises
# to show "which pathways/phenotypes/processes connect the disease to this candidate".
# biokg.links.tsv has no such nodes at all — GO annotations, tissue expression, the MeSH
# disease tree and the Reactome pathway hierarchy live in the biokg.properties.* files
# instead. Without them the graph is proteins/pathways/complexes/disorders only, and no
# process ever appears on a path. These loaders add them.
#
# They are kept in this module's own structures and are never written back into
# txkg_test's shared globals or its pickle cache, so the legacy context-graph endpoints
# behave exactly as before.

#: file → {attribute: (node type of the value, edge direction)}. "forward" means the
#: row's subject links to the value; "reverse" means the value is the parent/container.
ANNOTATION_FILES: Dict[str, Dict[str, str]] = {
    "biokg.properties.protein.tsv": {
        "GO_BP": "biological_process",
        "GO_MF": "molecular_function",
        "GO_CC": "cellular_component",
        "PROTEIN_EXPRESSED_IN": "tissue",
    },
    "biokg.properties.pathway.tsv": {
        "PATHWAY_GO_BP": "biological_process",
        "PATHWAY_GO_MF": "molecular_function",
        "PATHWAY_GO_CC": "cellular_component",
        "HAS_PARENT_PATHWAY": "pathway",
    },
    "biokg.properties.cell.tsv": {
        "PART_OF_TISSUE": "tissue",
    },
    "biokg.properties.disease.tsv": {
        "DISEASE_SUPERGRP": "disease_category",
    },
}

#: Which annotation edge types are loaded. Defaults deliberately exclude the two
#: lowest-specificity, highest-volume ones:
#:   PROTEIN_EXPRESSED_IN (986k edges) makes every protein sharing a tissue "connected",
#:   GO_CC (363k) does the same for every protein sharing a compartment ("nucleus").
#: Both would dominate propagation with co-occurrence rather than disease relevance —
#: the same hub artifact §3 exists to suppress, but introduced upstream of the
#: correction. Enable them explicitly if you want that trade-off:
#:   TXKG_ANNOTATION_EDGES="GO_BP,GO_MF,GO_CC,PROTEIN_EXPRESSED_IN,..."
DEFAULT_ANNOTATION_EDGES = (
    "GO_BP,GO_MF,PATHWAY_GO_BP,PATHWAY_GO_MF,HAS_PARENT_PATHWAY,DISEASE_SUPERGRP"
)
ENABLED_ANNOTATION_EDGES: Set[str] = {
    e.strip()
    for e in os.getenv("TXKG_ANNOTATION_EDGES", DEFAULT_ANNOTATION_EDGES).split(",")
    if e.strip()
}

_ANNOTATION_CACHE_PATH = os.path.join(kg.BIOKG_DATA_DIR, "annotation_edges.pkl")

#: Types and display names for annotation nodes. Held here rather than in
#: kg.id_to_type / kg.id_to_name so no shared state is mutated.
_ANNOT_TYPE: Dict[str, str] = {}
_ANNOT_NAME: Dict[str, str] = {}
_ANNOT_ADJ: Dict[str, Dict[str, str]] = {}
_ANNOT_EDGES: List[Tuple[str, str, str]] = []
_ANNOTATIONS_LOADED = False

_ANNOT_LABEL_PREFIX = {
    "biological_process": "GO BP",
    "molecular_function": "GO MF",
    "cellular_component": "GO CC",
    "tissue": "Tissue",
    "cell": "Cell",
    "disease_category": "MeSH group",
}


def node_type(node_id: str) -> str:
    """Type of any node, annotation nodes included."""
    annotated = _ANNOT_TYPE.get(node_id)
    if annotated is not None:
        return annotated
    return kg.id_to_type.get(node_id, "other")


#: GO term labels, resolved once each and cached for the process. Unlike KEGG's
#: 372 human pathways these are too many to bundle, and only the handful that
#: appear on a path are ever needed. A miss is cached as well as a hit so an
#: unreachable API costs one attempt per term, not one per render.
_GO_NAMES: Dict[str, str] = {}
_GO_API = "https://api.geneontology.org/api/ontology/term/"


def _go_label(term: str) -> str:
    """The GO term's label, or "" when it cannot be resolved."""
    if term in _GO_NAMES:
        return _GO_NAMES[term]
    label = ""
    try:
        import json as _json
        import urllib.request

        request = urllib.request.Request(
            _GO_API + term.replace(":", "%3A"), headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            label = (_json.loads(response.read().decode()) or {}).get("label") or ""
    except Exception:  # noqa: BLE001 — degrade to the accession
        label = ""
    _GO_NAMES[term] = label
    return label


def node_name(node_id: str) -> str:
    """Display name of any node, annotation nodes included."""
    name = kg.id_to_name.get(node_id)
    if name is not None and name != node_id:
        return name

    annotated = _ANNOT_NAME.get(node_id)
    if annotated and annotated != node_id:
        return annotated

    # Accession-shaped ids the graph has no name for.
    kegg = KEGG_PATHWAY_NAMES.get(node_id)
    if kegg:
        return kegg
    match = _GO_ID_RE.search(node_id)
    if match:
        label = _go_label(match.group(0))
        if label:
            return label

    return annotated or name or node_id


_GO_ID_RE = re.compile(r"GO:\d{7}")


def _annotation_signature() -> str:
    return ",".join(sorted(ENABLED_ANNOTATION_EDGES))


def load_annotations(force: bool = False) -> List[Tuple[str, str, str]]:
    """
    Parse the enabled annotation edges out of the biokg.properties.* files.

    Cached to disk because a full pass over biokg.properties.protein.tsv (91 MB) is far
    too slow to repeat on every boot.
    """
    global _ANNOTATIONS_LOADED, _ANNOT_EDGES
    if _ANNOTATIONS_LOADED and not force:
        return _ANNOT_EDGES

    signature = _annotation_signature()
    if not ENABLED_ANNOTATION_EDGES:
        _ANNOTATIONS_LOADED = True
        return _ANNOT_EDGES

    if not force and os.path.exists(_ANNOTATION_CACHE_PATH):
        try:
            with open(_ANNOTATION_CACHE_PATH, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("signature") == signature:
                _ANNOT_EDGES = blob["edges"]
                _ANNOT_TYPE.update(blob["types"])
                _ANNOT_NAME.update(blob["names"])
                _rebuild_annotation_adjacency()
                _ANNOTATIONS_LOADED = True
                print(f"   ⚡ Annotation edges loaded from cache ({len(_ANNOT_EDGES):,})")
                return _ANNOT_EDGES
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  Annotation cache unusable ({exc}) — reparsing")

    print(f"\n📚 Loading annotation layers [{signature}]…")
    t0 = time.time()
    edges: List[Tuple[str, str, str]] = []
    per_type: Dict[str, int] = defaultdict(int)

    for filename, attributes in ANNOTATION_FILES.items():
        wanted = {a: t for a, t in attributes.items() if a in ENABLED_ANNOTATION_EDGES}
        if not wanted:
            continue
        path = os.path.join(kg.BIOKG_DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"   ⚠️  {filename} not found — skipping")
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    subject, attribute, value = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    value_type = wanted.get(attribute)
                    if value_type is None or not subject or not value:
                        continue
                    edges.append((subject, value, attribute))
                    per_type[attribute] += 1
                    # Only *new* node types get a local type/name entry; a value that is
                    # already a known entity (a parent pathway, say) keeps its own.
                    if value not in kg.id_to_type and value not in _ANNOT_TYPE:
                        _ANNOT_TYPE[value] = value_type
                        prefix = _ANNOT_LABEL_PREFIX.get(value_type)
                        _ANNOT_NAME[value] = (
                            f"{prefix}: {value}" if prefix else value.replace("_", " ")
                        )
        except Exception as exc:  # noqa: BLE001 — annotations must never break the module
            print(f"   ⚠️  Could not read {filename}: {exc}")

    _ANNOT_EDGES = edges
    _rebuild_annotation_adjacency()
    _ANNOTATIONS_LOADED = True

    print(f"   ✅ {len(edges):,} annotation edges, {len(_ANNOT_TYPE):,} new nodes — {time.time() - t0:.1f}s")
    for attribute, count in sorted(per_type.items(), key=lambda kv: -kv[1]):
        print(f"      {attribute}: {count:,}")

    try:
        with open(_ANNOTATION_CACHE_PATH, "wb") as fh:
            pickle.dump(
                {
                    "signature": signature,
                    "edges": edges,
                    "types": _ANNOT_TYPE,
                    "names": _ANNOT_NAME,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Could not cache annotation edges: {exc}")
    return _ANNOT_EDGES


def _rebuild_annotation_adjacency() -> None:
    """Undirected adjacency over the annotation edges, for §6 path reconstruction."""
    _ANNOT_ADJ.clear()
    for subject, value, attribute in _ANNOT_EDGES:
        _ANNOT_ADJ.setdefault(subject, {})[value] = attribute
        _ANNOT_ADJ.setdefault(value, {})[subject] = attribute


def traversal_neighbours(node_id: str) -> Dict[str, str]:
    """
    Neighbours available to §6 path reconstruction: the legacy pre-filtered context
    adjacency (drugs and other diseases already excluded) plus the annotation layers.
    """
    neighbours = dict(kg.ctx_adj.get(node_id, {}))
    neighbours.update(_ANNOT_ADJ.get(node_id, {}))
    return neighbours

#: §5 — an edge below this source confidence does not count as properly sourced.
MIN_SOURCE_CONFIDENCE = float(os.getenv("TXKG_MIN_SOURCE_CONFIDENCE", "0.60"))

#: §6 — bounded BFS budget for redrawing a surviving candidate's path.
PATH_MAX_HOPS = int(os.getenv("TXKG_PATH_MAX_HOPS", "3"))
PATH_MAX_PER_TARGET = int(os.getenv("TXKG_PATH_MAX_PER_TARGET", "5"))

_MATRIX_CACHE_PATH = os.path.join(kg.BIOKG_DATA_DIR, "rwrh_transition.pkl")
_NULL_MODEL_PATH = os.path.join(kg.BIOKG_DATA_DIR, "rwrh_null_model.npz")


# --------------------------------------------------------------------------------------
# Graph index — built once, reused for every query
# --------------------------------------------------------------------------------------


@dataclass
class GraphIndex:
    """Sparse, layer-aware representation of the *entire* knowledge graph."""

    nodes: List[str]
    index: Dict[str, int]
    transition: Any            # scipy.sparse.csr_matrix — column-stochastic, RWRH-weighted
    adjacency: Any             # scipy.sparse.csr_matrix — binary, for the degree correction
    degree: np.ndarray
    layers: np.ndarray         # integer layer id per node
    layer_names: List[str]

    def has(self, node_id: str) -> bool:
        return node_id in self.index


_GRAPH_INDEX: Optional[GraphIndex] = None
_NULL_MODEL: Optional[Dict[str, np.ndarray]] = None
_NULL_MODEL_LOADED = False

#: RWR score vectors, keyed by (disease_id, restart, lambda).
_RWR_CACHE: Dict[Tuple[str, float, float], np.ndarray] = {}

#: Curated protein↔disease pairs (§0/§4), built once from df_links.
_CURATED_ASSOC: Optional[Dict[str, Set[str]]] = None


def _require_data() -> None:
    if kg.df_links is None:
        raise RuntimeError(
            "BioKG data is not loaded. TxKG could not read its dataset "
            f"(BIOKG_DATA_DIR={kg.BIOKG_DATA_DIR})."
        )


def build_graph_index(force: bool = False) -> GraphIndex:
    """
    Build (or load) the full-graph RWRH transition matrix.

    The heterogeneous variant matters because the graph mixes node types. At each
    node, ``LAYER_JUMP_PROB`` of the outgoing probability mass is spread across the
    *other* node-type layers it touches (evenly per layer, then evenly within the
    layer), and the remainder stays inside its own layer. A node with neighbours in
    only one of the two categories sends all its mass there.
    """
    global _GRAPH_INDEX
    if _GRAPH_INDEX is not None and not force:
        return _GRAPH_INDEX

    _require_data()

    import pandas as pd
    from scipy import sparse

    load_annotations()

    if not force and os.path.exists(_MATRIX_CACHE_PATH):
        try:
            with open(_MATRIX_CACHE_PATH, "rb") as fh:
                blob = pickle.load(fh)
            if (
                blob.get("restart_lambda") == LAYER_JUMP_PROB
                and blob.get("annotation_signature") == _annotation_signature()
            ):
                _GRAPH_INDEX = GraphIndex(
                    nodes=blob["nodes"],
                    index={n: i for i, n in enumerate(blob["nodes"])},
                    transition=blob["transition"],
                    adjacency=blob["adjacency"],
                    degree=blob["degree"],
                    layers=blob["layers"],
                    layer_names=blob["layer_names"],
                )
                print(f"   ⚡ RWRH transition matrix loaded from cache ({len(blob['nodes']):,} nodes)")
                return _GRAPH_INDEX
        except Exception as exc:  # noqa: BLE001 — a stale cache must never be fatal
            print(f"   ⚠️  RWRH matrix cache unusable ({exc}) — rebuilding")

    print("\n🧮 Building full-graph RWRH transition matrix (no hop cutoff)…")
    t0 = time.time()

    src = kg.df_links["source"].astype(str).to_numpy()
    tgt = kg.df_links["target"].astype(str).to_numpy()

    # Annotation layers (§0's process/phenotype node types) join the same graph, so
    # propagation can travel disease → protein → biological process → protein.
    if _ANNOT_EDGES:
        annot_src = np.array([e[0] for e in _ANNOT_EDGES], dtype=object)
        annot_tgt = np.array([e[1] for e in _ANNOT_EDGES], dtype=object)
        src = np.concatenate([src, annot_src])
        tgt = np.concatenate([tgt, annot_tgt])

    nodes = pd.unique(np.concatenate([src, tgt]))
    nodes = [str(n) for n in nodes]
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    ui = np.fromiter((index[s] for s in src), dtype=np.int32, count=len(src))
    vi = np.fromiter((index[t] for t in tgt), dtype=np.int32, count=len(tgt))

    # Undirected: propagation must be able to travel in both directions.
    keep = ui != vi
    ui, vi = ui[keep], vi[keep]
    row = np.concatenate([ui, vi])
    col = np.concatenate([vi, ui])

    node_types = [node_type(node) for node in nodes]
    layer_names = sorted(set(node_types))
    layer_of_name = {name: i for i, name in enumerate(layer_names)}
    layers = np.fromiter(
        (layer_of_name[t] for t in node_types), dtype=np.int16, count=n
    )

    adjacency = sparse.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n)
    )
    adjacency.data[:] = 1.0        # collapse parallel edges
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()

    # ── layer-aware edge weights (RWRH) ───────────────────────────────────────────
    a_coo = adjacency.tocoo()
    e_from, e_to = a_coo.row, a_coo.col
    same_layer = layers[e_from] == layers[e_to]

    same_counts = np.bincount(e_from[same_layer], minlength=n).astype(np.float64)
    cross_counts = np.bincount(e_from[~same_layer], minlength=n).astype(np.float64)

    # distinct cross layers reached from each node, and neighbours per (node, layer)
    cross_from = e_from[~same_layer]
    cross_layer = layers[e_to][~same_layer]
    pair_keys = cross_from.astype(np.int64) * len(layer_names) + cross_layer.astype(np.int64)
    uniq_pairs, pair_inv, pair_sizes = np.unique(pair_keys, return_inverse=True, return_counts=True)
    n_cross_layers = np.bincount(
        (uniq_pairs // len(layer_names)).astype(np.int64), minlength=n
    ).astype(np.float64)

    lam = np.where(
        (same_counts > 0) & (cross_counts > 0),
        LAYER_JUMP_PROB,
        np.where(same_counts > 0, 0.0, 1.0),
    )

    weights = np.zeros(len(e_from), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_same = (1.0 - lam[e_from[same_layer]]) / same_counts[e_from[same_layer]]
    weights[same_layer] = np.nan_to_num(w_same)
    if cross_from.size:
        w_cross = (
            lam[cross_from]
            / np.maximum(n_cross_layers[cross_from], 1.0)
            / pair_sizes[pair_inv].astype(np.float64)
        )
        weights[~same_layer] = np.nan_to_num(w_cross)

    # column-stochastic: transition[v, u] = P(u → v)
    transition = sparse.csr_matrix(
        (weights.astype(np.float32), (e_to, e_from)), shape=(n, n)
    )

    _GRAPH_INDEX = GraphIndex(
        nodes=nodes,
        index=index,
        transition=transition,
        adjacency=adjacency,
        degree=degree,
        layers=layers,
        layer_names=layer_names,
    )
    print(
        f"   ✅ RWRH matrix: {n:,} nodes, {adjacency.nnz // 2:,} undirected edges, "
        f"{len(layer_names)} layers — {time.time() - t0:.1f}s"
    )

    try:
        with open(_MATRIX_CACHE_PATH, "wb") as fh:
            pickle.dump(
                {
                    "nodes": nodes,
                    "transition": transition,
                    "adjacency": adjacency,
                    "degree": degree,
                    "layers": layers,
                    "layer_names": layer_names,
                    "restart_lambda": LAYER_JUMP_PROB,
                    "annotation_signature": _annotation_signature(),
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"   💾 Cached transition matrix → {_MATRIX_CACHE_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Could not cache transition matrix: {exc}")

    return _GRAPH_INDEX


def _curated_assoc() -> Dict[str, Set[str]]:
    """§0/§4 — protein → set of diseases carrying a curated disease-association edge."""
    global _CURATED_ASSOC
    if _CURATED_ASSOC is not None:
        return _CURATED_ASSOC

    _require_data()
    assoc: Dict[str, Set[str]] = defaultdict(set)
    mask = kg.df_links["edge_type"].astype(str) == CURATED_DISEASE_ASSOC_EDGE
    sub = kg.df_links[mask]
    for s, t in zip(sub["source"].astype(str), sub["target"].astype(str)):
        s_type = kg.id_to_type.get(s, "other")
        if s_type == "gene/protein":
            protein, disease = s, t
        else:
            protein, disease = t, s
        assoc[protein].add(disease)
    _CURATED_ASSOC = dict(assoc)
    print(f"   ✅ Curated disease-association index: {len(_CURATED_ASSOC):,} proteins")
    return _CURATED_ASSOC


def has_curated_association(protein_id: str, disease_id: str) -> bool:
    """§4 — the one direct fact that separates Known from Hidden."""
    return disease_id in _curated_assoc().get(protein_id, ())


# --------------------------------------------------------------------------------------
# TOOL 1 (part 1) — §1 Disease resolution
# --------------------------------------------------------------------------------------


@dataclass
class ResolvedDisease:
    id: str
    name: str
    method: str
    clarification: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.id)


def resolve_disease(query: str) -> ResolvedDisease:
    """
    §1 — free-text disease name → one confirmed anchor node.

    Exact match → normalized match → fuzzy match against the disease node index →
    synonym-table fallback. If nothing matches closely enough, returns a single
    clarifying question instead of guessing; nothing downstream may run.
    """
    _require_data()
    raw = (query or "").strip()
    if not raw:
        return ResolvedDisease("", "", "none", "Which disease should I analyse?")

    import re
    from difflib import SequenceMatcher, get_close_matches

    q = raw.lower()
    if q in kg.disease_name_to_id and kg.disease_name_to_id[q] in kg.diseases_in_kg:
        did = kg.disease_name_to_id[q]
        return ResolvedDisease(did, kg.id_to_name.get(did, did), "exact")

    # Synonym-table fallback: the same normalizations the disease index was built with.
    for variant, method in (
        (q.replace(",", ""), "normalized"),
        (q.replace("-", " "), "normalized"),
        (re.sub(r"\s+", " ", q.replace(",", "").replace("-", " ")).strip(), "synonym_table"),
    ):
        did = kg.disease_name_to_id.get(variant)
        if did and did in kg.diseases_in_kg:
            return ResolvedDisease(did, kg.id_to_name.get(did, did), method)

    candidates = [name for name, did in kg.disease_name_to_id.items() if did in kg.diseases_in_kg]

    # Fuzzy match, but only accepted when the match is genuinely close. The legacy
    # resolver falls through to bare substring containment, which anchors free text like
    # "not a disease" onto whatever disease name happens to be a substring of it; §1
    # requires one confirmed anchor or one clarifying question, never a silent guess.
    for match in get_close_matches(q, candidates, n=5, cutoff=0.6):
        if SequenceMatcher(None, q, match).ratio() >= 0.75:
            did = kg.disease_name_to_id[match]
            return ResolvedDisease(did, kg.id_to_name.get(did, did), "fuzzy")

    # Whole-phrase containment ("severe asthma in adults" → "Asthma"), long names only.
    phrase_hits = [
        name for name in candidates
        if len(name) >= 5 and re.search(rf"\b{re.escape(name)}\b", q)
    ]
    if phrase_hits:
        best = max(phrase_hits, key=len)
        did = kg.disease_name_to_id[best]
        return ResolvedDisease(did, kg.id_to_name.get(did, did), "phrase_match")

    # Nothing close enough — surface the near misses as one clarifying question.
    near = get_close_matches(q, candidates, n=5, cutoff=0.3)
    suggestions = sorted({kg.id_to_name.get(kg.disease_name_to_id[m], m) for m in near})
    question = (
        f"I could not match '{raw}' to a disease in the knowledge graph."
        + (f" Did you mean one of: {', '.join(suggestions)}?" if suggestions else "")
    )
    return ResolvedDisease("", "", "unresolved", question)


# --------------------------------------------------------------------------------------
# TOOL 2 (part 1) — §2 Full-graph RWRH propagation
# --------------------------------------------------------------------------------------


def propagate(
    disease_id: str,
    restart: float = RESTART_PROB,
    layer_jump: float = LAYER_JUMP_PROB,
) -> np.ndarray:
    """
    §2 — Random Walk with Restart (heterogeneous), seeded at the disease node, run
    over the *whole* graph. No hop cutoff: distant nodes decay smoothly under the
    restart probability rather than being cut off by a wall, so a well-supported but
    distant protein can still enter the candidate pool.
    """
    key = (disease_id, restart, layer_jump)
    if key in _RWR_CACHE:
        return _RWR_CACHE[key]

    gi = build_graph_index()
    if disease_id not in gi.index:
        raise ValueError(f"Disease '{disease_id}' has no edges in the knowledge graph")

    n = len(gi.nodes)
    seed = np.zeros(n, dtype=np.float64)
    seed[gi.index[disease_id]] = 1.0

    p = seed.copy()
    W = gi.transition
    for iteration in range(RWR_MAX_ITER):
        p_next = (1.0 - restart) * (W @ p) + restart * seed
        total = p_next.sum()
        if total > 0:
            p_next /= total
        delta = np.abs(p_next - p).sum()
        p = p_next
        if delta < RWR_TOL:
            print(f"   ✅ RWRH converged in {iteration + 1} iterations (Δ={delta:.2e})")
            break
    else:
        print(f"   ⚠️  RWRH hit the {RWR_MAX_ITER}-iteration cap without full convergence")

    _RWR_CACHE[key] = p
    return p


# --------------------------------------------------------------------------------------
# TOOL 2 (part 2) — §3 Degree correction
# --------------------------------------------------------------------------------------


def _load_null_model() -> Optional[Dict[str, np.ndarray]]:
    """
    §3 (alternative) — precomputed, degree-matched null distributions.

    Only ever *loaded* at query time; it is built offline by ``build_null_model()``
    because rerunning the propagation hundreds of times per query is not practical.
    """
    global _NULL_MODEL, _NULL_MODEL_LOADED
    if _NULL_MODEL_LOADED:
        return _NULL_MODEL
    _NULL_MODEL_LOADED = True
    if not os.path.exists(_NULL_MODEL_PATH):
        return None
    try:
        blob = np.load(_NULL_MODEL_PATH)
        gi = build_graph_index()
        if int(blob["n_nodes"]) != len(gi.nodes):
            print("   ⚠️  Null model was built for a different graph — ignoring")
            return None
        _NULL_MODEL = {"mean": blob["mean"], "std": blob["std"], "n_samples": blob["n_samples"]}
        print(f"   ✅ Null model loaded ({int(blob['n_samples'])} degree-matched seeds)")
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Null model unusable: {exc}")
        _NULL_MODEL = None
    return _NULL_MODEL


def build_null_model(n_samples: int = 100, seed: int = 0) -> str:
    """
    Offline job (§3, alternative path): rerun the same propagation from randomized,
    degree-matched disease seeds and store the per-node mean/std of the scores nodes
    receive by chance. Rebuild whenever the graph itself changes.

    Not called at query time. Run it from a Databricks job or a shell:
        python -c "from DRP_Main.app.modules.txkg.discovery_service import build_null_model; build_null_model(200)"
    """
    gi = build_graph_index()
    rng = np.random.default_rng(seed)

    disease_idx = np.array(
        [gi.index[d] for d in kg.diseases_in_kg if d in gi.index], dtype=np.int64
    )
    if disease_idx.size == 0:
        raise RuntimeError("No disease nodes available to sample degree-matched seeds from")

    # Degree-matched sampling: bucket disease nodes by log-degree, draw across buckets.
    degrees = gi.degree[disease_idx]
    buckets = np.digitize(np.log1p(degrees), np.quantile(np.log1p(degrees), [0.25, 0.5, 0.75]))

    picks: List[int] = []
    for b in range(4):
        pool = disease_idx[buckets == b]
        if pool.size:
            take = max(1, n_samples // 4)
            picks.extend(rng.choice(pool, size=min(take, pool.size), replace=False).tolist())

    n = len(gi.nodes)
    total = np.zeros(n)
    total_sq = np.zeros(n)
    W = gi.transition

    for i, node_i in enumerate(picks, start=1):
        s = np.zeros(n)
        s[node_i] = 1.0
        p = s.copy()
        for _ in range(RWR_MAX_ITER):
            p_next = (1.0 - RESTART_PROB) * (W @ p) + RESTART_PROB * s
            tot = p_next.sum()
            if tot > 0:
                p_next /= tot
            if np.abs(p_next - p).sum() < RWR_TOL:
                p = p_next
                break
            p = p_next
        total += p
        total_sq += p * p
        if i % 10 == 0:
            print(f"   … {i}/{len(picks)} null samples")

    k = float(len(picks))
    mean = total / k
    var = np.maximum(total_sq / k - mean * mean, 0.0)
    std = np.sqrt(var)

    np.savez_compressed(
        _NULL_MODEL_PATH, mean=mean, std=std, n_samples=np.int64(len(picks)), n_nodes=np.int64(n)
    )
    print(f"✅ Null model written to {_NULL_MODEL_PATH} ({len(picks)} seeds)")
    return _NULL_MODEL_PATH


def degree_corrected_scores(
    disease_id: str,
    raw_scores: np.ndarray,
    candidate_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    §3 — turn a raw propagation score into a disease-specific one.

    Analytical path (preferred, one calculation per candidate): treat the top of the
    propagation ranking as the disease-relevant region R of the graph, then ask, per
    candidate, how surprising its number of neighbours inside R is for a node of its
    degree — a hypergeometric tail probability. A generically well-connected hub lands
    a lot of neighbours in R purely by degree, so it scores unremarkably; a protein
    with far more connections into R than its degree predicts scores high.

    Returns ``corrected`` (=-log10 p), ``p_value``, ``expected`` hits, ``observed``
    hits, and — when a precomputed null model exists — a degree-matched ``z_score``.
    """
    from scipy.stats import hypergeom

    gi = build_graph_index()
    n_nodes = len(gi.nodes)

    region_size = max(RELEVANT_REGION_MIN, int(n_nodes * RELEVANT_REGION_FRACTION))

    # The region is the top of the propagation ranking restricted to biological layers.
    eligible = np.array(
        [i for i, name in enumerate(gi.layer_names) if name in RELEVANT_REGION_TYPES],
        dtype=np.int16,
    )
    eligible_mask = np.isin(gi.layers, eligible)
    eligible_idx = np.flatnonzero(eligible_mask)
    region_size = min(region_size, max(eligible_idx.size - 1, 1))

    eligible_scores = raw_scores[eligible_idx]
    top_local = np.argpartition(-eligible_scores, region_size - 1)[:region_size]
    region_idx = eligible_idx[top_local]

    in_region = np.zeros(n_nodes, dtype=np.float32)
    in_region[region_idx] = 1.0

    # One sparse mat-vec gives every node's neighbour count inside R.
    hits_all = np.asarray(gi.adjacency @ in_region).ravel()

    observed = hits_all[candidate_idx].astype(np.int64)
    degrees = gi.degree[candidate_idx].astype(np.int64)
    self_in_region = in_region[candidate_idx].astype(np.int64)

    # Population excludes the candidate itself; so does R if the candidate is in it.
    population = n_nodes - 1
    successes = region_size - self_in_region
    draws = np.minimum(degrees, population)

    p_values = hypergeom.sf(observed - 1, population, successes, draws)
    p_values = np.clip(np.nan_to_num(p_values, nan=1.0), 1e-300, 1.0)
    corrected = -np.log10(p_values)
    expected = draws * successes / np.maximum(population, 1)

    out = {
        "corrected": corrected,
        "p_value": p_values,
        "observed": observed,
        "expected": expected,
        "degree": degrees,
        "method": "hypergeometric_degree_conditioned",
    }

    null = _load_null_model()
    if null is not None:
        std = np.maximum(null["std"][candidate_idx], 1e-15)
        out["z_score"] = (raw_scores[candidate_idx] - null["mean"][candidate_idx]) / std
        out["method"] = "hypergeometric_degree_conditioned+precomputed_null_z"
    return out


# --------------------------------------------------------------------------------------
# TOOL 1 (part 2) — §6 Path reconstruction (only for candidates that survived scoring)
# --------------------------------------------------------------------------------------


def _edge_provenance(a: str, b: str) -> Dict[str, Any]:
    """§0/§5 — resolve one edge back to the curated record it came from."""
    rel = (
        kg.edge_lookup.get((a, b))
        or kg.edge_lookup.get((b, a))
        or _ANNOT_ADJ.get(a, {}).get(b)
        or _ANNOT_ADJ.get(b, {}).get(a)
    )
    if rel is None:
        return {"relation": None, "source": None, "confidence": 0.0, "sourced": False}
    prov = EDGE_PROVENANCE.get(rel)
    if prov is None:
        return {"relation": rel, "source": None, "confidence": 0.0, "sourced": False}
    return {
        "relation": rel,
        "source": prov["source"],
        "confidence": prov["confidence"],
        "sourced": prov["confidence"] >= MIN_SOURCE_CONFIDENCE,
    }


def reconstruct_paths(
    disease_id: str,
    protein_id: str,
    max_hops: int = PATH_MAX_HOPS,
    max_paths: int = PATH_MAX_PER_TARGET,
) -> List[Dict[str, Any]]:
    """
    §6 — bounded BFS from the disease to *this already-selected* candidate.

    BFS decides nothing about who gets considered; scoring did that. This only redraws
    the connecting route so the user can see which pathways / phenotypes / processes
    carry the signal. Traversal runs on the legacy context adjacency (drugs and other
    diseases already excluded) plus the annotation layers.
    """
    _require_data()
    load_annotations()
    if disease_id == protein_id:
        return []

    paths: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, ...]] = set()
    queue: deque = deque([(disease_id, (disease_id,))])
    iterations = 0

    while queue and len(paths) < max_paths and iterations < 200_000:
        iterations += 1
        current, path = queue.popleft()
        depth = len(path) - 1
        if depth >= max_hops:
            continue

        neighbours = traversal_neighbours(current)

        if protein_id in neighbours and path + (protein_id,) not in seen:
            full = path + (protein_id,)
            seen.add(full)
            paths.append(_describe_path(full))
            if len(paths) >= max_paths:
                break

        if depth >= max_hops - 1:
            continue
        for neighbour in neighbours:
            if neighbour == protein_id or neighbour in path:
                continue
            n_type = node_type(neighbour)
            if n_type in ("disease", "drug"):
                continue
            if n_type == "gene/protein" and neighbour not in kg.PROTEIN_NAME_INDEX:
                continue
            queue.append((neighbour, path + (neighbour,)))

    # Shortest and best-sourced first.
    paths.sort(key=lambda p: (p["hop_count"], -p["min_confidence"]))
    return paths


def _describe_path(nodes: Tuple[str, ...]) -> Dict[str, Any]:
    edges = [_edge_provenance(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    confidences = [e["confidence"] for e in edges] or [0.0]
    return {
        "nodes": list(nodes),
        "node_names": [node_name(n) for n in nodes],
        "node_types": [node_type(n) for n in nodes],
        "edges": [e["relation"] for e in edges],
        "edge_labels": [(e["relation"] or "related_to").replace("_", " ").title() for e in edges],
        "edge_sources": [e["source"] for e in edges],
        "edge_confidences": confidences,
        "hop_count": len(nodes) - 1,
        "fully_sourced": all(e["sourced"] for e in edges) and bool(edges),
        "min_confidence": min(confidences),
        "metapath": "→".join(node_type(n) for n in nodes),
    }


def build_candidate_subgraph(disease_id: str, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """§6/§9 — the collapsed subgraph/metapath visual, assembled from the reconstructed
    paths of the candidates that survived. Nothing is drawn for the whole graph upfront."""
    nodes: Dict[str, Dict[str, Any]] = {}
    links: Dict[Tuple[str, str], Dict[str, Any]] = {}
    metapath_counts: Dict[str, int] = defaultdict(int)

    def _add_node(node_id: str, is_target: bool = False) -> None:
        entry = nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "name": node_name(node_id),
                "type": node_type(node_id),
                "is_disease": node_id == disease_id,
                "is_candidate": False,
            },
        )
        entry["is_candidate"] = entry["is_candidate"] or is_target

    _add_node(disease_id)
    for cand in candidates:
        _add_node(cand["uniprot_id"], is_target=True)
        for path in cand.get("paths", []):
            metapath_counts[path["metapath"]] += 1
            path_nodes = path["nodes"]
            for node_id in path_nodes:
                _add_node(node_id, is_target=node_id == cand["uniprot_id"])
            for i in range(len(path_nodes) - 1):
                a, b = path_nodes[i], path_nodes[i + 1]
                links.setdefault(
                    (a, b),
                    {
                        "source": a,
                        "target": b,
                        "type": path["edges"][i],
                        "label": path["edge_labels"][i],
                        "provenance": path["edge_sources"][i],
                        "confidence": path["edge_confidences"][i],
                    },
                )

    type_counts: Dict[str, int] = defaultdict(int)
    for node in nodes.values():
        type_counts[node["type"]] += 1

    return {
        "nodes": list(nodes.values()),
        "links": list(links.values()),
        "metapaths": [
            {"metapath": mp, "count": c}
            for mp, c in sorted(metapath_counts.items(), key=lambda kv: -kv[1])
        ],
        "statistics": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "entity_counts": dict(type_counts),
            "candidates_drawn": sum(1 for n in nodes.values() if n["is_candidate"]),
        },
        "collapsed": True,   # §9 — shown collapsed until the user opens it
    }


#: Node colours for the rendered subgraph, keyed by node type.
_VIS_COLOURS = {
    "disease": "#0A2E52",
    "gene/protein": "#2D6A4F",
    "pathway": "#007B82",
    "biological_process": "#1E3A5F",
    "molecular_function": "#02A7B0",
    "cellular_component": "#5B6FA3",
    "complex": "#7B5EA7",
    "genetic_disorder": "#D32F2F",
    "disease_category": "#B25C00",
    "tissue": "#7FB685",
    "cell": "#9AD1D4",
    "drug": "#43A047",
    "other": "#64748B",
}
_KNOWN_COLOUR = "#1B7F3B"
_HIDDEN_COLOUR = "#C2410C"
_UNCONFIRMED_COLOUR = "#9CA3AF"


def render_candidate_subgraph_html(
    disease_id: str,
    disease_name: str,
    subgraph: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> str:
    """
    §6/§9 — render the candidate subgraph as an openable page: the visual evidence
    behind the scores. Candidates are coloured by their §4 category and outlined by
    their §5 sourcing status; every edge carries its provenance in the tooltip.

    Returns the filename written under ``txkg_test.GRAPHS_DIR``.
    """
    by_id = {c["uniprot_id"]: c for c in candidates}

    vis_nodes = []
    for node in subgraph["nodes"]:
        cand = by_id.get(node["id"])
        colour = _VIS_COLOURS.get(node["type"], _VIS_COLOURS["other"])
        border = colour
        size = 18
        title = f"{node['name']}\nType: {node['type']}\nID: {node['id']}"

        if node["is_disease"]:
            size = 46
            title = f"{node['name']}\nDisease anchor\nID: {node['id']}"
        elif cand is not None:
            known = cand.get("category") == "Known/Direct"
            colour = _KNOWN_COLOUR if known else _HIDDEN_COLOUR
            border = colour if cand.get("confirmed", True) else _UNCONFIRMED_COLOUR
            size = 30
            title = (
                f"{cand.get('name')} ({cand['uniprot_id']})\n"
                f"{cand.get('category')}\n"
                f"Corrected score: {cand.get('corrected_score')} (-log10 p)\n"
                f"Sourcing: {cand.get('sourcing_status')}\n"
                f"Novelty: {cand.get('novelty_label', 'Unknown')}\n"
                f"Degree {cand.get('degree')}, "
                f"{cand.get('neighbours_in_disease_region')} neighbours in the "
                f"disease-relevant region (expected {cand.get('expected_by_degree')})"
            )

        label = node["name"]
        vis_nodes.append(
            {
                "id": node["id"],
                "label": label[:34] + "…" if len(label) > 34 else label,
                "title": title,
                "color": {"background": colour, "border": border},
                "borderWidth": 4 if cand is not None else 1,
                "size": size,
            }
        )

    vis_edges = [
        {
            "from": link["source"],
            "to": link["target"],
            "label": link["label"],
            "title": (
                f"{link['label']}\nSource: {link.get('provenance') or 'unsourced'}"
                f"\nConfidence: {link.get('confidence')}"
            ),
            "arrows": "to",
            "color": {"color": "#94A3B8", "opacity": 0.7},
            "font": {"size": 9, "align": "middle"},
        }
        for link in subgraph["links"]
    ]

    rows = "".join(
        f"<tr><td>{c.get('name', '')}</td><td>{c['uniprot_id']}</td>"
        f"<td>{'Known' if c.get('category') == 'Known/Direct' else 'Hidden'}</td>"
        f"<td>{c.get('corrected_score')}</td>"
        f"<td>{c.get('sourcing_status', '')}</td>"
        f"<td>{c.get('novelty_label', 'Unknown')}</td></tr>"
        for c in candidates
    )
    legend = "".join(
        f'<div class="li"><span class="dot" style="background:{colour}"></span>{label}</div>'
        for label, colour in (
            ("Disease anchor", _VIS_COLOURS["disease"]),
            ("Known/Direct candidate", _KNOWN_COLOUR),
            ("Hidden/Novel candidate", _HIDDEN_COLOUR),
            ("Pathway", _VIS_COLOURS["pathway"]),
            ("Biological process", _VIS_COLOURS["biological_process"]),
            ("Molecular function", _VIS_COLOURS["molecular_function"]),
            ("Protein complex", _VIS_COLOURS["complex"]),
            ("Genetic disorder", _VIS_COLOURS["genetic_disorder"]),
            ("Other protein", _VIS_COLOURS["gene/protein"]),
        )
    )
    metapath_rows = "".join(
        f"<div class='mp'><code>{m['metapath']}</code> × {m['count']}</div>"
        for m in subgraph["metapaths"][:12]
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>TxKG evidence — {disease_name}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:0;padding:18px;background:#fff;color:#0f172a}}
 h2{{color:#0A2E52;margin:0 0 4px}} .sub{{color:#64748b;margin:0 0 14px;font-size:13px}}
 #wrap{{display:flex;gap:18px;align-items:flex-start}}
 #net{{flex:1;height:680px;border:1px solid #e2e8f0;border-radius:8px;background:#fafcfe}}
 #side{{width:330px;max-height:680px;overflow-y:auto}}
 .card{{background:#f6f8fa;border-radius:8px;padding:12px;margin-bottom:12px;font-size:12.5px}}
 .card h3{{margin:0 0 8px;font-size:13px}}
 .li{{display:flex;align-items:center;margin:4px 0}}
 .dot{{width:13px;height:13px;border-radius:50%;margin-right:8px;display:inline-block}}
 table{{width:100%;border-collapse:collapse;font-size:11.5px}}
 th,td{{text-align:left;padding:3px 4px;border-bottom:1px solid #e2e8f0}}
 th{{color:#475569}} .mp{{margin:3px 0;font-size:11px}} code{{background:#e8eef4;padding:1px 4px;border-radius:3px}}
 button{{padding:6px 12px;margin-right:6px;border:0;border-radius:5px;background:#007B82;color:#fff;cursor:pointer}}
</style></head><body>
<h2>Evidence subgraph — {disease_name}</h2>
<p class="sub">Reconstructed connecting paths for the {len(candidates)} candidate(s) that survived
degree-corrected scoring and the sourcing gate. Node outline grey = unconfirmed sourcing.</p>
<div id="wrap">
  <div id="net"></div>
  <div id="side">
    <div class="card"><button onclick="network.fit()">Fit</button>
      <button onclick="network.setOptions({{physics:{{enabled:!phys}}}});phys=!phys">Physics</button></div>
    <div class="card"><h3>Candidates</h3><table>
      <tr><th>Name</th><th>UniProt</th><th>Group</th><th>Score</th><th>Sourcing</th><th>Novelty</th></tr>
      {rows}</table></div>
    <div class="card"><h3>Metapaths</h3>{metapath_rows}</div>
    <div class="card"><h3>Legend</h3>{legend}</div>
  </div>
</div>
<script>
 let phys=true;
 const nodes=new vis.DataSet({json.dumps(vis_nodes)});
 const edges=new vis.DataSet({json.dumps(vis_edges)});
 const network=new vis.Network(document.getElementById('net'),{{nodes,edges}},{{
   nodes:{{shape:'dot',font:{{size:12}}}},
   edges:{{width:1,smooth:{{type:'continuous',roundness:0.4}}}},
   physics:{{solver:'forceAtlas2Based',forceAtlas2Based:{{gravitationalConstant:-60,
     centralGravity:0.012,springLength:170,springConstant:0.08,damping:0.4,avoidOverlap:0.6}},
     stabilization:{{iterations:180}}}},
   interaction:{{hover:true,tooltipDelay:150}}}});
 network.once('stabilizationIterationsDone',()=>network.fit());
</script></body></html>"""

    filename = f"txkg_discovery_{re.sub(r'[^A-Za-z0-9_.-]', '_', disease_id)}.html"
    os.makedirs(kg.GRAPHS_DIR, exist_ok=True)
    with open(os.path.join(kg.GRAPHS_DIR, filename), "w", encoding="utf-8") as fh:
        fh.write(html)
    return filename


# --------------------------------------------------------------------------------------
# TOOL 2 (parts 1–4) — the "is this real and is this new" decision
# --------------------------------------------------------------------------------------


@dataclass
class ScoredCandidates:
    disease_id: str
    disease_name: str
    known: List[Dict[str, Any]] = field(default_factory=list)
    hidden: List[Dict[str, Any]] = field(default_factory=list)
    dropped: int = 0
    dropped_unsourced: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def all_candidates(self) -> List[Dict[str, Any]]:
        return self.known + self.hidden


def score_candidates(
    disease_id: str,
    top_known: int = 25,
    top_hidden: int = 25,
    hidden_threshold: float = HIDDEN_SCORE_THRESHOLD,
    keep_unconfirmed: bool = True,
    reconstruct: bool = True,
) -> ScoredCandidates:
    """
    Tool 2 (§2–§5). Propagate over the full graph, correct for degree bias, split
    Known vs Hidden on the curated-association fact, and run the sourcing gate before
    anything is finalized.
    """
    _require_data()
    gi = build_graph_index()
    disease_name = kg.id_to_name.get(disease_id, disease_id)

    t0 = time.time()
    raw = propagate(disease_id)                                              # §2
    propagation_seconds = time.time() - t0

    # Every protein in the graph is a candidate — no hop cutoff was applied.
    protein_index = kg.PROTEIN_NAME_INDEX
    protein_positions = np.fromiter(
        (i for i, node in enumerate(gi.nodes) if node in protein_index),
        dtype=np.int64,
    )

    if protein_positions.size == 0:
        return ScoredCandidates(disease_id, disease_name, diagnostics={"error": "no protein nodes"})

    protein_raw = raw[protein_positions]
    pool_size = min(CANDIDATE_POOL_SIZE, protein_positions.size)
    top_local = np.argpartition(-protein_raw, pool_size - 1)[:pool_size]
    candidate_idx = protein_positions[top_local]

    correction = degree_corrected_scores(disease_id, raw, candidate_idx)      # §3

    curated = _curated_assoc()
    known: List[Dict[str, Any]] = []
    hidden: List[Dict[str, Any]] = []
    dropped = 0
    dropped_unsourced = 0

    order = np.argsort(-correction["corrected"])
    for rank_pos in order:
        node_i = int(candidate_idx[rank_pos])
        protein_id = gi.nodes[node_i]
        corrected = float(correction["corrected"][rank_pos])
        is_known = disease_id in curated.get(protein_id, ())                  # §4

        if not is_known and corrected < hidden_threshold:
            dropped += 1                                                      # §4 — dropped
            continue
        if is_known and len(known) >= top_known:
            continue
        if not is_known and len(hidden) >= top_hidden:
            continue

        info = kg.PROTEIN_NAME_INDEX.get(protein_id, {})
        record: Dict[str, Any] = {
            "uniprot_id": protein_id,
            "name": info.get("name") or protein_id,
            "gene_name": info.get("gene"),
            "full_name": info.get("full_name"),
            "category": "Known/Direct" if is_known else "Hidden/Novel",
            "raw_propagation_score": float(raw[node_i]),
            "corrected_score": round(corrected, 3),
            "p_value": float(correction["p_value"][rank_pos]),
            "degree": int(correction["degree"][rank_pos]),
            "neighbours_in_disease_region": int(correction["observed"][rank_pos]),
            "expected_by_degree": round(float(correction["expected"][rank_pos]), 2),
            "correction_method": correction["method"],
            "curated_association": is_known,
        }
        if "z_score" in correction:
            record["null_z_score"] = round(float(correction["z_score"][rank_pos]), 3)

        if reconstruct:
            record["paths"] = reconstruct_paths(disease_id, protein_id)       # §6
        else:
            record["paths"] = []

        if is_known:
            record["sourcing_status"] = "curated"
            curated_source = EDGE_PROVENANCE.get(CURATED_DISEASE_ASSOC_EDGE, {}).get(
                "source", "curated disease-gene association database"
            )
            record["sourcing_note"] = (
                f"Backed by a curated {CURATED_DISEASE_ASSOC_EDGE} edge ({curated_source})."
            )
            known.append(record)
            continue

        # ── §5 sourcing gate — Hidden candidates only ─────────────────────────────
        gate = _apply_sourcing_gate(record["paths"])
        record.update(gate)
        if gate["sourcing_status"] == "unsourced" and not keep_unconfirmed:
            dropped_unsourced += 1
            continue
        if gate["sourcing_status"] == "unsourced":
            dropped_unsourced += 1
        hidden.append(record)

    return ScoredCandidates(
        disease_id=disease_id,
        disease_name=disease_name,
        known=known,
        hidden=hidden,
        dropped=dropped,
        dropped_unsourced=dropped_unsourced,
        diagnostics={
            "graph_nodes": len(gi.nodes),
            "graph_edges": int(gi.adjacency.nnz // 2),
            "proteins_propagated_to": int(protein_positions.size),
            "candidate_pool": int(pool_size),
            "restart_probability": RESTART_PROB,
            "layer_jump_probability": LAYER_JUMP_PROB,
            "hop_cutoff": None,
            "hidden_threshold": hidden_threshold,
            "correction_method": correction["method"],
            "propagation_seconds": round(propagation_seconds, 2),
            # The labels were being read off the table with no way to find out
            # what they meant — "Well-explored" against "High" tells a researcher
            # nothing about which is the better lead, or what counts as explored.
            # The thresholds are published with the results rather than living
            # only in this file.
            "label_definitions": label_definitions(hidden_threshold),
        },
    )


def _apply_sourcing_gate(paths: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    §5 — a strong corrected score is a statistical signal, not proof.

    Confirm the edges forming the connection each carry a real source reference. A
    candidate whose every edge on at least one path is properly sourced is confirmed;
    one whose only routes lean on an unsourced or low-confidence edge is passed
    through explicitly marked unconfirmed (or dropped by the caller), never presented
    with the same confidence as a properly evidenced Hidden target.
    """
    if not paths:
        return {
            "sourcing_status": "unsourced",
            "confirmed": False,
            "sourcing_note": "No connecting path could be reconstructed within the path budget — unconfirmed.",
            "supporting_sources": [],
        }

    sourced_paths = [p for p in paths if p["fully_sourced"]]
    if sourced_paths:
        sources = sorted({s for p in sourced_paths for s in p["edge_sources"] if s})
        return {
            "sourcing_status": "confirmed",
            "confirmed": True,
            "sourcing_note": (
                f"{len(sourced_paths)} of {len(paths)} connecting path(s) are sourced end to end "
                f"({', '.join(sources)})."
            ),
            "supporting_sources": sources,
        }

    weakest = min(paths, key=lambda p: p["min_confidence"])
    unsourced_edges = [
        rel or "unknown_edge"
        for rel, conf in zip(weakest["edges"], weakest["edge_confidences"])
        if conf < MIN_SOURCE_CONFIDENCE
    ]
    return {
        "sourcing_status": "unsourced",
        "confirmed": False,
        "sourcing_note": (
            "Every connecting path relies on an unsourced or low-confidence edge "
            f"({', '.join(sorted(set(unsourced_edges))) or 'unknown'}) — reported as unconfirmed."
        ),
        "supporting_sources": [],
    }


# --------------------------------------------------------------------------------------
# TOOL 3 — §7 novelty confirmation, §8 evidence retrieval and interpretation
# --------------------------------------------------------------------------------------

_NOVELTY_BANDS = (
    (0, "High"),          # no existing coverage
    (10, "Moderate"),
    (50, "Low"),
)


def label_definitions(hidden_threshold: float) -> Dict[str, Any]:
    """
    What every label on a target row means, in the same response as the labels.

    Derived from `_NOVELTY_BANDS` rather than restated, so a change to the bands
    cannot leave the published definitions describing the old behaviour.
    """
    bands, previous = [], None
    for ceiling, label in _NOVELTY_BANDS:
        if previous is None:
            rng = f"{ceiling} hits" if ceiling == 0 else f"up to {ceiling} hits"
        else:
            rng = f"{previous + 1}-{ceiling} hits"
        bands.append({"label": label, "range": rng})
        previous = ceiling
    bands.append({"label": "Well-explored", "range": f"more than {previous} hits"})
    bands.append({"label": "Unknown", "range": "hit counts unavailable for this target"})

    return {
        "category": {
            "meaning": "Whether the knowledge graph already records this protein as "
                       "associated with the disease.",
            "values": [
                {"label": "Known/Direct",
                 "definition": f"A curated {CURATED_DISEASE_ASSOC_EDGE} edge already links "
                               "the protein to the disease."},
                {"label": "Hidden/Novel",
                 "definition": "No curated disease-gene edge exists, and the degree-corrected "
                               f"relevance score clears the {hidden_threshold} threshold "
                               "(-log10 p) — the graph connects them, the literature has not "
                               "recorded it as an association."},
            ],
        },
        "noveltyLabel": {
            "meaning": "How much existing patent and literature coverage the protein already "
                       "has for this disease — combined hits, counted independently of the "
                       "graph. High means least explored, so most novel.",
            "values": bands,
        },
        "sourcingStatus": {
            "meaning": "Whether the connecting path survives the edge-level sourcing gate.",
            "values": [
                {"label": "confirmed",
                 "definition": "At least one connecting path is sourced end to end — every "
                               "edge on it comes from a named curated database."},
                {"label": "curated",
                 "definition": "Backed by the curated disease-gene association itself."},
                {"label": "unsourced",
                 "definition": "Every connecting path relies on an unsourced or low-confidence "
                               "edge; reported but flagged as unconfirmed."},
            ],
        },
        "score": {
            "meaning": "Relevance out of 100, relative to the strongest target in the same run. "
                       "`correctedScore` is the underlying -log10(p) from the degree-corrected "
                       "random walk and is unbounded; `score` is what to display.",
        },
    }

#: NCBI allows 3 requests/second without an API key, 10 with one. Exceeding it returns
#: 429s that would silently turn every novelty label into "Unknown", so every eutils
#: call from this module goes through one shared gate.
_EUTILS_MAX_CONCURRENCY = 10 if os.getenv("NCBI_API_KEY") else 2
_EUTILS_MIN_INTERVAL = 0.11 if os.getenv("NCBI_API_KEY") else 0.4

#: Keyed by event loop: an ``asyncio`` primitive is bound to the loop that created it,
#: and reusing one across loops raises "bound to a different event loop". The server runs
#: a single long-lived loop, but tests and scripts call ``asyncio.run`` repeatedly.
_eutils_gates: Dict[int, Tuple[asyncio.Semaphore, asyncio.Lock, List[float]]] = {}


def _eutils_gate() -> Tuple[asyncio.Semaphore, asyncio.Lock, List[float]]:
    try:
        loop_key = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_key = 0
    gate = _eutils_gates.get(loop_key)
    if gate is None:
        gate = (asyncio.Semaphore(_EUTILS_MAX_CONCURRENCY), asyncio.Lock(), [0.0])
        # Keep the map from growing without bound across many short-lived loops.
        if len(_eutils_gates) > 32:
            _eutils_gates.clear()
        _eutils_gates[loop_key] = gate
    return gate


async def _eutils_throttle() -> None:
    """
    Space out consecutive eutils requests to stay inside NCBI's rate limit.

    Delegates to the process-wide gate: NCBI meters per IP, so a limiter private
    to this module only paces TxKG against itself. Running the full pipeline made
    that visible — this module, CurateX and LitMineX each stayed under the limit
    alone while the process as a whole was well over it.
    """
    from DRP_Main.app.shared import eutils

    await eutils.throttle_async()


def search_aliases(candidate: Dict[str, Any]) -> List[str]:
    """
    Search terms for one protein. BioKG's NAME attribute is the UniProt mnemonic
    (PGH2, TNFA), not the HGNC symbol, so searching it alone under-counts badly.
    The descriptive full name is included alongside it.
    """
    aliases: List[str] = []
    for value in (candidate.get("full_name"), candidate.get("gene_name"), candidate.get("name")):
        if not value:
            continue
        text = str(value).replace("UniProt:", "").strip()
        if text and text != candidate.get("uniprot_id") and text not in aliases:
            aliases.append(text)
    return aliases or [candidate["uniprot_id"]]


async def _literature_hit_count(disease_name: str, aliases: Sequence[str]) -> Optional[int]:
    """PubMed result count for 'this protein + this disease'."""
    import httpx

    alias_clause = " OR ".join(f'"{a}"[Title/Abstract]' for a in aliases)
    term = f'("{disease_name}"[Title/Abstract]) AND ({alias_clause})'
    params: Dict[str, Any] = {"db": "pubmed", "term": term, "retmax": 0, "retmode": "json"}
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key

    semaphore, _, _ = _eutils_gate()
    try:
        async with semaphore:
            await _eutils_throttle()
            async with httpx.AsyncClient(timeout=20.0) as http:
                resp = await http.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params
                )
        if resp.status_code != 200:
            print(f"      ⚠️  PubMed count HTTP {resp.status_code} for {aliases[0]}")
            return None
        return int(resp.json().get("esearchresult", {}).get("count", 0))
    except Exception as exc:  # noqa: BLE001 — novelty is advisory, never fatal
        print(f"      ⚠️  PubMed count failed for {aliases[0]}: {exc}")
        return None


async def fetch_articles(disease_name: str, candidate: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """§8 — supporting literature for one candidate, through the shared eutils gate."""
    semaphore, _, _ = _eutils_gate()
    for alias in search_aliases(candidate):
        async with semaphore:
            await _eutils_throttle()
            articles = await kg.fetch_pubmed_articles_for_target(disease_name, alias, limit)
        if articles:
            for article in articles:
                article["matched_alias"] = alias
            return articles
    return []


#: None = untried, True = usable, False = latched off for this process.
_NOVELTY_AGENT_AVAILABLE: Optional[bool] = None


async def _patent_search(disease_name: str, target_name: str) -> Tuple[Optional[int], List[Dict[str, str]]]:
    """
    §7 — cross-reference the candidate against existing patents.

    Prefers the platform's own patent capability (module 5's novelty search agent), which
    already owns the SerpAPI integration, its result cleaning and its usage accounting,
    and returns real patent records rather than a bare count. Falls back to a direct
    SerpAPI call if that module cannot be imported here (it also pulls in Gemini, Qdrant
    and embedding models, none of which are guaranteed in every environment).
    """
    global _NOVELTY_AGENT_AVAILABLE
    query = f'"{target_name}" "{disease_name}"'

    if _NOVELTY_AGENT_AVAILABLE is not False:
        try:
            from DRP_Main.app.api.v1.endpoints.novelty_search_agent import search_patents_only

            results = await search_patents_only(query, num_results=10, rerank=False)
            _NOVELTY_AGENT_AVAILABLE = True
            patents = [
                {"id": r.get("PatentID", ""), "title": r.get("Title", ""), "url": r.get("Link", "")}
                for r in (results or [])
            ]
            return len(patents), patents
        except Exception as exc:  # noqa: BLE001 — heavy optional module; fall back quietly
            # Latched: it pulls in Gemini, Qdrant and embedding models, so if it is not
            # usable here it will not become usable mid-run. Retrying per candidate
            # would only repeat the cost and the noise.
            _NOVELTY_AGENT_AVAILABLE = False
            print(
                f"      ℹ️  Novelty agent unavailable for patents ({str(exc).splitlines()[0]})"
                " — using Europe PMC directly for the rest of this run"
            )

    # Europe PMC's patent corpus (SureChEMBL), not SerpAPI: this runs once per
    # candidate, so a metered $0.015 search was the single largest cost in a run
    # and the first thing to fail when the quota ran out — which silently turned
    # every novelty label into "Unknown".
    import httpx

    terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query or "") if len(t) > 2]
    if not terms:
        return None, []

    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            hits, total = [], None
            for group in ([terms, terms[:2], terms[:1]]):
                resp = await http.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params={
                        "query": f'({" AND ".join(group)}) AND (SRC:"PAT")',
                        "format": "json", "resultType": "core", "pageSize": 10,
                    },
                )
                if resp.status_code != 200:
                    return None, []
                data = resp.json()
                hits = (data.get("resultList") or {}).get("result") or []
                if hits:
                    # `hitCount` is the corpus-wide total, which is what the
                    # novelty label is thresholded against — not the page size.
                    total = data.get("hitCount")
                    break
        patents = [
            {
                "id": str(r.get("id") or ""),
                "title": r.get("title", ""),
                "url": f"https://europepmc.org/article/PAT/{r.get('id')}" if r.get("id") else "",
            }
            for r in hits
        ]
        return (int(total) if total is not None else len(patents)), patents
    except Exception as exc:  # noqa: BLE001
        print(f"      ⚠️  Patent lookup failed for {target_name}: {exc}")
        return None, []


def _novelty_label(total_hits: Optional[int]) -> str:
    if total_hits is None:
        return "Unknown"
    for ceiling, label in _NOVELTY_BANDS:
        if total_hits <= ceiling:
            return label
    return "Well-explored"


async def novelty_label(disease_name: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    §7 — a separate axis from §4's Known/Hidden label. A protein can be biologically
    Hidden (no curated association, strong corrected score) and still carry heavy
    patent/literature activity for other reasons; heavy coverage lowers the novelty
    label, little to no coverage raises it.
    """
    aliases = search_aliases(candidate)

    lit_count, (patent_count, patents) = await asyncio.gather(
        _literature_hit_count(disease_name, aliases),
        _patent_search(disease_name, aliases[0]),
    )
    combined = None
    if lit_count is not None or patent_count is not None:
        combined = (lit_count or 0) + (patent_count or 0)

    return {
        "novelty_label": _novelty_label(combined),
        "literature_hits": lit_count,
        "patent_hits": patent_count,
        "patents": patents[:5],
        "combined_hits": combined,
        "novelty_basis": (
            "inverse of existing "
            + (
                "patent + literature coverage"
                if patent_count is not None
                else "literature coverage (patent search unavailable — set SERPAPI_API_KEY)"
            )
            + " for this protein/disease pair"
        ),
    }


async def interpret_candidates(
    disease_name: str,
    candidates: Sequence[Dict[str, Any]],
    articles_by_target: Dict[str, List[Dict[str, Any]]],
) -> str:
    """
    §8 — hand the retrieved articles, the corrected score, the Known/Hidden label, the
    novelty label and the reconstructed path to the LLM, and ask it to explain in plain
    language why each candidate is being surfaced — grounded in the sourced path and
    the retrieved evidence, not just the number.
    """
    if not candidates:
        return f"No candidate passed scoring and sourcing for {disease_name}."

    blocks: List[str] = []
    for cand in candidates:
        path_txt = "no reconstructed path"
        if cand.get("paths"):
            best = cand["paths"][0]
            path_txt = " → ".join(
                f"{name} [{ntype}]" for name, ntype in zip(best["node_names"], best["node_types"])
            )
            path_txt += f"  (sources: {', '.join(s for s in best['edge_sources'] if s) or 'none'})"

        arts = articles_by_target.get(cand["uniprot_id"], [])
        art_txt = "; ".join(f"{a['title']} (PMID {a['pmid']})" for a in arts[:3]) or "no articles retrieved"

        blocks.append(
            f"- {cand.get('name')} ({cand['uniprot_id']})\n"
            f"    category: {cand['category']}"
            f" | sourcing: {cand.get('sourcing_status', 'curated')}\n"
            f"    corrected relevance score: {cand['corrected_score']} "
            f"(-log10 p; {cand['neighbours_in_disease_region']} neighbours in the disease-relevant "
            f"region vs {cand['expected_by_degree']} expected for degree {cand['degree']})\n"
            f"    novelty: {cand.get('novelty_label', 'Unknown')} "
            f"(literature hits: {cand.get('literature_hits')}, patent hits: {cand.get('patent_hits')})\n"
            f"    connecting path: {path_txt}\n"
            f"    literature: {art_txt}"
        )

    prompt = f"""You are a biomedical research assistant reporting knowledge-graph target predictions for {disease_name}.

Each target below was scored by degree-corrected random-walk-with-restart propagation over the
full knowledge graph, labelled Known/Direct or Hidden/Novel by whether a curated disease-gene
association already exists, and checked for edge-level sourcing.

{chr(10).join(blocks)}

For each target, explain in plain language why it is being surfaced. Ground every claim in the
connecting path and the retrieved literature shown above — do not restate the score as if it were
evidence, and do not introduce facts that are not present above. Note explicitly where a target
is Hidden but already heavily covered in the literature, and where a target is unconfirmed by
the sourcing check. Keep it to 2-3 sentences per target.

These are protein targets, so call them "targets" — never "candidates".

Begin directly with the first target. Do not write an introductory line such as
"Here are the explanations..." — the UI renders this text immediately under the
user's question, so any preamble reads as filler."""

    def _call() -> str:
        # Databricks Model Serving, the same path txkg_test's own interpretation
        # uses. This previously called `kg.client` — the module-level Groq client
        # that was removed when the boot-time `Groq(api_key="")` was made lazy —
        # so every interpretation failed with "no attribute 'client'" and fell
        # through to the factual summary below.
        from DRP_Main.app.core.llm import llm_client

        return llm_client.databricks(
            messages=[
                {
                    "role": "system",
                    "content": "You are a biomedical AI expert in target discovery. You never assert "
                               "anything the provided evidence does not support.",
                },
                {"role": "user", "content": prompt},
            ],
            # 2-3 sentences per target across the whole finalized set. At 900 the
            # reply ran out mid-sentence and the UI showed the cut — a paragraph
            # ending on a bare "Tumor" with nothing after it. Budgeted per target
            # with a floor, so adding targets cannot silently truncate again.
            max_tokens=max(900, 220 * len(candidates)),
            temperature=0.4,
        )

    try:
        # `llm_client.databricks` returns the completion text directly, unlike the
        # OpenAI-shaped response object the Groq client returned.
        return _strip_preamble((await asyncio.to_thread(_call)).strip())
    except Exception as exc:  # noqa: BLE001 — fall back to a factual, ungenerated summary
        print(f"   ⚠️  LLM interpretation unavailable: {exc}")
        lines = [f"LLM interpretation unavailable ({exc}). Factual summary:"]
        for cand in candidates:
            lines.append(
                f"- {cand.get('name')} ({cand['uniprot_id']}): {cand['category']}, "
                f"corrected score {cand['corrected_score']}, "
                f"sourcing {cand.get('sourcing_status', 'curated')}, "
                f"novelty {cand.get('novelty_label', 'Unknown')}."
            )
        return "\n".join(lines)


#: An opening line that only announces what follows ("Here are the explanations for
#: each candidate:"). The UI prints this interpretation directly beneath the user's
#: own question, where such a line is pure filler — and it is the one place the word
#: "candidate" kept reaching the screen. Asking the model not to write one is not
#: reliable on its own, so the lead-in is also removed here.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here\s+(?:are|is)|below\s+(?:are|is)|the\s+following\s+(?:are|is))\b[^\n:]*:\s*\n+",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    """Drop a leading 'Here are the explanations...:' line from an interpretation."""
    return _trim_to_sentence(_PREAMBLE_RE.sub("", text or "", count=1).lstrip())


def _trim_to_sentence(text: str) -> str:
    """
    Drop a trailing half-sentence left by the model hitting its token ceiling.

    The budget above should prevent this, but a long enough run can still reach
    the cap, and a paragraph cut mid-clause reads as a data error rather than a
    length limit. Trimming back to the last complete sentence loses the fragment
    and nothing else. A short reply is left alone — there is nothing to trim back
    to, and returning "" would be worse than returning the fragment.
    """
    cleaned = (text or "").rstrip()
    if not cleaned or cleaned[-1] in ".!?:":
        return cleaned
    cut = max(cleaned.rfind(". "), cleaned.rfind(".\n"),
              cleaned.rfind("! "), cleaned.rfind("? "))
    if cut == -1 or cut < len(cleaned) * 0.5:
        return cleaned
    return cleaned[: cut + 1]


# --------------------------------------------------------------------------------------
# §10 — Recommendation logic
# --------------------------------------------------------------------------------------


def build_ranked_table(scored: ScoredCandidates) -> List[Dict[str, Any]]:
    """
    §9 — the ranked candidate table: one row per candidate, labelled Known/Direct or
    Hidden/Novel, Hidden entries further marked confirmed or unconfirmed, carrying the
    corrected relevance score and the novelty label.
    """
    rows = []
    for cand in sorted(scored.all_candidates(), key=lambda c: -c["corrected_score"]):
        hidden = cand["category"] == "Hidden/Novel"
        label = cand["category"]
        if hidden:
            label += " (confirmed)" if cand.get("confirmed") else " (unconfirmed)"
        rows.append(
            {
                "rank": len(rows) + 1,
                "uniprot_id": cand["uniprot_id"],
                "name": cand.get("name"),
                "gene_name": cand.get("gene_name"),
                "label": label,
                "category": cand["category"],
                "confirmed": cand.get("confirmed", True),
                "corrected_score": cand["corrected_score"],
                "novelty_label": cand.get("novelty_label", "Unknown"),
                "sourcing_status": cand.get("sourcing_status"),
                "path_count": len(cand.get("paths", [])),
            }
        )
    return rows


def relevance_out_of_100(scored: "ScoredCandidates", candidate: Dict[str, Any]) -> str:
    """
    A candidate's relevance as the 0-100 figure the UI shows.

    `corrected_score` is -log10(p) and unbounded — 131.5 in a recent run — so
    quoting it in a sentence aimed at a researcher reads as a broken percentage.
    The API already normalises it against the strongest candidate in the same run
    for the `score` column; this states the recommendation in those same terms so
    the number in the prose matches the number in the table.
    """
    top = max((float(c.get("corrected_score") or 0.0)
               for c in scored.all_candidates()), default=0.0)
    value = float(candidate.get("corrected_score") or 0.0)
    pct = round(100.0 * value / top, 1) if top > 0 else 0.0
    return f"relevance {pct}/100"


def build_recommendation(scored: ScoredCandidates) -> Dict[str, Any]:
    """
    §10 — always names the specific target(s) and states which factors drove it.
    """
    confirmed_hidden = [c for c in scored.hidden if c.get("confirmed")]
    under_explored = [
        c for c in confirmed_hidden if c.get("novelty_label") in ("High", "Moderate")
    ]
    top_known = scored.known[0] if scored.known else None

    # Rule 1 — a strong, sourced, under-explored Hidden candidate takes precedence: it is
    # the least externally validated and gains the most from a closer look.
    if under_explored:
        best = under_explored[0]
        return {
            "action": "literature_mining",
            "next_module": "LitMineX",
            "candidates": [best["uniprot_id"]],
            "text": (
                f"Run a deeper literature check on {best['name']} ({best['uniprot_id']}) next. "
                f"It is Hidden/Novel with {relevance_out_of_100(scored, best)}, its connecting "
                f"path is sourced end to end ({best['sourcing_note']}), and its novelty label is "
                f"{best['novelty_label']} ({best.get('combined_hits')} existing patent/literature hits) — "
                "it is the least externally validated candidate and benefits most from a closer look."
            ),
            "drivers": ["corrected_score", "sourcing_status", "novelty_label"],
        }

    # Rule 3 — both groups have a strong, confirmed entry: name both, let the user choose.
    if confirmed_hidden and top_known:
        best = confirmed_hidden[0]
        return {
            "action": "user_choice",
            "next_module": None,
            "candidates": [best["uniprot_id"], top_known["uniprot_id"]],
            "text": (
                f"Both groups have a strong, confirmed entry: {best['name']} "
                f"({best['uniprot_id']}, Hidden/Novel, {relevance_out_of_100(scored, best)}, "
                f"sourcing {best['sourcing_status']}, novelty {best.get('novelty_label', 'Unknown')}) "
                f"and {top_known['name']} ({top_known['uniprot_id']}, Known/Direct, "
                f"{relevance_out_of_100(scored, top_known)}, backed by a curated "
                "disease association). "
                "Neither Hidden entry is under-explored, so there is no clear winner: the Hidden target "
                "is the less externally validated of the two, the Known target is the safer route into "
                "drug candidate generation."
            ),
            "drivers": ["corrected_score", "sourcing_status", "novelty_label"],
        }

    if confirmed_hidden:
        best = confirmed_hidden[0]
        return {
            "action": "literature_mining",
            "next_module": "LitMineX",
            "candidates": [best["uniprot_id"]],
            "text": (
                f"Run a deeper literature check on {best['name']} ({best['uniprot_id']}). It is the "
                f"strongest confirmed Hidden target ({relevance_out_of_100(scored, best)}, sourced "
                f"path) and there is no Known target to fall back on, though its novelty label is "
                f"{best.get('novelty_label', 'Unknown')} — existing coverage suggests it may already be "
                "actively pursued."
            ),
            "drivers": ["corrected_score", "sourcing_status", "novelty_label"],
        }

    # Rule 2 — Hidden group weak, empty or entirely unconfirmed.
    if top_known:
        names = ", ".join(f"{c['name']} ({c['uniprot_id']})" for c in scored.known[:3])
        reason = (
            "the Hidden group is empty" if not scored.hidden
            else "every Hidden candidate failed the sourcing check and is unconfirmed"
        )
        return {
            "action": "drug_curation",
            "next_module": "DrugCurationX",
            "candidates": [c["uniprot_id"] for c in scored.known[:3]],
            "text": (
                f"Proceed straight to drug candidate generation using the top Known targets: {names}. "
                f"Recommended because {reason}, so there is nothing novel worth a deeper literature "
                f"check first; the Known targets carry curated disease associations and corrected scores "
                f"of {', '.join(str(c['corrected_score']) for c in scored.known[:3])}."
            ),
            "drivers": ["sourcing_status", "curated_association", "corrected_score"],
        }

    return {
        "action": "none",
        "next_module": None,
        "candidates": [],
        "text": (
            f"Nothing surfaced for {scored.disease_name}: no candidate carried a curated association "
            f"and none cleared the corrected-score threshold of {HIDDEN_SCORE_THRESHOLD}. "
            f"{scored.dropped} candidate(s) were dropped as unremarkable."
        ),
        "drivers": ["corrected_score", "curated_association"],
    }


# --------------------------------------------------------------------------------------
# §9 / §12 — full chain, one delivery
# --------------------------------------------------------------------------------------


async def run_discovery(
    disease_query: str,
    top_known: int = 10,
    top_hidden: int = 10,
    interpret_top: int = 5,
    articles_per_target: int = 2,
    hidden_threshold: float = HIDDEN_SCORE_THRESHOLD,
    keep_unconfirmed: bool = True,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Input to output: §1 → §2 → §3 → §4 → §5 → §6 → §7 → §8 → §9 → §10."""
    started = time.time()

    resolved = resolve_disease(disease_query)                                  # §1
    if not resolved.ok:
        return {
            "success": False,
            "resolved": False,
            "clarification": resolved.clarification,
            "query": disease_query,
        }

    scored = await asyncio.to_thread(                                          # §2–§6
        score_candidates,
        resolved.id,
        top_known,
        top_hidden,
        hidden_threshold,
        keep_unconfirmed,
        True,
    )

    # §7 — novelty label for every finalized candidate.
    finalized = scored.all_candidates()
    novelty_results = await asyncio.gather(
        *(novelty_label(resolved.name, c) for c in finalized)
    )
    for cand, nov in zip(finalized, novelty_results):
        cand.update(nov)

    # Re-rank each group by corrected score (novelty does not reorder — it is a separate axis).
    scored.known.sort(key=lambda c: -c["corrected_score"])
    scored.hidden.sort(key=lambda c: (not c.get("confirmed"), -c["corrected_score"]))

    # §8 — retrieve supporting literature for the top of both groups, then interpret.
    interpret_set = scored.known[:interpret_top] + scored.hidden[:interpret_top]
    article_lists = await asyncio.gather(
        *(fetch_articles(resolved.name, c, articles_per_target) for c in interpret_set)
    )
    articles_by_target = {
        c["uniprot_id"]: arts for c, arts in zip(interpret_set, article_lists)
    }
    for cand in interpret_set:
        cand["articles"] = articles_by_target.get(cand["uniprot_id"], [])

    interpretation = await interpret_candidates(resolved.name, interpret_set, articles_by_target)

    recommendation = build_recommendation(scored)                              # §10
    subgraph = build_candidate_subgraph(resolved.id, finalized)                # §6/§9

    # §9 — one ranked table, each row labelled, Hidden rows further marked
    # confirmed/unconfirmed. Known and Hidden stay available separately for callers
    # that render the two groups side by side.
    table = build_ranked_table(scored)

    html_url = None
    try:
        filename = render_candidate_subgraph_html(
            resolved.id, resolved.name, subgraph, finalized
        )
        html_url = f"/api/v1/graphs/discovery/{filename}"
    except Exception as exc:  # noqa: BLE001 — the visual is never worth failing the run
        print(f"   ⚠️  Could not render the evidence subgraph: {exc}")
    subgraph["html_url"] = html_url

    payload = {                                                                # §9
        "success": True,
        "resolved": True,
        "disease": {"id": resolved.id, "name": resolved.name, "resolution_method": resolved.method},
        "table": table,
        "candidates": {
            "known": scored.known,
            "hidden": scored.hidden,
        },
        "counts": {
            "known": len(scored.known),
            "hidden": len(scored.hidden),
            "hidden_confirmed": sum(1 for c in scored.hidden if c.get("confirmed")),
            "hidden_unconfirmed": sum(1 for c in scored.hidden if not c.get("confirmed")),
            "dropped_weak": scored.dropped,
            "dropped_or_flagged_unsourced": scored.dropped_unsourced,
        },
        "interpretation": interpretation,
        "recommendation": recommendation,
        "subgraph": subgraph,
        "method": scored.diagnostics,
        "elapsed_seconds": round(time.time() - started, 2),
    }

    if session_id:
        SESSION_STATE[session_id] = payload                                    # §11
        payload["session_id"] = session_id
    return payload


# --------------------------------------------------------------------------------------
# §11 — supervising layer: session state + scoped follow-ups
# --------------------------------------------------------------------------------------

#: session_id → the last full discovery payload. A follow-up about an already-scored
#: disease is answered from here; only a fresh disease restarts the chain at §1.
SESSION_STATE: Dict[str, Dict[str, Any]] = {}


def needs_full_chain(session_id: Optional[str], disease_query: Optional[str]) -> bool:
    """A fresh disease needs the whole chain; anything else is a scoped follow-up."""
    if not session_id or session_id not in SESSION_STATE:
        return True
    if not disease_query:
        return False
    resolved = resolve_disease(disease_query)
    if not resolved.ok:
        # No new disease named — a question about the active one, not a fresh run.
        return False
    return resolved.id != SESSION_STATE[session_id]["disease"]["id"]


def answer_followup(session_id: str, uniprot_id: str) -> Dict[str, Any]:
    """
    §11 — 'why is this one Hidden?' reads Tool 2's already-computed scores and Tool 1's
    already-reconstructed path. No rerun.
    """
    state = SESSION_STATE.get(session_id)
    if state is None:
        raise KeyError(f"No active TxKG session '{session_id}'")

    for group in ("known", "hidden"):
        for cand in state["candidates"][group]:
            if cand["uniprot_id"].lower() == uniprot_id.lower():
                explanation = (
                    f"{cand['name']} ({cand['uniprot_id']}) is labelled {cand['category']} for "
                    f"{state['disease']['name']} because "
                    + (
                        f"a curated {CURATED_DISEASE_ASSOC_EDGE} edge already links them — it is "
                        "documented, not a new finding."
                        if cand["curated_association"]
                        else f"no curated {CURATED_DISEASE_ASSOC_EDGE} edge links them, while its "
                             f"degree-corrected relevance score of {cand['corrected_score']} "
                             f"(-log10 p) clears the {state['method']['hidden_threshold']} threshold: "
                             f"it has {cand['neighbours_in_disease_region']} neighbours inside the "
                             f"disease-relevant region of the graph against "
                             f"{cand['expected_by_degree']} expected for a node of degree "
                             f"{cand['degree']}."
                    )
                    + f" Sourcing: {cand.get('sourcing_note', 'curated association')} "
                    + f"Novelty: {cand.get('novelty_label', 'Unknown')} "
                      f"({cand.get('literature_hits')} literature, {cand.get('patent_hits')} patent hits)."
                )
                return {
                    "session_id": session_id,
                    "disease": state["disease"],
                    "candidate": cand,
                    "explanation": explanation,
                    "recomputed": False,
                }
    raise KeyError(f"'{uniprot_id}' is not among the scored candidates for this session")


_UNIPROT_RE = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b")


async def handle_message(message: str, session_id: str = "default", **kwargs) -> Dict[str, Any]:
    """
    §11 — the supervising layer's entry point.

    Decides whether an incoming message needs the full chain (a fresh disease) or a
    scoped follow-up against results already produced. A follow-up naming a candidate
    ("why is this one Hidden?") is answered from Tool 2's computed scores and Tool 1's
    reconstructed paths without rerunning anything.
    """
    text = (message or "").strip()
    state = SESSION_STATE.get(session_id)

    # A candidate named in the message, against an active session → scoped follow-up.
    if state is not None:
        accessions = _UNIPROT_RE.findall(text.upper())
        mentioned = list(accessions)
        if not mentioned:
            known_ids = {
                c["uniprot_id"].upper()
                for group in ("known", "hidden")
                for c in state["candidates"][group]
            }
            mentioned = [t for t in re.findall(r"[A-Za-z0-9]+", text.upper()) if t in known_ids]
        # Also match by gene or display name.
        if not mentioned:
            lowered = text.lower()
            for group in ("known", "hidden"):
                for cand in state["candidates"][group]:
                    for alias in filter(None, (cand.get("gene_name"), cand.get("name"))):
                        if len(str(alias)) > 2 and str(alias).lower() in lowered:
                            mentioned = [cand["uniprot_id"]]
                            break
                    if mentioned:
                        break
                if mentioned:
                    break
        if mentioned:
            try:
                answer = answer_followup(session_id, mentioned[0])
                return {"success": True, "mode": "followup", **answer}
            except KeyError:
                # A specific accession that was not scored here: say so rather than
                # answering about something else, and rather than silently rerunning.
                if accessions:
                    return {
                        "success": False,
                        "mode": "not_in_results",
                        "session_id": session_id,
                        "disease": state["disease"],
                        "requested": accessions[0],
                        "message": (
                            f"{accessions[0]} is not among the candidates scored for "
                            f"{state['disease']['name']} in this session. It was either dropped as "
                            "unremarkable after degree correction or falls outside the reported top "
                            "candidates — rerun with a larger top_known/top_hidden to see further "
                            "down the ranking."
                        ),
                        "scored_candidates": [
                            c["uniprot_id"]
                            for group in ("known", "hidden")
                            for c in state["candidates"][group]
                        ],
                    }

    if not needs_full_chain(session_id, text):
        # Same disease, no specific candidate — restate what was already computed.
        return {
            "success": True,
            "mode": "recap",
            "recomputed": False,
            "session_id": session_id,
            **{k: v for k, v in (state or {}).items() if k != "session_id"},
        }

    payload = await run_discovery(text, session_id=session_id, **kwargs)
    payload["mode"] = "full_chain"
    return payload


def clear_discovery_caches() -> Dict[str, int]:
    """Drop the query-time caches. The transition matrix is rebuilt on next use."""
    global _GRAPH_INDEX, _CURATED_ASSOC, _NULL_MODEL, _NULL_MODEL_LOADED
    counts = {"rwr_vectors": len(_RWR_CACHE), "sessions": len(SESSION_STATE)}
    _RWR_CACHE.clear()
    SESSION_STATE.clear()
    _GRAPH_INDEX = None
    _CURATED_ASSOC = None
    _NULL_MODEL = None
    _NULL_MODEL_LOADED = False
    return counts
