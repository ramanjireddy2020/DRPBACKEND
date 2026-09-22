"""
TxKG discovery endpoints — the spec's three tools plus the supervising layer.

    Tool 1  /txkg/tools/resolve-disease   §1
            /txkg/tools/paths             §6
    Tool 2  /txkg/tools/score             §2–§5
    Tool 3  /txkg/tools/evidence          §7–§8
            /txkg/discover                §9/§12 full chain, one delivery
            /txkg/explain                 §11 scoped follow-up, no rerun
"""
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from DRP_Main.app.modules.txkg import discovery_service as ds

router = APIRouter()


# ── Tool 1 ────────────────────────────────────────────────────────────────────────────


@router.get("/txkg/tools/resolve-disease")
def resolve_disease_endpoint(disease: str = Query(..., description="Disease name, typed freely")):
    """Tool 1 (§1) — free text to one confirmed anchor node, or one clarifying question."""
    resolved = ds.resolve_disease(disease)
    if not resolved.ok:
        return {
            "success": False,
            "resolved": False,
            "query": disease,
            "clarification": resolved.clarification,
        }
    return {
        "success": True,
        "resolved": True,
        "query": disease,
        "disease": {"id": resolved.id, "name": resolved.name, "resolution_method": resolved.method},
    }


@router.get("/txkg/tools/paths")
def paths_endpoint(
    disease: str = Query(..., description="Disease name"),
    uniprot_id: str = Query(..., description="A candidate that already survived scoring"),
    max_hops: int = Query(ds.PATH_MAX_HOPS, ge=1, le=5),
    max_paths: int = Query(ds.PATH_MAX_PER_TARGET, ge=1, le=25),
):
    """
    Tool 1 (§6) — bounded BFS from the disease to one already-selected candidate.

    This does not decide who gets scored; it redraws the connecting path (and its edge
    provenance) for a candidate that already made it through Tool 2.
    """
    resolved = ds.resolve_disease(disease)
    if not resolved.ok:
        raise HTTPException(status_code=404, detail=resolved.clarification)

    paths = ds.reconstruct_paths(resolved.id, uniprot_id, max_hops=max_hops, max_paths=max_paths)
    return {
        "success": True,
        "disease": {"id": resolved.id, "name": resolved.name},
        "uniprot_id": uniprot_id,
        "path_count": len(paths),
        "paths": paths,
        "subgraph": ds.build_candidate_subgraph(
            resolved.id, [{"uniprot_id": uniprot_id, "paths": paths}]
        ),
    }


# ── Tool 2 ────────────────────────────────────────────────────────────────────────────


@router.get("/txkg/tools/score")
def score_endpoint(
    disease: str = Query(..., description="Disease name"),
    top_known: int = Query(25, ge=1, le=200),
    top_hidden: int = Query(25, ge=1, le=200),
    hidden_threshold: float = Query(
        ds.HIDDEN_SCORE_THRESHOLD, description="Minimum corrected score (-log10 p) for a Hidden candidate"
    ),
    keep_unconfirmed: bool = Query(True, description="Keep unsourced Hidden candidates, flagged"),
    reconstruct_paths: bool = Query(True, description="Reconstruct paths (required by the §5 gate)"),
):
    """
    Tool 2 (§2–§5) — full-graph RWRH propagation, degree correction, Known/Hidden
    categorization against the curated disease-association edge, and the sourcing gate.
    This is where the 'is this real and is this new' decision is made.
    """
    resolved = ds.resolve_disease(disease)
    if not resolved.ok:
        raise HTTPException(status_code=404, detail=resolved.clarification)

    try:
        scored = ds.score_candidates(
            resolved.id,
            top_known=top_known,
            top_hidden=top_hidden,
            hidden_threshold=hidden_threshold,
            keep_unconfirmed=keep_unconfirmed,
            reconstruct=reconstruct_paths,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    return {
        "success": True,
        "disease": {"id": resolved.id, "name": resolved.name},
        "candidates": {"known": scored.known, "hidden": scored.hidden},
        "counts": {
            "known": len(scored.known),
            "hidden": len(scored.hidden),
            "hidden_confirmed": sum(1 for c in scored.hidden if c.get("confirmed")),
            "dropped_weak": scored.dropped,
            "dropped_or_flagged_unsourced": scored.dropped_unsourced,
        },
        "method": scored.diagnostics,
    }


# ── Tool 3 ────────────────────────────────────────────────────────────────────────────


@router.get("/txkg/tools/evidence")
async def evidence_endpoint(
    disease: str = Query(..., description="Disease name"),
    uniprot_ids: str = Query(..., description="Comma-separated finalized candidates"),
    articles_per_target: int = Query(2, ge=1, le=10),
):
    """
    Tool 3 (§7–§8) — novelty label from existing patent/literature coverage, supporting
    literature retrieval, and a plain-language interpretation grounded in the sourced
    path and the retrieved evidence.
    """
    import asyncio

    from DRP_Main.app.api.v1.endpoints import txkg_test as kg

    resolved = ds.resolve_disease(disease)
    if not resolved.ok:
        raise HTTPException(status_code=404, detail=resolved.clarification)

    ids = [i.strip() for i in uniprot_ids.split(",") if i.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="uniprot_ids cannot be empty")

    candidates = []
    for uid in ids:
        info = kg.PROTEIN_NAME_INDEX.get(uid, {})
        is_known = ds.has_curated_association(uid, resolved.id)
        paths = ds.reconstruct_paths(resolved.id, uid)
        cand = {
            "uniprot_id": uid,
            "name": info.get("name") or uid,
            "gene_name": info.get("gene"),
            "full_name": info.get("full_name"),
            "category": "Known/Direct" if is_known else "Hidden/Novel",
            "curated_association": is_known,
            "corrected_score": None,
            "degree": None,
            "neighbours_in_disease_region": None,
            "expected_by_degree": None,
            "paths": paths,
        }
        cand.update(
            {"sourcing_status": "curated", "confirmed": True}
            if is_known
            else ds._apply_sourcing_gate(paths)
        )
        candidates.append(cand)

    novelty = await asyncio.gather(*(ds.novelty_label(resolved.name, c) for c in candidates))
    for cand, nov in zip(candidates, novelty):
        cand.update(nov)

    article_lists = await asyncio.gather(
        *(ds.fetch_articles(resolved.name, c, articles_per_target) for c in candidates)
    )
    articles_by_target = {c["uniprot_id"]: a for c, a in zip(candidates, article_lists)}
    for cand in candidates:
        cand["articles"] = articles_by_target[cand["uniprot_id"]]

    interpretation = await ds.interpret_candidates(resolved.name, candidates, articles_by_target)
    return {
        "success": True,
        "disease": {"id": resolved.id, "name": resolved.name},
        "candidates": candidates,
        "interpretation": interpretation,
    }


# ── Supervising layer ─────────────────────────────────────────────────────────────────


@router.get("/txkg/discover")
async def discover_endpoint(
    disease: str = Query(..., description="Disease name, typed freely"),
    top_known: int = Query(10, ge=1, le=100),
    top_hidden: int = Query(10, ge=1, le=100),
    interpret_top: int = Query(5, ge=1, le=20, description="Candidates per group to interpret"),
    articles_per_target: int = Query(2, ge=1, le=10),
    hidden_threshold: float = Query(ds.HIDDEN_SCORE_THRESHOLD),
    keep_unconfirmed: bool = Query(True),
    session_id: Optional[str] = Query(None, description="Reuse for scoped follow-ups"),
):
    """
    §12 full flow, §9 single delivery: the labelled ranked table (Known/Direct vs
    Hidden/Novel, Hidden further marked confirmed or unconfirmed), corrected relevance
    scores, novelty labels, the plain-language interpretation, the recommendation for
    what to do next, and the collapsed subgraph/metapath visual.
    """
    try:
        return await ds.run_discovery(
            disease,
            top_known=top_known,
            top_hidden=top_hidden,
            interpret_top=interpret_top,
            articles_per_target=articles_per_target,
            hidden_threshold=hidden_threshold,
            keep_unconfirmed=keep_unconfirmed,
            session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TxKG discovery failed: {exc}")


@router.get("/txkg/explain")
def explain_endpoint(
    session_id: str = Query(..., description="A session from /txkg/discover"),
    uniprot_id: str = Query(..., description="A candidate already scored in that session"),
):
    """§11 — 'why is this one Hidden?' answered from the already-computed scores and
    already-reconstructed paths. Nothing is recomputed."""
    try:
        return {"success": True, **ds.answer_followup(session_id, uniprot_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/txkg/ask")
async def ask_endpoint(
    message: str = Query(..., description="Free-text message, e.g. 'targets for asthma' or 'why is COX-2 hidden?'"),
    session_id: str = Query("default", description="Conversation/session key"),
    top_known: int = Query(10, ge=1, le=100),
    top_hidden: int = Query(10, ge=1, le=100),
):
    """
    §11 — the supervising layer. Routes an incoming message to the full chain (fresh
    disease) or to a scoped follow-up against results already produced. The response's
    `mode` says which happened: `full_chain`, `followup` or `recap`.
    """
    try:
        return await ds.handle_message(
            message, session_id=session_id, top_known=top_known, top_hidden=top_hidden
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TxKG supervisor failed: {exc}")


@router.get("/graphs/discovery/{filename}")
def serve_discovery_graph(filename: str):
    """Serve a rendered evidence subgraph (§6/§9) — the collapsed visual, once opened."""
    import os

    from fastapi.responses import FileResponse

    from DRP_Main.app.api.v1.endpoints import txkg_test as kg

    if not re.fullmatch(r"txkg_discovery_[A-Za-z0-9_.-]+\.html", filename):
        raise HTTPException(status_code=400, detail="Not a discovery graph filename")

    path = os.path.join(kg.GRAPHS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404, detail="Not generated yet — call /txkg/discover for this disease first."
        )
    return FileResponse(
        path,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Access-Control-Allow-Origin": "*"},
    )


@router.get("/txkg/discovery-health")
def discovery_health():
    """Configuration and readiness of the spec implementation."""
    from DRP_Main.app.api.v1.endpoints import txkg_test as kg

    null_present = ds.os.path.exists(ds._NULL_MODEL_PATH)
    matrix_present = ds.os.path.exists(ds._MATRIX_CACHE_PATH)
    return {
        "success": True,
        "data_loaded": kg.df_links is not None,
        "graph_index_built": ds._GRAPH_INDEX is not None,
        "transition_matrix_cached": matrix_present,
        "null_model_present": null_present,
        "config": {
            "restart_probability": ds.RESTART_PROB,
            "layer_jump_probability": ds.LAYER_JUMP_PROB,
            "hop_cutoff": None,
            "candidate_pool_size": ds.CANDIDATE_POOL_SIZE,
            "hidden_score_threshold": ds.HIDDEN_SCORE_THRESHOLD,
            "curated_association_edge": ds.CURATED_DISEASE_ASSOC_EDGE,
            "min_source_confidence": ds.MIN_SOURCE_CONFIDENCE,
            "annotation_edges_enabled": sorted(ds.ENABLED_ANNOTATION_EDGES),
            "annotation_edges_available": sorted(
                a for attrs in ds.ANNOTATION_FILES.values() for a in attrs
            ),
            "node_types": (
                ds._GRAPH_INDEX.layer_names if ds._GRAPH_INDEX is not None else None
            ),
            "degree_correction": (
                "hypergeometric_degree_conditioned"
                + ("+precomputed_null_z" if null_present else "")
            ),
        },
        "edge_provenance_registry": ds.EDGE_PROVENANCE,
        "active_sessions": len(ds.SESSION_STATE),
    }


@router.post("/txkg/discovery-clear-cache")
def discovery_clear_cache():
    """Drop RWR score vectors, sessions and the in-memory graph index."""
    return {"success": True, "cleared": ds.clear_discovery_caches()}
