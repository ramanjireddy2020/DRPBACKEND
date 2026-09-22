"""
TxKG discovery — §13 calibration / testing notes from the functional spec.

These exercise the real BioKG dataset, so they skip cleanly when it is absent.
Nothing here touches the network: novelty (§7) and interpretation (§8) are excluded.

    1. A disease with well-known targets stays stable and correctly labelled Known.
    2. A target reachable only through indirect network evidence scores as Hidden.
    3. A generic hub protein scores high raw but unremarkable after degree correction.
    4. A candidate whose connecting path relies on an unsourced edge is dropped or
       marked unconfirmed by the sourcing gate — never presented as confirmed.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def ds():
    service = pytest.importorskip("DRP_Main.app.modules.txkg.discovery_service")
    from DRP_Main.app.api.v1.endpoints import txkg_test as kg

    if kg.df_links is None:
        pytest.skip("BioKG dataset not available in this environment")
    return service


@pytest.fixture(scope="module")
def asthma(ds):
    resolved = ds.resolve_disease("asthma")
    if not resolved.ok:
        pytest.skip("Asthma not present in this build of the graph")
    return resolved


# ── §1 disease resolution ─────────────────────────────────────────────────────────────


def test_resolution_produces_one_anchor(ds, asthma):
    assert asthma.id
    assert asthma.name.lower() == "asthma"


def test_unresolvable_disease_asks_one_question_instead_of_guessing(ds):
    resolved = ds.resolve_disease("zzzz not a disease qqqq")
    assert not resolved.ok
    assert resolved.clarification
    assert not resolved.id


# ── §2 propagation reaches the whole graph, with no hop cutoff ────────────────────────


def test_propagation_covers_the_full_graph(ds, asthma):
    scores = ds.propagate(asthma.id)
    gi = ds.build_graph_index()

    assert scores.shape == (len(gi.nodes),)
    assert np.isclose(scores.sum(), 1.0, atol=1e-6)

    # A hop-bounded search would leave whole regions at exactly zero. Restart decay
    # instead leaves distant nodes with small-but-nonzero signal.
    reached = int((scores > 0).sum())
    assert reached > len(gi.nodes) * 0.5, f"only {reached}/{len(gi.nodes)} nodes received signal"


# ── §13 case 1: known disease stays stable and correctly labelled ─────────────────────


def test_known_targets_are_labelled_known_and_ranking_is_stable(ds, asthma):
    first = ds.score_candidates(asthma.id, top_known=10, top_hidden=10, reconstruct=False)
    assert first.known, "asthma should surface curated Known targets"

    for cand in first.known:
        assert cand["category"] == "Known/Direct"
        assert cand["curated_association"] is True
        assert ds.has_curated_association(cand["uniprot_id"], asthma.id)

    for cand in first.hidden:
        assert cand["category"] == "Hidden/Novel"
        assert cand["curated_association"] is False
        assert not ds.has_curated_association(cand["uniprot_id"], asthma.id)

    # Deterministic: the same disease scored twice gives the same ranking.
    second = ds.score_candidates(asthma.id, top_known=10, top_hidden=10, reconstruct=False)
    assert [c["uniprot_id"] for c in first.known] == [c["uniprot_id"] for c in second.known]
    assert [c["uniprot_id"] for c in first.hidden] == [c["uniprot_id"] for c in second.hidden]


# ── §13 case 2: indirect network evidence surfaces as Hidden ──────────────────────────


def test_hidden_group_is_reached_indirectly_not_by_a_curated_edge(ds, asthma):
    scored = ds.score_candidates(asthma.id, top_known=5, top_hidden=5, reconstruct=True)
    assert scored.hidden, "asthma should surface Hidden candidates"

    for cand in scored.hidden:
        assert cand["corrected_score"] >= ds.HIDDEN_SCORE_THRESHOLD
        for path in cand["paths"]:
            # A curated edge to some *other* protein is fine mid-path — that is what
            # "indirect network evidence" looks like. What must not exist is a curated
            # association from the disease straight to this candidate.
            if path["hop_count"] == 1:
                assert path["edges"][0] != ds.CURATED_DISEASE_ASSOC_EDGE


# ── §13 case 3: the degree correction neutralises a generic hub ────────────────────────


def test_degree_correction_demotes_the_most_connected_protein(ds, asthma):
    gi = ds.build_graph_index()
    raw = ds.propagate(asthma.id)

    protein_positions = np.fromiter(
        (i for i, node in enumerate(gi.nodes) if node in ds.kg.PROTEIN_NAME_INDEX),
        dtype=np.int64,
    )
    # The single most-connected human protein in the graph — a generic hub.
    hub_i = int(protein_positions[np.argmax(gi.degree[protein_positions])])

    pool = np.argsort(-raw[protein_positions])[: ds.CANDIDATE_POOL_SIZE]
    candidate_idx = protein_positions[pool]
    if hub_i not in set(candidate_idx.tolist()):
        candidate_idx = np.append(candidate_idx, hub_i)

    correction = ds.degree_corrected_scores(asthma.id, raw, candidate_idx)
    order_raw = np.argsort(-raw[candidate_idx])
    order_corrected = np.argsort(-correction["corrected"])
    where = {int(candidate_idx[j]): pos for pos, j in enumerate(order_raw)}
    where_corrected = {int(candidate_idx[j]): pos for pos, j in enumerate(order_corrected)}

    hub_raw_rank = where[hub_i]
    hub_corrected_rank = where_corrected[hub_i]

    assert hub_raw_rank < len(candidate_idx) * 0.10, (
        "the hub should rank near the top on raw propagation score for this test to mean anything "
        f"(raw rank {hub_raw_rank}/{len(candidate_idx)})"
    )
    assert hub_corrected_rank > hub_raw_rank, (
        f"degree correction did not demote the hub: raw rank {hub_raw_rank} → "
        f"corrected rank {hub_corrected_rank}"
    )


def test_correction_reports_observed_against_degree_expectation(ds, asthma):
    scored = ds.score_candidates(asthma.id, top_known=3, top_hidden=3, reconstruct=False)
    for cand in scored.all_candidates():
        assert cand["degree"] > 0
        assert cand["neighbours_in_disease_region"] >= 0
        # A surfaced candidate beats its degree-based expectation, by construction.
        assert cand["neighbours_in_disease_region"] > cand["expected_by_degree"]
        assert 0.0 < cand["p_value"] <= 1.0


# ── §13 case 4: the sourcing gate ─────────────────────────────────────────────────────


def test_sourcing_gate_confirms_a_fully_sourced_path(ds, asthma):
    scored = ds.score_candidates(asthma.id, top_known=1, top_hidden=5, reconstruct=True)
    for cand in scored.hidden:
        if cand["sourcing_status"] == "confirmed":
            assert cand["confirmed"] is True
            assert cand["supporting_sources"]
            return
    pytest.skip("no confirmed Hidden candidate in this run")


def test_sourcing_gate_marks_an_unsourced_path_unconfirmed(ds, monkeypatch, asthma):
    """A path leaning on an edge type with no curated source must never read confirmed."""
    scored = ds.score_candidates(asthma.id, top_known=1, top_hidden=1, reconstruct=True)
    if not scored.hidden or not scored.hidden[0]["paths"]:
        pytest.skip("no Hidden candidate with a reconstructed path in this run")

    # Strip one edge type out of the provenance registry, exactly as an edge with no
    # source reference would appear, and re-reconstruct.
    protein_id = scored.hidden[0]["uniprot_id"]
    registry = dict(ds.EDGE_PROVENANCE)
    for rel in {e for p in scored.hidden[0]["paths"] for e in p["edges"]}:
        registry.pop(rel, None)
    monkeypatch.setattr(ds, "EDGE_PROVENANCE", registry)

    paths = ds.reconstruct_paths(asthma.id, protein_id)
    gate = ds._apply_sourcing_gate(paths)
    assert gate["sourcing_status"] == "unsourced"
    assert gate["confirmed"] is False
    assert not gate["supporting_sources"]


def test_unsourced_candidates_can_be_dropped_outright(ds, monkeypatch, asthma):
    monkeypatch.setattr(ds, "EDGE_PROVENANCE", {})
    scored = ds.score_candidates(
        asthma.id, top_known=3, top_hidden=5, keep_unconfirmed=False, reconstruct=True
    )
    assert scored.hidden == [], "with no sourced edge type, no Hidden candidate may survive"
    assert scored.dropped_unsourced > 0
    # Known candidates are unaffected: their label rests on the curated edge itself.
    assert scored.known


# ── §6 path reconstruction runs only for survivors, and is a real route ───────────────


def test_reconstructed_paths_avoid_drug_and_disease_intermediates(ds, asthma):
    scored = ds.score_candidates(asthma.id, top_known=2, top_hidden=2, reconstruct=True)
    for cand in scored.all_candidates():
        for path in cand["paths"]:
            assert path["nodes"][0] == asthma.id
            assert path["nodes"][-1] == cand["uniprot_id"]
            assert len(path["nodes"]) == len(path["edges"]) + 1
            for node_type in path["node_types"][1:-1]:
                assert node_type not in ("drug", "disease")


def test_candidate_subgraph_is_built_only_from_survivors(ds, asthma):
    scored = ds.score_candidates(asthma.id, top_known=2, top_hidden=2, reconstruct=True)
    finalized = scored.all_candidates()
    subgraph = ds.build_candidate_subgraph(asthma.id, finalized)

    assert subgraph["collapsed"] is True
    assert subgraph["statistics"]["total_nodes"] <= 400, "the whole graph must not be drawn"
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert asthma.id in node_ids
    for cand in finalized:
        assert cand["uniprot_id"] in node_ids
    assert subgraph["metapaths"], "metapath summary should not be empty"


# ── §0 the graph carries the node types the spec requires ─────────────────────────────


def test_graph_includes_process_node_types(ds):
    """
    §0 lists biological process among the node types and §6 promises to show processes.
    biokg.links.tsv has none — they come from the biokg.properties.* annotation layers.
    """
    if not ds.ENABLED_ANNOTATION_EDGES:
        pytest.skip("annotation layers disabled via TXKG_ANNOTATION_EDGES")

    gi = ds.build_graph_index()
    assert "biological_process" in gi.layer_names, gi.layer_names
    assert "gene/protein" in gi.layer_names
    assert "pathway" in gi.layer_names

    # And they are reachable, not merely present.
    processes = [n for n in gi.nodes if ds.node_type(n) == "biological_process"]
    assert len(processes) > 1000
    assert gi.degree[gi.index[processes[0]]] > 0


def test_annotation_nodes_do_not_mutate_legacy_state(ds):
    """The legacy context-graph endpoints must be unaffected by the annotation layers."""
    from DRP_Main.app.api.v1.endpoints import txkg_test as kg

    ds.build_graph_index()
    processes = [n for n in ds._ANNOT_TYPE if ds._ANNOT_TYPE[n] == "biological_process"]
    assert processes, "expected annotation nodes to have been loaded"
    for node in processes[:50]:
        assert node not in kg.id_to_type
        assert node not in kg.ctx_adj


def test_paths_can_traverse_a_biological_process(ds, asthma):
    """A process should be usable as a connecting intermediate, per §6."""
    if not ds.ENABLED_ANNOTATION_EDGES:
        pytest.skip("annotation layers disabled")

    scored = ds.score_candidates(asthma.id, top_known=8, top_hidden=8, reconstruct=True)
    seen_types = {
        node_type
        for cand in scored.all_candidates()
        for path in cand["paths"]
        for node_type in path["node_types"]
    }
    assert seen_types & {"biological_process", "molecular_function"}, (
        f"no process/function intermediate appeared on any path: {seen_types}"
    )


# ── §9 output assembly ────────────────────────────────────────────────────────────────


def test_ranked_table_labels_every_row(ds, asthma):
    scored = ds.score_candidates(asthma.id, top_known=3, top_hidden=3, reconstruct=True)
    for cand in scored.hidden:
        cand.setdefault("novelty_label", "Unknown")
    table = ds.build_ranked_table(scored)

    assert table
    assert [r["rank"] for r in table] == list(range(1, len(table) + 1))
    # Ranked by corrected score, descending.
    scores = [r["corrected_score"] for r in table]
    assert scores == sorted(scores, reverse=True)

    for row in table:
        assert row["label"] in (
            "Known/Direct",
            "Hidden/Novel (confirmed)",
            "Hidden/Novel (unconfirmed)",
        )
        if row["category"] == "Hidden/Novel":
            expected = "confirmed" if row["confirmed"] else "unconfirmed"
            assert expected in row["label"]


def test_evidence_subgraph_renders_an_openable_page(ds, asthma, tmp_path):
    scored = ds.score_candidates(asthma.id, top_known=2, top_hidden=2, reconstruct=True)
    finalized = scored.all_candidates()
    subgraph = ds.build_candidate_subgraph(asthma.id, finalized)

    filename = ds.render_candidate_subgraph_html(asthma.id, asthma.name, subgraph, finalized)
    assert filename.startswith("txkg_discovery_")
    assert filename.endswith(".html")

    import os

    path = os.path.join(ds.kg.GRAPHS_DIR, filename)
    html = open(path, encoding="utf-8").read()
    assert "vis-network" in html
    assert asthma.name in html
    for cand in finalized:
        assert cand["uniprot_id"] in html
    # The provenance of each drawn edge must be visible, per §5/§9.
    assert "Source:" in html


# ── §7 novelty banding is inverse to coverage ─────────────────────────────────────────


def test_novelty_label_is_inverse_to_existing_coverage(ds):
    assert ds._novelty_label(0) == "High"
    assert ds._novelty_label(5) == "Moderate"
    assert ds._novelty_label(30) == "Low"
    assert ds._novelty_label(5000) == "Well-explored"
    assert ds._novelty_label(None) == "Unknown"


# ── §10 recommendation logic ──────────────────────────────────────────────────────────


def _candidate(uid, category, corrected, confirmed=True, novelty="High"):
    return {
        "uniprot_id": uid,
        "name": uid,
        "category": category,
        "corrected_score": corrected,
        "confirmed": confirmed,
        "sourcing_status": "confirmed" if confirmed else "unsourced",
        "sourcing_note": "test",
        "novelty_label": novelty,
        "combined_hits": 0,
    }


def test_recommendation_prefers_under_explored_hidden(ds):
    scored = ds.ScoredCandidates(
        "D1",
        "Test disease",
        known=[_candidate("K1", "Known/Direct", 50.0)],
        hidden=[_candidate("H1", "Hidden/Novel", 20.0, novelty="High")],
    )
    rec = ds.build_recommendation(scored)
    assert rec["action"] == "literature_mining"
    assert rec["candidates"] == ["H1"]
    assert "H1" in rec["text"]


def test_recommendation_falls_back_to_known_when_hidden_unconfirmed(ds):
    scored = ds.ScoredCandidates(
        "D1",
        "Test disease",
        known=[_candidate("K1", "Known/Direct", 50.0)],
        hidden=[_candidate("H1", "Hidden/Novel", 20.0, confirmed=False)],
    )
    rec = ds.build_recommendation(scored)
    assert rec["action"] == "drug_curation"
    assert "K1" in rec["text"]
    assert "unconfirmed" in rec["text"]


def test_recommendation_offers_both_when_neither_is_under_explored(ds):
    scored = ds.ScoredCandidates(
        "D1",
        "Test disease",
        known=[_candidate("K1", "Known/Direct", 50.0)],
        hidden=[_candidate("H1", "Hidden/Novel", 20.0, novelty="Well-explored")],
    )
    rec = ds.build_recommendation(scored)
    assert rec["action"] == "user_choice"
    assert set(rec["candidates"]) == {"H1", "K1"}


def test_recommendation_names_nothing_when_everything_was_dropped(ds):
    scored = ds.ScoredCandidates("D1", "Test disease", dropped=42)
    rec = ds.build_recommendation(scored)
    assert rec["action"] == "none"
    assert "42" in rec["text"]


# ── §11 supervising layer: follow-ups reuse computed state ────────────────────────────


def test_followup_reads_state_without_recomputing(ds, asthma):
    scored = ds.score_candidates(asthma.id, top_known=2, top_hidden=2, reconstruct=True)
    payload = {
        "disease": {"id": asthma.id, "name": asthma.name},
        "candidates": {"known": scored.known, "hidden": scored.hidden},
        "method": scored.diagnostics,
    }
    ds.SESSION_STATE["test-session"] = payload
    try:
        target = (scored.hidden or scored.known)[0]["uniprot_id"]
        answer = ds.answer_followup("test-session", target)
        assert answer["recomputed"] is False
        assert answer["candidate"]["uniprot_id"] == target
        assert answer["explanation"]

        assert not ds.needs_full_chain("test-session", asthma.name)
        assert ds.needs_full_chain("test-session", "cystic fibrosis")
        assert ds.needs_full_chain("no-such-session", asthma.name)

        with pytest.raises(KeyError):
            ds.answer_followup("test-session", "P00000")
    finally:
        ds.SESSION_STATE.pop("test-session", None)


def test_supervisor_routes_messages_without_rerunning(ds, asthma):
    """§11 — the supervising layer decides full chain vs scoped follow-up."""
    import asyncio

    scored = ds.score_candidates(asthma.id, top_known=2, top_hidden=2, reconstruct=True)
    target = (scored.hidden or scored.known)[0]["uniprot_id"]
    ds.SESSION_STATE["route-session"] = {
        "disease": {"id": asthma.id, "name": asthma.name},
        "candidates": {"known": scored.known, "hidden": scored.hidden},
        "method": scored.diagnostics,
    }
    try:
        # A candidate named by accession → follow-up, answered from stored state.
        answer = asyncio.run(ds.handle_message(f"why is {target} hidden?", session_id="route-session"))
        assert answer["mode"] == "followup"
        assert answer["candidate"]["uniprot_id"] == target
        assert answer["recomputed"] is False

        # An accession that was never scored → say so, do not answer about another one.
        missing = asyncio.run(ds.handle_message("what about P00000?", session_id="route-session"))
        assert missing["mode"] == "not_in_results"
        assert missing["requested"] == "P00000"

        # No new disease named and no candidate → recap, still no rerun.
        recap = asyncio.run(ds.handle_message("summarise that again", session_id="route-session"))
        assert recap["mode"] == "recap"
        assert recap["recomputed"] is False
    finally:
        ds.SESSION_STATE.pop("route-session", None)


def test_supervisor_starts_a_fresh_chain_for_an_unknown_session(ds):
    """A session with no state must be routed to the full chain, not a recap."""
    ds.SESSION_STATE.pop("brand-new", None)
    assert ds.needs_full_chain("brand-new", "asthma")
