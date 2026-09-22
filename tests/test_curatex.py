"""
Offline calibration tests for the CurateX functional spec.

No network, no LLM: these pin the parts of the spec that are pure logic — the
§4.1 weight table, the §4.2 normalization and composite rules, the §5 step 4
merge precedence, the §5 step 5 label parsing, and the §6 step 2 exclusion
filter. Anything that would hit ChEMBL / Open Targets / DailyMed is fed fixtures.
"""
import pytest

from DRP_Main.app.modules.drug_curation import criteria_config as cc
from DRP_Main.app.modules.drug_curation import dailymed_service, profile_service
from DRP_Main.app.modules.drug_curation.candidate_service import (
    Candidate,
    apply_exclusion_filter,
)
from DRP_Main.app.modules.drug_curation.evidence_service import (
    EVIDENCE_BACKED,
    SCORE_ONLY,
    CandidateEvidence,
    build_recommendation,
    build_query,
)
from DRP_Main.app.modules.drug_curation.scoring_service import (
    build_baselines,
    score_candidate,
)


# ── §4.1 the locked criteria table ────────────────────────────────────────────
def test_nineteen_weighted_criteria_summing_to_100():
    # 20 scored rows, because criterion 11 is split into 11a (withdrawn status)
    # and 11b (withdrawal reason) — the spec still counts it as one criterion.
    assert len(cc.CRITERIA) == 20
    assert len({c.number.rstrip("ab") for c in cc.CRITERIA}) == 19
    assert sum(c.default_weight for c in cc.CRITERIA) == 100


def test_category_budgets_match_the_spec():
    assert cc.category_totals() == {
        "Safety": 30.0,
        "Regulatory status": 25.0,
        "Physicochemical / drug-likeness": 30.0,
        "Dosage & PK/ADME": 15.0,
    }


def test_exclusion_and_metadata_rows_are_unweighted():
    assert cc.EXCLUSION_CRITERION.default_weight == 0
    assert cc.METADATA_CRITERION.default_weight == 0
    assert cc.EXCLUSION_CRITERION.key not in cc.DEFAULT_WEIGHTS
    assert cc.METADATA_CRITERION.key not in cc.DEFAULT_WEIGHTS


def test_withdrawal_reason_lookup_is_the_spec_example():
    reason = cc.CRITERIA_BY_KEY["withdrawal_reason"]
    assert reason.value_map["safety"] == 0.0
    assert reason.value_map["unknown"] == 0.5
    assert reason.value_map["commercial"] == 1.0


def test_weight_validation_rejects_unknown_and_negative():
    problems = cc.validate_weight_table({"qed": -1, "not_a_criterion": 5})
    assert any("negative" in p for p in problems)
    assert any("Unknown criterion" in p for p in problems)


# ── §4.2 normalization and composite ──────────────────────────────────────────
@pytest.fixture
def baselines():
    """A three-ligand reference set: QED 0.2-0.8, Ro5 violations 0-2."""
    return build_baselines(
        [
            {"qed": 0.2, "ro5_violations": 0, "max_clinical_phase": 4},
            {"qed": 0.5, "ro5_violations": 1, "max_clinical_phase": 3},
            {"qed": 0.8, "ro5_violations": 2, "max_clinical_phase": 2},
        ]
    )


def test_numeric_normalization_is_min_max_and_clipped(baselines):
    weights = {"qed": 10}
    top = score_candidate({"qed": 0.8}, baselines, weights)
    bottom = score_candidate({"qed": 0.2}, baselines, weights)
    above = score_candidate({"qed": 5.0}, baselines, weights)  # outside the range
    assert top.composite == pytest.approx(100.0)
    assert bottom.composite == pytest.approx(0.0)
    assert above.composite == pytest.approx(100.0)  # clipped, not extrapolated


def test_inverse_criteria_are_flipped(baselines):
    weights = {"ro5_violations": 10}
    clean = score_candidate({"ro5_violations": 0}, baselines, weights)
    dirty = score_candidate({"ro5_violations": 2}, baselines, weights)
    assert clean.composite == pytest.approx(100.0)
    assert dirty.composite == pytest.approx(0.0)


def test_missing_data_leaves_the_weight_out_of_both_sums(baselines):
    """A failed DailyMed extraction must not score as zero."""
    weights = {"qed": 10, "half_life": 10}
    scored = score_candidate({"qed": 0.8}, baselines, weights)
    assert scored.composite == pytest.approx(100.0)   # not 50
    assert scored.weight_used == 10
    assert scored.weight_available == 20
    assert "half_life" in scored.missing
    assert scored.data_completeness == 0.5


def test_zero_weight_excludes_a_criterion(baselines):
    scored = score_candidate({"qed": 0.2, "ro5_violations": 0}, baselines,
                             {"qed": 0, "ro5_violations": 10})
    assert scored.composite == pytest.approx(100.0)
    assert scored.weight_available == 10


def test_categorical_scoring_uses_the_lookup_table(baselines):
    withdrawn = score_candidate({"withdrawn_status": "withdrawn"}, baselines,
                                {"withdrawn_status": 10})
    clean = score_candidate({"withdrawn_status": "not_withdrawn"}, baselines,
                            {"withdrawn_status": 10})
    assert withdrawn.composite == pytest.approx(0.0)
    assert clean.composite == pytest.approx(100.0)


def test_degenerate_baseline_range_scores_exact_matches_only(baselines):
    flat = build_baselines([{"logp": 2.0}, {"logp": 2.0}])
    match = score_candidate({"logp": 2.0}, flat, {"logp": 10})
    miss = score_candidate({"logp": 4.0}, flat, {"logp": 10})
    assert match.composite == pytest.approx(100.0)
    assert miss.composite == pytest.approx(0.0)


def test_coverage_is_computed_over_the_whole_ligand_set():
    baselines = build_baselines([{"half_life": 4.0}, {}, {}, {"half_life": 6.0}])
    assert baselines["half_life"].count == 2
    assert baselines["half_life"].total == 4
    assert baselines["half_life"].coverage == 0.5


# ── §2.2 ChEMBL field parsing ─────────────────────────────────────────────────
def test_chembl_orphan_flag_is_tri_state():
    """ChEMBL uses -1 for 'not assessed'; it must not read as a designation."""
    from DRP_Main.app.modules.drug_curation.ligand_sources import parse_chembl_molecule

    def orphan(value):
        parsed = parse_chembl_molecule(
            {"molecule_chembl_id": "CHEMBL1", "pref_name": "X", "orphan": value}
        )
        return parsed["fields"].get("orphan_designation")

    assert orphan(1) == "yes"
    assert orphan(0) == "no"
    assert orphan(-1) is None      # absent, not "yes"


def test_chembl_withdrawn_flag_without_a_reason_falls_back_to_unknown():
    from DRP_Main.app.modules.drug_curation.ligand_sources import parse_chembl_molecule

    parsed = parse_chembl_molecule(
        {"molecule_chembl_id": "CHEMBL468", "pref_name": "THALIDOMIDE",
         "withdrawn_flag": True, "max_phase": "4.0", "black_box_warning": 1}
    )
    assert parsed["fields"]["withdrawn_status"] == "withdrawn"
    assert parsed["fields"]["withdrawal_reason"] == "unknown"  # backfilled from drug_warning
    assert parsed["fields"]["max_clinical_phase"] == 4.0
    assert parsed["fields"]["black_box_warning"] == "yes"


def test_teratogenicity_is_bucketed_as_a_safety_withdrawal():
    from DRP_Main.app.modules.drug_curation.ligand_sources import _normalize_withdrawal_reason

    assert _normalize_withdrawal_reason("teratogenicity") == "safety"
    assert _normalize_withdrawal_reason("commercial reasons") == "commercial"
    assert _normalize_withdrawal_reason(None) == "unknown"


def test_open_targets_clinical_stage_maps_onto_the_phase_scale():
    from DRP_Main.app.modules.drug_curation.ligand_sources import clinical_stage_to_phase

    assert clinical_stage_to_phase("APPROVAL") == 4.0
    assert clinical_stage_to_phase("PHASE_1_2") == 1.5
    assert clinical_stage_to_phase("NOT_A_STAGE") is None


# ── §5 step 4 merge precedence ────────────────────────────────────────────────
def test_open_targets_wins_and_chembl_is_kept_as_secondary():
    ligands = profile_service.merge_records(
        [
            {
                "source": "chembl", "record_id": "CHEMBL25", "chembl_id": "CHEMBL25",
                "name": "ASPIRIN", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "fields": {"max_clinical_phase": 3.0, "qed": 0.55},
            },
            {
                "source": "open_targets", "record_id": "CHEMBL25", "chembl_id": "CHEMBL25",
                "name": "ASPIRIN", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "fields": {"max_clinical_phase": 4.0},
            },
        ],
        resolve_missing_structures=False,
    )
    assert len(ligands) == 1
    ligand = ligands[0]
    assert ligand.fields["max_clinical_phase"] == 4.0
    assert ligand.secondary_values["max_clinical_phase"] == 3.0
    assert ligand.fields["qed"] == 0.55           # ChEMBL-only field survives
    assert {p["source"] for p in ligand.provenance} == {"chembl", "open_targets"}


def test_discovery_only_sources_contribute_structure_not_criteria():
    ligands = profile_service.merge_records(
        [
            {
                "source": "chembl", "record_id": "CHEMBL25", "chembl_id": "CHEMBL25",
                "name": "ASPIRIN", "inchikey": "KEY1", "fields": {"qed": 0.55},
            },
            {
                "source": "bindingdb", "record_id": "42", "name": "aspirin",
                "inchikey": "KEY1", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "bioactivity": {"type": "Ki", "value_nm": 120.0},
                "fields": {"qed": 0.99},          # must be ignored
            },
        ],
        resolve_missing_structures=False,
    )
    ligand = ligands[0]
    assert ligand.fields["qed"] == 0.55
    assert ligand.bioactivity == [{"type": "Ki", "value_nm": 120.0}]
    assert ligand.smiles == "CC(=O)Oc1ccccc1C(=O)O"


def test_duplicate_chembl_records_collapse_to_one_ligand():
    ligands = profile_service.merge_records(
        [
            {"source": "chembl", "record_id": "A", "chembl_id": "CHEMBL25",
             "name": "ASPIRIN", "inchikey": "KEY1", "fields": {"qed": 0.55}},
            {"source": "chembl", "record_id": "B", "chembl_id": "CHEMBL25",
             "name": "ASPIRIN SODIUM", "inchikey": "KEY1", "fields": {"qed": 0.4}},
        ],
        resolve_missing_structures=False,
    )
    assert len(ligands) == 1
    assert ligands[0].fields["qed"] == 0.55


# ── §5 step 5 DailyMed parsing ────────────────────────────────────────────────
def test_dosage_section_parsing():
    fields, evidence = dailymed_service.parse_dosage_section(
        "The recommended dose is one 50 mg tablet taken orally twice daily with food."
    )
    assert fields["dosage_form"] == "oral_solid"
    assert fields["dosing_frequency"] == "twice_daily"
    assert evidence["dosage_form"]


def test_clinical_pharmacology_parsing():
    fields, _ = dailymed_service.parse_clinical_pharmacology(
        "Absolute bioavailability is approximately 74%. The terminal half-life "
        "ranges from 3 to 6 hours. The drug is metabolized primarily by CYP3A4 "
        "and to a lesser extent CYP2C9."
    )
    assert fields["bioavailability"] == 74.0
    assert fields["half_life"] == pytest.approx(4.5)
    assert fields["metabolism_cyp"] == "cyp3a4"


def test_half_life_unit_conversion():
    fields, _ = dailymed_service.parse_clinical_pharmacology(
        "The elimination half-life is 30 minutes."
    )
    assert fields["half_life"] == pytest.approx(0.5)


def test_unmetabolized_drug_scores_best_on_cyp():
    fields, _ = dailymed_service.parse_clinical_pharmacology(
        "The compound is not significantly metabolized and is excreted unchanged."
    )
    assert fields["metabolism_cyp"] == "none"
    assert cc.CRITERIA_BY_KEY["metabolism_cyp"].value_map["none"] == 1.0


# ── §6 step 2 exclusion filter ────────────────────────────────────────────────
def test_exclusion_filter_removes_descendant_matches_before_scoring():
    candidates = [
        Candidate(chembl_id="CHEMBL1", name="Already Indicated"),
        Candidate(chembl_id="CHEMBL2", name="Genuine Candidate"),
    ]
    exclusion = {
        "active": True,
        "drugs": {
            "CHEMBL1": {
                "drugId": "CHEMBL1",
                "matchedEfoIds": ["EFO_0001360"],          # a descendant term
                "matchedDiseaseNames": ["type II diabetes mellitus"],
            }
        },
    }
    kept, removed = apply_exclusion_filter(candidates, exclusion)
    assert [c.chembl_id for c in kept] == ["CHEMBL2"]
    assert removed[0]["matchedDiseases"] == ["type II diabetes mellitus"]
    assert removed[0]["matchedEfoIds"] == ["EFO_0001360"]


def test_inactive_exclusion_keeps_everything():
    candidates = [Candidate(chembl_id="CHEMBL1", name="X")]
    kept, removed = apply_exclusion_filter(candidates, {"active": False})
    assert len(kept) == 1 and removed == []


# ── §7 evidence and recommendation ────────────────────────────────────────────
def test_evidence_query_scopes_drug_to_target_and_disease():
    query = build_query("Ruxolitinib", "JAK2", "Myelofibrosis")
    assert '"Ruxolitinib"' in query
    assert '"JAK2" OR "Myelofibrosis"' in query
    assert query.count(" AND ") == 1


def test_recommendation_separates_score_only_candidates():
    backed = Candidate(chembl_id="C1", name="Backed")
    only = Candidate(chembl_id="C2", name="Unsupported")
    baselines = build_baselines([{"qed": 0.2}, {"qed": 0.8}])
    backed.score = score_candidate({"qed": 0.8}, baselines, {"qed": 10})
    only.score = score_candidate({"qed": 0.7}, baselines, {"qed": 10})

    evidence = {
        "C1": CandidateEvidence(
            chembl_id="C1", name="Backed",
            validated_articles=[{"pmid": "1", "title": "t"}], strength=EVIDENCE_BACKED,
        ),
        "C2": CandidateEvidence(chembl_id="C2", name="Unsupported", strength=SCORE_ONLY),
    }
    text = build_recommendation(
        [backed, only], evidence, "JAK2", "Myelofibrosis",
        {"active": True, "count": 7},
    )
    assert "Backed" in text
    assert "Score-only" in text and "Unsupported" in text
    assert "7 drug(s) already indicated" in text


def test_recommendation_handles_an_empty_candidate_list():
    assert "No candidate survived" in build_recommendation([], {}, "JAK2", None)
