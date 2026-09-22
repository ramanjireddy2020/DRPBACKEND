"""
NovSearch spec pins — offline (no network, no Databricks, no LLM).

Covers the parts of the spec that are pure logic and would otherwise only be
caught in a live run: the §2 input branch, the §3 ranking arithmetic, the §4
section weights and claim tagging, and the §5 report split and QA mode names.

Run with `--noconftest` locally: tests/conftest.py requires Databricks auth.
"""
from __future__ import annotations

import pytest

from DRP_Main.app.modules.novelty import indexing_service as ix
from DRP_Main.app.modules.novelty import novsearch_service as ns
from DRP_Main.app.modules.novelty import google_patents_service as pv
from DRP_Main.app.modules.novelty import retrieval_service as rs
from DRP_Main.app.modules.novelty import synthesis_service as sy


# ── §2 input handling ────────────────────────────────────────────────────────
def test_carryover_builds_query_and_keeps_trace_ids():
    normalized = ns.normalize_input(
        {
            "source": "screensuite_carryover",
            "candidate_id": "cand_0091",
            "docking_id": "dock_0142",
            "target": "JAK2",
            "drug": "Fedratinib",
            "disease": "Thrombocytosis",
        }
    )
    assert normalized.query == "JAK2 Fedratinib Thrombocytosis"
    assert normalized.source == ns.SOURCE_CARRYOVER
    assert normalized.as_dict()["candidate_id"] == "cand_0091"
    assert normalized.as_dict()["docking_id"] == "dock_0142"


def test_user_direct_uses_query_as_typed_and_omits_trace_ids():
    normalized = ns.normalize_input(
        {
            "source": "user_direct",
            "query": "Is there existing patent coverage for JAK inhibitors in platelet disorders?",
        }
    )
    assert normalized.query.startswith("Is there existing patent coverage")
    payload = normalized.as_dict()
    # Absent, not null-filled — downstream branches on presence.
    assert "candidate_id" not in payload
    assert "docking_id" not in payload


def test_empty_query_asks_for_clarification_rather_than_guessing():
    with pytest.raises(ns.AmbiguousQueryError):
        ns.normalize_input({"source": "user_direct", "query": "   "})


# ── §3 Tool 1 ────────────────────────────────────────────────────────────────
def test_title_detection():
    assert rs.looks_like_title('"JAK2 inhibitor compositions"')
    assert rs.looks_like_title("JAK2 inhibitor compositions for myeloproliferative disorders")
    assert not rs.looks_like_title("Is there existing patent coverage for JAK inhibitors?")


def test_reciprocal_rank_fusion_sums_inverse_ranks():
    fused = rs.reciprocal_rank_fusion(
        [
            [{"patent_id": "A", "rank": 1}, {"patent_id": "B", "rank": 2}],
            [{"patent_id": "B", "rank": 1}, {"patent_id": "A", "rank": 3}],
        ],
        k=60,
    )
    # A: 1/61 + 1/63, B: 1/62 + 1/61 — B wins.
    assert fused == ["B", "A"]


def test_patent_id_round_trip():
    assert pv.normalize_patent_id("US10,123,456 B2") == "US10123456B2"
    assert pv.display_patent_id("US10123456B2") == "US10123456B2"


def test_dependent_claim_ranges_are_expanded():
    assert pv._depends_on("The composition of claims 1 to 3, wherein") == [1, 3]


# ── Google Patents HTML parsing (scraped page, not USPTO ODP's XML) ──────────
PATENT_HTML = """<html><body>
<span itemprop="title">JAK2 inhibitor compositions</span>
<section itemprop="abstract">Abstract Compositions comprising a JAK2 inhibitor.</section>
BACKGROUND OF THE INVENTION
Prior approaches were inadequate.
BRIEF SUMMARY
The invention provides compositions.
DETAILED DESCRIPTION
In one embodiment the compound is FEDRATINIB.
<section itemprop="claims">
  <div class="claim" num="1">1. A composition comprising a JAK2 inhibitor.</div>
  <div class="claim" num="2">2. The composition of claim 1, wherein the inhibitor is FEDRATINIB.</div>
</section>
</body></html>
"""


def test_patent_html_yields_structured_claims_with_dependencies():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(PATENT_HTML, "html.parser")
    claims = pv._parse_claims(soup, soup.get_text("\n", strip=True))
    assert [c["number"] for c in claims] == [1, 2]

    independent, dependent = claims
    assert independent["dependent"] is False
    assert independent["depends_on"] == []
    # Dependency comes from a regex over the claim's own text.
    assert dependent["dependent"] is True
    assert dependent["depends_on"] == [1]
    # The leading claim number is stripped from the stored text.
    assert independent["text"].startswith("A composition")


def test_patent_html_splits_description_sections_for_the_weight_table():
    parsed = pv._parse_patent_html(PATENT_HTML, "US10123456B2")
    assert parsed["title"] == "JAK2 inhibitor compositions"
    assert "JAK2 inhibitor" in parsed["abstract"]
    assert parsed["background"].startswith("Prior approaches")
    assert parsed["summary"].startswith("The invention provides")
    assert "FEDRATINIB" in parsed["description"]
    # Background must not leak into description — it describes prior art, and §4
    # weights it lowest for exactly that reason.
    assert "Prior approaches" not in parsed["description"]


def test_empty_html_degrades_instead_of_raising():
    assert pv._parse_patent_html("<html></html>", "US1").get("claims", []) == []


# ── §4 Tool 2 ────────────────────────────────────────────────────────────────
def test_section_weights_match_the_spec():
    assert ix.SECTION_WEIGHTS == {
        "independent_claims": 1.00,
        "abstract": 0.95,
        "summary": 0.90,
        "dependent_claims": 0.85,
        "metadata": 0.85,
        "description": 0.80,
        "background": 0.70,
    }


def test_claim_chunks_are_tagged_by_claim_number():
    chunks = ix.chunker.chunks_for(
        {
            "patent_id": "10123456",
            "title": "JAK2 inhibitor compositions",
            "abstract": "An abstract.",
            "assignee": "Acme",
            "independent_claims": [{"number": 1, "text": "A composition comprising..."}],
            "dependent_claims": [
                {"number": 2, "text": "The composition of claim 1", "depends_on": [1]}
            ],
            "summary": "",
            "description": "",
            "background": "",
        }
    )
    by_section = {c["section_type"]: c for c in chunks}
    assert by_section["independent_claims"]["claim_number"] == 1
    assert by_section["independent_claims"]["is_claim"] is True
    assert by_section["dependent_claims"]["depends_on"] == [1]
    assert by_section["independent_claims"]["section_weight"] == 1.00
    assert "metadata" in by_section


def test_empty_sections_produce_no_chunks():
    chunks = ix.chunker.chunks_for(
        {"patent_id": "1", "abstract": "", "summary": "", "description": "", "background": ""}
    )
    assert chunks == []


# ── §5 Tool 3 ────────────────────────────────────────────────────────────────
def test_report_splits_on_recommended_next_steps():
    answer, recommendations = sy._split_report(
        "AGENT ANALYSIS:\nx\nKEY FINDINGS:\n- y\nRECOMMENDED NEXT STEPS\n1. z\n"
        "GAPS IN CURRENT ANALYSIS:\nw"
    )
    assert answer.startswith("AGENT ANALYSIS")
    assert "KEY FINDINGS" in answer
    assert recommendations.startswith("RECOMMENDED NEXT STEPS")
    assert "GAPS IN CURRENT ANALYSIS" in recommendations


def test_report_without_the_marker_keeps_the_whole_body():
    answer, recommendations = sy._split_report("AGENT ANALYSIS:\nonly this")
    assert answer == "AGENT ANALYSIS:\nonly this"
    assert recommendations == ""


@pytest.mark.asyncio
async def test_qa_modes(monkeypatch):
    monkeypatch.setattr(sy.vector_store, "indexed_patent_ids", lambda: ["10123456", "9876543"])

    async def fake_search(question, patent_id, top_k=None):
        return [
            {
                "section_type": "abstract",
                "chunk_text": "content",
                "is_claim": False,
                "section_weight": 0.95,
            }
        ]

    async def fake_generate(prompt):
        return "answer", "saullm-legal-databricks-serving"

    monkeypatch.setattr(sy.vector_store, "search_by_patent", fake_search)
    monkeypatch.setattr(sy.llm_client, "generate", fake_generate)

    single = await sy.answer_question("What is claimed?", ["US10123456B2"])
    assert single["mode"] == "single_patent"
    assert single["patent_ids_used"] == ["US10123456"]

    subset = await sy.answer_question("What is claimed?", ["10123456", "9876543"])
    assert subset["mode"] == "multiple_patent"

    everything = await sy.answer_question("What is claimed?")
    assert everything["mode"] == "multi_patent"
    assert len(everything["patent_ids_used"]) == 2


@pytest.mark.asyncio
async def test_qa_rejects_patents_that_are_not_indexed(monkeypatch):
    monkeypatch.setattr(sy.vector_store, "indexed_patent_ids", lambda: ["10123456"])
    result = await sy.answer_question("Who owns it?", ["US9999999B1"])
    assert result["patent_ids_used"] == []
    assert "not indexed" in result["answer"]
