"""
Offline calibration tests for the LitMineX functional spec (§3-§7).

Nothing here touches the network or an LLM: Tool 1's query construction, Tool 2's
four scoring signals and Tool 3's evidence selection and citation extraction are all
deterministic and are asserted directly.
"""
import pytest

from DRP_Main.app.modules.literature import relation_config as rc
from DRP_Main.app.modules.literature.ner_service import NERService
from DRP_Main.app.modules.literature.pubmed_service import PubMedService
from DRP_Main.app.modules.literature.relevance_service import RelevanceService
from DRP_Main.app.modules.literature.summarization_service import (
    SummarizationService,
    _cited_pmids,
)

TARGET_TERMS = ["AMPK", "AMP-Activated Protein Kinases"]
DISEASE_TERMS = ["Type 2 Diabetes", "Diabetes Mellitus, Type 2", "NIDDM"]
QUERY = "What is the involvement of AMPK in type 2 diabetes?"


class _NoEmbeddings:
    """Deterministic stand-in so scores don't depend on model weights."""

    backend = "stub"

    @staticmethod
    def similarities(query, sentences):
        return [0.5] * len(sentences)


@pytest.fixture
def scorer():
    return RelevanceService(embedding_service=_NoEmbeddings())


# ── Tool 1 (§3 step 3): query construction ───────────────────────────────────
def test_query_ands_the_two_concept_groups():
    query = PubMedService().build_litminex_query(TARGET_TERMS, DISEASE_TERMS)
    assert ") AND (" in query, "target and disease groups must be ANDed, not ORed"
    assert '"AMPK"[Title/Abstract]' in query
    assert '"AMPK"[MeSH Terms]' in query
    assert '"NIDDM"[Title/Abstract]' in query
    # Neither concept may be satisfiable alone.
    target_half, disease_half = query.split(") AND (")
    assert "AMPK" in target_half and "AMPK" not in disease_half
    assert "NIDDM" in disease_half


def test_publication_type_filter_is_optional():
    plain = PubMedService().build_litminex_query(TARGET_TERMS, DISEASE_TERMS)
    filtered = PubMedService().build_litminex_query(
        TARGET_TERMS, DISEASE_TERMS, exclude_reviews=True
    )
    assert "review[Publication Type]" not in plain
    assert filtered.endswith("NOT (review[Publication Type])")


def test_group_is_capped_so_the_uri_cannot_overrun():
    many = [f"Synonym Number {i}" for i in range(60)]
    query = PubMedService().build_litminex_query(many, DISEASE_TERMS, max_terms=12)
    assert query.count("[Title/Abstract]") <= 12 + len(DISEASE_TERMS)


# ── Tool 2 (§4): the four signals ────────────────────────────────────────────
def test_positional_weights_match_the_lookup_table():
    assert rc.section_weight("title") == 1.0
    assert rc.section_weight("abstract") == 1.0
    assert rc.section_weight("Results and Discussion") == 0.6
    assert rc.section_weight("references") == 0.3
    assert rc.section_weight("Introduction") == 0.3


def test_relation_phrase_detection_prefers_the_longest_match():
    assert rc.match_relation_phrase("AMPK plays a key role in diabetes") == "plays a key role in"
    assert rc.match_relation_phrase("AMPK activates glucose uptake") == "activates"
    assert rc.match_relation_phrase("AMPK and diabetes were measured") is None


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("AMPK activates uptake in diabetes [12].", True),
        ("AMPK activates uptake in diabetes (Smith et al. 2019).", True),
        ("We found that AMPK activates uptake in diabetes.", False),
    ],
)
def test_evidence_type_tagging(sentence, expected):
    phrase = rc.match_relation_phrase(sentence)
    assert rc.is_cited(sentence, rc.find_relation_span(sentence, phrase)) is expected


def test_score_weights_sum_to_one():
    assert sum(rc.SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    assert rc.SCORE_WEIGHTS == {
        "relation": 0.40,
        "position": 0.25,
        "similarity": 0.20,
        "primary": 0.15,
    }


def test_primary_abstract_evidence_outranks_cited_reference_evidence(scorer):
    def article(pmid, section, text):
        return {
            "pmid": pmid,
            "title": "t",
            "abstract": text,
            "sections": {section: text},
        }

    strong = article(
        "1", "abstract", "AMPK activation increases glucose uptake in type 2 diabetes."
    )
    weak = article(
        "2", "references", "AMPK was linked to type 2 diabetes [12], per Smith et al."
    )
    ranked = scorer.run(
        {
            "query": QUERY,
            "query_terms": {"target": TARGET_TERMS, "disease": DISEASE_TERMS},
            "articles": [weak, strong],
        }
    )["ranked_articles"]

    assert [a["pmid"] for a in ranked] == ["1", "2"]
    # 40 relation + 25 position + 10 similarity + 15 primary
    assert ranked[0]["relevance_score"] == pytest.approx(90.0)
    # 40 relation + 7.5 position + 10 similarity + 0 primary (cited)
    assert ranked[1]["relevance_score"] == pytest.approx(57.5)


def test_cooccurrence_without_a_relation_scores_below_any_relation_match(scorer):
    article = {
        "pmid": "3",
        "title": "t",
        "abstract": "AMPK and type 2 diabetes were both measured in this cohort.",
        "sections": {"abstract": "AMPK and type 2 diabetes were both measured in this cohort."},
    }
    ranked = scorer.run(
        {
            "query": QUERY,
            "query_terms": {"target": TARGET_TERMS, "disease": DISEASE_TERMS},
            "articles": [article],
        }
    )["ranked_articles"]
    sentence = ranked[0]["relation_sentences"][0]
    assert sentence["relation_found"] is False
    assert 0 < ranked[0]["relevance_score"] < rc.SCORE_WEIGHTS["relation"] * 100


def test_single_concept_run_still_scores(scorer):
    """The /v1 target-only path passes an empty disease group; the co-occurrence gate
    must relax to the concepts actually searched for instead of zeroing everything."""
    article = {
        "pmid": "5",
        "title": "t",
        "abstract": "AMPK activation increases glucose uptake in skeletal muscle.",
        "sections": {"abstract": "AMPK activation increases glucose uptake in skeletal muscle."},
    }
    ranked = scorer.run(
        {
            "query": "What is known about AMPK?",
            "query_terms": {"target": TARGET_TERMS, "disease": []},
            "articles": [article],
        }
    )["ranked_articles"]
    assert ranked[0]["relevance_score"] > 0
    assert ranked[0]["relation_sentences"][0]["relation_found"] is True


def test_sentence_needs_both_entities_to_be_tagged(scorer):
    article = {
        "pmid": "4",
        "title": "t",
        "abstract": "AMPK activates acetyl-CoA carboxylase in muscle.",
        "sections": {"abstract": "AMPK activates acetyl-CoA carboxylase in muscle."},
    }
    ranked = scorer.run(
        {
            "query": QUERY,
            "query_terms": {"target": TARGET_TERMS, "disease": DISEASE_TERMS},
            "articles": [article],
        }
    )["ranked_articles"]
    assert ranked[0]["relation_sentences"] == []
    assert ranked[0]["relevance_score"] == 0.0


# ── Tool 3 (§5): evidence selection and citation ─────────────────────────────
def _scored(pmid, score, relation_found=True):
    return {
        "pmid": pmid,
        "title": f"Article {pmid}",
        "relevance_score": score,
        "pdf_link": "",
        "relation_sentences": [
            {
                "text": "AMPK activation increases glucose uptake.",
                "section": "abstract",
                "relation_found": relation_found,
                "relation_phrase": "increases",
                "evidence_type": "primary",
                "position_weight": 1.0,
                "similarity_score": 0.8,
            }
        ],
    }


def test_evidence_selection_drops_articles_without_relation_sentences():
    selected = SummarizationService().select_evidence(
        [_scored("1", 90), _scored("2", 85, relation_found=False)], top_n=10, min_score=50
    )
    assert [a["pmid"] for a in selected] == ["1"]


def test_evidence_selection_falls_back_when_nothing_clears_the_threshold():
    selected = SummarizationService().select_evidence(
        [_scored("1", 20), _scored("2", 10)], top_n=10, min_score=50
    )
    assert [a["pmid"] for a in selected] == ["1", "2"], "must not return an empty answer set"


def test_context_block_carries_only_relation_sentences_not_abstracts():
    article = _scored("18477788", 91)
    article["abstract"] = "AN ABSTRACT THAT MUST NOT LEAK INTO THE PROMPT"
    context = SummarizationService()._build_context([article])
    assert "PMID:18477788" in context
    assert "AMPK activation increases glucose uptake." in context
    assert "MUST NOT LEAK" not in context


def test_cited_pmids_are_restricted_to_the_supplied_evidence():
    answer = "Claim one [PMID:111]. Claim two [PMID:999]. Claim three [PMID:111]."
    assert _cited_pmids(answer, [_scored("111", 90), _scored("222", 80)]) == ["111"]


# ── §2: entity extraction and the clarify-rather-than-guess rule ─────────────
def test_query_entity_extraction():
    extracted = NERService().extract_query_entities(QUERY)
    assert extracted["target"] == "AMPK"
    assert "diabetes" in (extracted["disease"] or "")


def test_missing_both_concepts_raises_rather_than_guessing():
    from DRP_Main.app.modules.literature.retrieval_service import (
        MissingEntityError,
        RetrievalService,
    )

    with pytest.raises(MissingEntityError) as exc:
        RetrievalService().resolve_entities({"query": "show me some recent papers"})
    assert set(exc.value.missing) == {"target", "disease"}


def test_a_single_missing_concept_also_prompts():
    """§2: prompt for *whatever* is missing — one concept alone is not searchable,
    because Tool 1's whole point is ANDing the two groups."""
    from DRP_Main.app.modules.literature.retrieval_service import (
        MissingEntityError,
        RetrievalService,
    )

    with pytest.raises(MissingEntityError) as exc:
        RetrievalService().resolve_entities({"query": "tell me about AMPK"})
    assert exc.value.missing == ["disease"]
    assert exc.value.extracted["target"] == "AMPK", "what NER did find must be carried back"


def test_v1_opt_out_allows_a_single_concept_run():
    from DRP_Main.app.modules.literature.retrieval_service import RetrievalService

    resolved = RetrievalService().resolve_entities(
        {"query": "tell me about AMPK"}, require_both=False
    )
    assert resolved["target"] == "AMPK"
    assert resolved["disease"] is None


def test_carryover_input_is_accepted_without_a_query():
    from DRP_Main.app.modules.literature.retrieval_service import RetrievalService

    resolved = RetrievalService().resolve_entities(
        {"target": "AMPK", "disease": "Type 2 Diabetes", "source": "txkg_carryover"}
    )
    assert resolved["target"] == "AMPK" and resolved["disease"] == "Type 2 Diabetes"
    # A query string is synthesised so semantic similarity has something to anchor on.
    assert "AMPK" in resolved["query"] and "Type 2 Diabetes" in resolved["query"]
