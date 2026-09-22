"""
Literature Mining HTTP surface — the keyword search that replaced the old
`pubmed_literature_search_script`.

This file used to import `app.services.pubmed_literature_search_script`, a module
from the pre-`modules/` monolith that was deleted when the app was consolidated.
Every assertion in it was commented out, so the only live statement was a call
into that missing module — which meant an `ImportError` at collection time and
an aborted test run for the whole directory.

There is no `clear_data_literature_search` equivalent in the current
architecture: the download-articles-to-disk workflow went away with the
monolith. What replaced that surface is `/search-keywords/` (legacy keyword
search) plus `/litminex/*` (the spec pipeline). The LitMineX *internals* are
pinned in `tests/test_litminex.py`; this file covers the HTTP contracts, which
nothing else does.

Offline: PubMed is stubbed, no network and no LLM.

Run:
    cd src && python -m pytest ../tests/unit/test_pubmed_literature_search.py --noconftest -q
"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from DRP_Main.app.modules.literature import router as literature_router  # noqa: E402


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(literature_router.router, prefix="/api/v1/literature-mining")
    return TestClient(app)


@pytest.fixture()
def stub_search(monkeypatch):
    """Replace the PubMed+LLM search with a recorder of what it was asked for."""
    calls = []

    def fake_search(article_keywords, search_keywords, max_results):
        calls.append(
            {
                "article_keywords": article_keywords,
                "search_keywords": search_keywords,
                "max_results": max_results,
            }
        )
        # Must satisfy `ArticleResult`, or FastAPI fails response validation —
        # which is itself part of this endpoint's contract.
        return [
            {
                "protein_name": "AcrB",
                "title": "AcrB efflux pump inhibition",
                "score": 87.5,
                "pdf_file_path": "https://pubmed.ncbi.nlm.nih.gov/40000001/",
                "found_keywords": ["AcrB", "efflux"],
                "preview": "An abstract preview.",
            }
        ]

    monkeypatch.setattr(
        literature_router.literature_service, "search_keywords", fake_search
    )
    return calls


# ── /search-keywords/ ─────────────────────────────────────────────────────────

def test_search_returns_results_and_echoes_the_parsed_keywords(client, stub_search):
    response = client.get(
        "/api/v1/literature-mining/search-keywords/",
        params={
            "article_keywords": "AcrB, MexAB-OprM",
            "search_keywords": "efflux, resistance",
            "max_results": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert body["results"][0]["protein_name"] == "AcrB"
    assert body["results"][0]["score"] == 87.5
    # Comma-separated input is split and trimmed, not passed through whole.
    assert body["article_keywords"] == ["AcrB", "MexAB-OprM"]
    assert body["search_keywords"] == ["efflux", "resistance"]
    assert isinstance(body["execution_time"], (int, float))

    assert stub_search[0]["article_keywords"] == ["AcrB", "MexAB-OprM"]
    assert stub_search[0]["max_results"] == 5


def test_context_keywords_are_optional(client, stub_search):
    """Protein names are mandatory; context keywords only add bonus scoring."""
    response = client.get(
        "/api/v1/literature-mining/search-keywords/",
        params={"article_keywords": "AcrB"},
    )
    assert response.status_code == 200
    assert response.json()["search_keywords"] == []
    assert stub_search[0]["search_keywords"] == []


def test_blank_and_missing_protein_names_are_rejected(client, stub_search):
    # Absent entirely.
    assert client.get("/api/v1/literature-mining/search-keywords/").status_code == 422
    # Present but empty — min_length=1 rejects it before the handler runs.
    assert (
        client.get(
            "/api/v1/literature-mining/search-keywords/",
            params={"article_keywords": ""},
        ).status_code
        == 422
    )
    # Only separators, so the parsed list is empty — the handler's own 400.
    response = client.get(
        "/api/v1/literature-mining/search-keywords/", params={"article_keywords": " , , "}
    )
    assert response.status_code == 400
    assert "required" in response.json()["detail"]
    # None of those reached the search.
    assert stub_search == []


def test_max_results_is_bounded(client, stub_search):
    """The cap keeps one request from pulling an unbounded PubMed page."""
    for value in (0, 101):
        response = client.get(
            "/api/v1/literature-mining/search-keywords/",
            params={"article_keywords": "AcrB", "max_results": value},
        )
        assert response.status_code == 422, f"max_results={value} should be rejected"

    ok = client.get(
        "/api/v1/literature-mining/search-keywords/",
        params={"article_keywords": "AcrB", "max_results": 100},
    )
    assert ok.status_code == 200


def test_default_max_results(client, stub_search):
    client.get(
        "/api/v1/literature-mining/search-keywords/", params={"article_keywords": "AcrB"}
    )
    assert stub_search[0]["max_results"] == 20


# ── /litminex/query ───────────────────────────────────────────────────────────

def test_missing_concepts_prompt_rather_than_guessing(client, monkeypatch):
    """
    A query naming neither a target nor a disease must ask, not search — one
    concept alone is not searchable, since the tool exists to AND two groups.
    """
    from DRP_Main.app.modules.literature.litminex_service import MissingEntityError

    def raise_missing(*args, **kwargs):
        raise MissingEntityError(missing=["target", "disease"], extracted={})

    monkeypatch.setattr(literature_router.litminex_service, "run", raise_missing)

    response = client.post(
        "/api/v1/literature-mining/litminex/query", json={"query": "what about resistance"}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["missing"] == ["target", "disease"]
    assert "target and disease" in detail["prompt"]


def test_followup_on_an_unknown_session_is_404(client, monkeypatch):
    def raise_key_error(*args, **kwargs):
        raise KeyError("no such session: abc")

    monkeypatch.setattr(
        literature_router.litminex_service, "answer_followup", raise_key_error
    )

    response = client.post(
        "/api/v1/literature-mining/litminex/followup",
        json={"question": "which paper said that?", "session_id": "abc"},
    )
    assert response.status_code == 404
    assert "no such session" in response.json()["detail"]


# ── /health/ ──────────────────────────────────────────────────────────────────

def test_health_reports_the_active_backends(client):
    """
    scispaCy and SapBERT are optional; health must say which backend is live so
    a quiet downgrade to the rule-based tagger is visible rather than guessed at.
    """
    response = client.get("/api/v1/literature-mining/health/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "Literature Mining"

    litminex = body["config"]["litminex"]
    assert litminex["ner_backend"] in ("scispacy", "rule-based")
    assert litminex["similarity_backend"]
    assert litminex["retrieval_limit"] > 0
