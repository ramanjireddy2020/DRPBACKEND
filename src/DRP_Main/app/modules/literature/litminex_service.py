"""
LitMineX orchestrator — calls Tools 1 → 2 → 3 in sequence and assembles the final
object returned to the supervisor (spec §7).

The three tools stay independently callable so the agent can re-run Tool 3 alone for
follow-up QA over an already-retrieved set (spec §6) without paying for retrieval and
scoring again.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.literature.relevance_service import RelevanceService
from DRP_Main.app.modules.literature.retrieval_service import (
    MissingEntityError,
    RetrievalService,
)
from DRP_Main.app.modules.literature.summarization_service import SummarizationService

logger = get_logger(__name__)


class LitMineXService:
    def __init__(self) -> None:
        self.retrieval = RetrievalService()
        self.relevance = RelevanceService()
        self.summarization = SummarizationService()
        # Ranked articles kept per session so §6 follow-ups can be scoped by PMID
        # without re-retrieving. Keyed by session id; the supervisor owns the id.
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ── full chain ───────────────────────────────────────────────────────────
    @observe(name="litminex_pipeline")
    def run(
        self,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        max_results: Optional[int] = None,
        top_n: Optional[int] = None,
        exclude_reviews: Optional[bool] = None,
        require_both: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full LitMineX chain.

        `payload` is either the TxKG carry-over shape (`target`/`disease`/`source`) or
        the fresh-query shape (`query`/`source`). Raises `MissingEntityError` when
        either concept is missing — the agent turns that into a single clarifying
        prompt rather than guessing. See `RetrievalService.resolve_entities` for what
        `require_both=False` gives up.
        """
        retrieval = self.retrieval.run(
            payload,
            max_results=max_results,
            exclude_reviews=exclude_reviews,
            require_both=require_both,
        )
        scored = self.relevance.run(retrieval)
        summary = self.summarization.run(
            retrieval["query"], scored["ranked_articles"], top_n=top_n
        )

        result = self._assemble(retrieval, scored, summary)
        if session_id:
            self._sessions[session_id] = {
                "query": retrieval["query"],
                "ranked_articles": scored["ranked_articles"],
            }
        langfuse_context.update_current_observation(
            output={
                "articles": len(scored["ranked_articles"]),
                "cited_pmids": len(result["query_answer"]["cited_pmids"]),
            }
        )
        return result

    # ── §7 final object ──────────────────────────────────────────────────────
    @staticmethod
    def _assemble(
        retrieval: Dict[str, Any], scored: Dict[str, Any], summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        ranked = scored["ranked_articles"]
        return {
            "module": "litminex",
            "target": retrieval.get("target"),
            "disease": retrieval.get("disease"),
            "query_answer": summary["query_answer"],
            "article_table": summary["article_table"],
            "session_state_update": {
                "litminex_last_query": retrieval["query"],
                "litminex_articles": [a["pmid"] for a in ranked if a.get("pmid")],
            },
            # Retained alongside the spec's fields for the /v1 layer and for §6
            # follow-ups, which need the tagged sentences rather than the table rows.
            "ranked_articles": ranked,
            "query_terms": retrieval["query_terms"],
            "pubmed_query": retrieval["pubmed_query"],
            "pipeline_backends": {
                "ner": scored.get("ner_backend"),
                "similarity": scored.get("similarity_backend"),
            },
        }

    # ── §6 QA over the retrieved set ─────────────────────────────────────────
    @observe(name="litminex_followup")
    def answer_followup(
        self,
        question: str,
        session_id: str,
        pmids: Optional[List[str]] = None,
        deep: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a follow-up scoped to one article, several, or the whole retrieved set.

        `pmids=None` means the full set; `deep=True` widens the context from relation
        sentences to the articles' full section text.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(
                f"No LitMineX retrieval is cached for session '{session_id}'. Run a query first."
            )
        articles = session["ranked_articles"]
        if pmids:
            wanted = {str(p) for p in pmids}
            articles = [a for a in articles if str(a.get("pmid")) in wanted]
            if not articles:
                raise KeyError(f"None of {sorted(wanted)} are in this session's retrieved set")
        else:
            articles = articles[: settings.LITMINEX_SUMMARY_TOP_N]

        answer = self.summarization.answer_followup(question, articles, deep=deep)
        return {
            "module": "litminex",
            "question": question,
            "scope_pmids": [a.get("pmid") for a in articles],
            "query_answer": answer,
        }

    # ── session helpers ──────────────────────────────────────────────────────
    def cache_session(self, session_id: str, query: str, ranked_articles: List[Dict]) -> None:
        self._sessions[session_id] = {"query": query, "ranked_articles": ranked_articles}

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)


__all__ = ["LitMineXService", "MissingEntityError"]
