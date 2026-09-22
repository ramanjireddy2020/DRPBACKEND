"""
LitMineX Tool 3 — Summarization (spec §5) and QA over the retrieved set (spec §6).

The answer shown to the user is built only from the highest-relevance, already-
extracted relation sentences — never from raw abstracts — with every claim cited
inline to its source PMID.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

MAX_SENTENCES_PER_ARTICLE = 6
_PMID_CITATION = re.compile(r"\[PMID:\s*(\d+)\s*\]")

_ANSWER_SYSTEM_PROMPT = (
    "You are a biomedical literature analyst. Answer the user's question using ONLY "
    "the supplied evidence sentences. Do not introduce any claim, mechanism, number "
    "or conclusion that is not present in them. Cite every claim inline in the form "
    "[PMID:12345678], using the PMID given with the sentence you relied on. If the "
    "evidence does not answer the question, say so plainly."
)

_PER_ARTICLE_SYSTEM_PROMPT = (
    "Summarise what this single article establishes, in 1-2 sentences, using ONLY the "
    "supplied sentences from that article. No citation markers, no hedging preamble."
)


class SummarizationService:
    def __init__(self) -> None:
        self.model = settings.DATABRICKS_LLM_ENDPOINT

    # ── §5 ───────────────────────────────────────────────────────────────────
    @observe(name="litminex_tool3_summarization")
    def run(
        self,
        query: str,
        ranked_articles: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Produce the query answer plus the per-article table rows."""
        selected = self.select_evidence(ranked_articles, top_n=top_n, min_score=min_score)
        langfuse_context.update_current_observation(
            input={"query": query, "selected_articles": len(selected)}
        )
        if not selected:
            return {
                "query_answer": {
                    "text": "No article in the retrieved set contained a scored "
                            "target-disease relation sentence for this query.",
                    "cited_pmids": [],
                },
                "article_table": [],
            }

        # The query answer and the per-article summaries are independent LLM calls;
        # run them concurrently rather than serially per article.
        with ThreadPoolExecutor(max_workers=settings.LIT_MAX_WORKERS) as pool:
            answer_future = pool.submit(self._answer_query, query, selected)
            summary_futures = {
                article["pmid"]: pool.submit(self._summarise_article, article)
                for article in selected
            }
            answer_text = answer_future.result()
            summaries = {pmid: future.result() for pmid, future in summary_futures.items()}

        cited = _cited_pmids(answer_text, selected)
        table = [
            {
                "pmid": article["pmid"],
                "title": article.get("title", ""),
                "relevance_score": article.get("relevance_score", 0.0),
                "pdf_link": article.get("pdf_link", ""),
                "journal": article.get("journal", ""),
                "pub_date": article.get("pub_date"),
                "per_article_summary": summaries.get(article["pmid"], ""),
            }
            for article in selected
        ]
        langfuse_context.update_current_observation(output={"cited_pmids": len(cited)})
        return {
            "query_answer": {"text": answer_text, "cited_pmids": cited},
            "article_table": table,
        }

    def select_evidence(
        self,
        ranked_articles: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Top N by relevance score, or everything above the threshold — whichever set is
        non-empty. Articles with no relation sentences are dropped: they carry nothing
        Tool 3 is allowed to summarise from.
        """
        top_n = settings.LITMINEX_SUMMARY_TOP_N if top_n is None else top_n
        min_score = settings.LITMINEX_SUMMARY_MIN_SCORE if min_score is None else min_score

        with_evidence = [
            a for a in ranked_articles
            if any(s.get("relation_found") for s in a.get("relation_sentences", []))
        ]
        above_threshold = [a for a in with_evidence if a.get("relevance_score", 0) >= min_score]
        return (above_threshold or with_evidence)[:top_n]

    # ── §6 ───────────────────────────────────────────────────────────────────
    @observe(name="litminex_qa")
    def answer_followup(
        self,
        question: str,
        articles: List[Dict[str, Any]],
        deep: bool = False,
    ) -> Dict[str, Any]:
        """
        Re-invoke the summarisation step with a narrower context: the relation
        sentences (or full section text when `deep`) of the selected articles, with
        the new question in place of the original query. Same citation requirement.
        """
        scoped = [a for a in articles if a]
        if not scoped:
            return {"text": "No articles were selected for this follow-up.", "cited_pmids": []}
        text = self._answer_query(question, scoped, deep=deep)
        return {"text": text, "cited_pmids": _cited_pmids(text, scoped)}

    # ── context assembly + LLM calls ─────────────────────────────────────────
    @staticmethod
    def _build_context(articles: List[Dict[str, Any]], deep: bool = False) -> str:
        """Relation sentences grouped by article, each tagged with its source PMID."""
        blocks: List[str] = []
        for article in articles:
            pmid = article.get("pmid", "")
            if deep and article.get("sections"):
                body = " ".join(
                    f"[{name}] {text}" for name, text in article["sections"].items() if text
                )[:6000]
                lines = [body] if body else []
            else:
                lines = [
                    f"- ({s.get('section', 'abstract')}, {s.get('evidence_type', 'primary')}) "
                    f"{s['text']}"
                    for s in _relation_only(article)[:MAX_SENTENCES_PER_ARTICLE]
                ]
            if not lines:
                continue
            blocks.append(
                f"PMID:{pmid} — {article.get('title', '')}\n" + "\n".join(lines)
            )
        return "\n\n".join(blocks)

    @observe(as_type="generation", name="litminex_answer_generation")
    def _answer_query(self, query: str, articles: List[Dict[str, Any]], deep: bool = False) -> str:
        context = self._build_context(articles, deep=deep)
        if not context:
            return "The retrieved articles contain no relation evidence for this question."
        prompt = (
            f"Question: {query}\n\n"
            f"Evidence sentences (grouped by source article):\n{context}\n\n"
            "Answer the question from this evidence only, citing each claim inline as "
            "[PMID:<pmid>]."
        )
        # 700 truncated multi-article answers mid-sentence in practice.
        text = self._call_llm(_ANSWER_SYSTEM_PROMPT, prompt, max_tokens=1200)
        return text or self._fallback_answer(articles)

    @observe(as_type="generation", name="litminex_article_summary")
    def _summarise_article(self, article: Dict[str, Any]) -> str:
        sentences = _relation_only(article)[:MAX_SENTENCES_PER_ARTICLE]
        if not sentences:
            return ""
        joined = " ".join(s["text"] for s in sentences)
        text = self._call_llm(
            _PER_ARTICLE_SYSTEM_PROMPT,
            f"Title: {article.get('title', '')}\nSentences: {joined}",
            max_tokens=120,
        )
        return text or _truncate(sentences[0]["text"], 240)

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if not (settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN):
            logger.warning("LitMineX: DATABRICKS_HOST/TOKEN unset; using extractive fallback")
            return ""
        try:
            from DRP_Main.app.core.llm import llm_client

            langfuse_context.update_current_observation(model=self.model)
            response = llm_client.databricks(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                endpoint=self.model,
                temperature=0.1,
                max_tokens=max_tokens,
                return_raw=True,
            )
            usage = response.get("usage") or {}
            if usage:
                langfuse_context.update_current_observation(
                    usage={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                    }
                )
            return (response["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            logger.error("LitMineX summarisation LLM call failed: %s", exc)
            return ""

    @staticmethod
    def _fallback_answer(articles: List[Dict[str, Any]]) -> str:
        """
        Extractive answer used when the LLM is unavailable: the highest-similarity
        relation sentence per article, cited. Still evidence-only, still cited.
        """
        parts: List[str] = []
        for article in articles[:5]:
            sentences = _relation_only(article)
            if not sentences:
                continue
            best = max(sentences, key=lambda s: s.get("similarity_score", 0.0))
            parts.append(f"{_truncate(best['text'], 300)} [PMID:{article.get('pmid', '')}]")
        return " ".join(parts) or "No relation evidence available to answer this query."


def _relation_only(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in article.get("relation_sentences", []) if s.get("relation_found")]


def _cited_pmids(answer: str, articles: List[Dict[str, Any]]) -> List[str]:
    """PMIDs the answer actually cites, restricted to the supplied evidence set."""
    available = {str(a.get("pmid", "")) for a in articles}
    seen, out = set(), []
    for pmid in _PMID_CITATION.findall(answer or ""):
        if pmid in available and pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
