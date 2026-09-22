"""
LitMineX Tool 2 — Relevance Scoring (spec §4).

Scores each retrieved article against the query with the biomedical NLP pipeline —
not keyword frequency. Per article, per section:

1. Entity tagging (scispaCy) — target and disease mentions, with section location.
2. Co-occurrence + relation detection — dependency-linked sentences matched against
   the fixed relation-phrase list.
3. Positional weighting — title/abstract 1.0, results/discussion 0.6, refs/intro 0.3.
4. Evidence-type tagging — citation markers near the relation phrase mean `cited`,
   otherwise `primary`.
5. Semantic similarity — SapBERT cosine between the query and each relation sentence.
6. Score combination — relation 40%, position 25%, similarity 20%, primary 15%.
7. Ranking — descending by final 0-100 score.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.literature import relation_config as rc
from DRP_Main.app.modules.literature.embedding_service import EmbeddingService
from DRP_Main.app.modules.literature.ner_service import NERService

logger = get_logger(__name__)

ABSTRACT_PREVIEW_CHARS = 400


class RelevanceService:
    def __init__(
        self,
        ner_service: Optional[NERService] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self.ner = ner_service or NERService()
        self.embeddings = embedding_service or EmbeddingService()

    @observe(name="litminex_tool2_scoring")
    def run(self, retrieval: Dict[str, Any]) -> Dict[str, Any]:
        """Score and rank every article in a Tool 1 output object."""
        articles = retrieval.get("articles", [])
        query = retrieval.get("query", "")
        target_terms = retrieval.get("query_terms", {}).get("target", [])
        disease_terms = retrieval.get("query_terms", {}).get("disease", [])
        langfuse_context.update_current_observation(
            input={"articles": len(articles), "query": query}
        )

        if not articles:
            return {"ranked_articles": [], "similarity_backend": self.embeddings.backend}

        with ThreadPoolExecutor(max_workers=settings.LIT_MAX_WORKERS) as pool:
            scored = list(
                pool.map(
                    lambda a: self._score_article(a, query, target_terms, disease_terms),
                    articles,
                )
            )

        ranked = sorted(
            [a for a in scored if a is not None],
            key=lambda a: a["relevance_score"],
            reverse=True,
        )
        langfuse_context.update_current_observation(output={"ranked": len(ranked)})
        return {
            "ranked_articles": ranked,
            "similarity_backend": self.embeddings.backend,
            "ner_backend": "scispacy" if self.ner.models_available else "rule-based",
        }

    # ── per-article pipeline ─────────────────────────────────────────────────
    def _score_article(
        self,
        article: Dict[str, Any],
        query: str,
        target_terms: List[str],
        disease_terms: List[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            sentences = self._relation_sentences(article, target_terms, disease_terms)
            self._attach_similarity(query, sentences)
            score = self._combine(sentences)
            abstract = article.get("abstract", "") or ""
            return {
                "pmid": article.get("pmid", ""),
                "pmcid": article.get("pmcid"),
                "doi": article.get("doi"),
                "title": article.get("title", ""),
                "journal": article.get("journal", ""),
                "authors": article.get("authors", []),
                "pub_date": article.get("pub_date"),
                "relevance_score": score,
                "relation_sentences": sentences,
                "pdf_link": article.get("pdf_link", ""),
                "pubmed_url": article.get("pubmed_url", ""),
                "abstract_preview": abstract[:ABSTRACT_PREVIEW_CHARS],
                "abstract": abstract,
                "full_text_available": bool(article.get("full_text_available")),
            }
        except Exception as exc:
            logger.warning("LitMineX scoring failed for PMID %s: %s", article.get("pmid"), exc)
            return None

    # ── steps 1–4 ────────────────────────────────────────────────────────────
    def _relation_sentences(
        self,
        article: Dict[str, Any],
        target_terms: List[str],
        disease_terms: List[str],
    ) -> List[Dict[str, Any]]:
        """Every sentence carrying both entities, tagged with relation/position/evidence.

        A concept group can legitimately be empty on the `/v1` single-concept path
        (see `RetrievalService.resolve_entities`). The co-occurrence gate then applies
        only to the concepts actually being searched for — requiring a disease mention
        when no disease was named would zero out every article.
        """
        need_target = bool(target_terms)
        need_disease = bool(disease_terms)
        results: List[Dict[str, Any]] = []
        for section, text in (article.get("sections") or {}).items():
            if not text:
                continue
            weight = rc.section_weight(section)
            for sentence in self.ner.sentences(text):
                tagged = self.ner.tag_mentions(sentence, target_terms, disease_terms)
                if need_target and not tagged["target"]:
                    continue
                if need_disease and not tagged["disease"]:
                    continue
                if not (tagged["target"] or tagged["disease"]):
                    continue
                phrase = rc.match_relation_phrase(sentence)
                # The dependency check confirms the phrase joins the two entities —
                # meaningful only when there are two entities to join.
                relation_found = bool(phrase) and (
                    self.ner.dependency_link(sentence, tagged["target"], tagged["disease"])
                    if (tagged["target"] and tagged["disease"])
                    else True
                )
                span = rc.find_relation_span(sentence, phrase) if phrase else None
                cited = rc.is_cited(sentence, span)
                results.append(
                    {
                        "text": sentence.strip(),
                        "section": section,
                        "relation_found": relation_found,
                        "relation_phrase": phrase if relation_found else None,
                        "evidence_type": "cited" if cited else "primary",
                        "cited": cited,
                        "primary": not cited,
                        "position_weight": weight,
                        "target_mentions": tagged["target"],
                        "disease_mentions": tagged["disease"],
                        "similarity_score": 0.0,
                    }
                )
        return results

    # ── step 5 ───────────────────────────────────────────────────────────────
    def _attach_similarity(self, query: str, sentences: List[Dict[str, Any]]) -> None:
        relation_bearing = [s for s in sentences if s["relation_found"]]
        if not relation_bearing:
            return
        scores = self.embeddings.similarities(query, [s["text"] for s in relation_bearing])
        for sentence, score in zip(relation_bearing, scores):
            sentence["similarity_score"] = round(float(score), 4)

    # ── step 6 ───────────────────────────────────────────────────────────────
    @staticmethod
    def _combine(sentences: List[Dict[str, Any]]) -> float:
        """
        Fold the four signals into one 0-100 score using the fixed weighted formula.

        Each component is the best evidence the article offers for it, so a single
        strong primary result sentence is not diluted by weaker co-occurrences:
        relation match is 1.0 when any sentence has one, positional weight and
        similarity take the maximum over relation-bearing sentences, and primary
        evidence scores 1.0 if any relation sentence is uncited.
        """
        if not sentences:
            return 0.0
        relation_bearing = [s for s in sentences if s["relation_found"]]
        if not relation_bearing:
            # Co-occurrence without a relation still carries some signal: position
            # only, and only a quarter of the weight it would get with a relation.
            position_only = max(s["position_weight"] for s in sentences)
            return round(rc.SCORE_WEIGHTS["position"] * position_only * 100 * 0.5, 1)

        relation = 1.0
        position = max(s["position_weight"] for s in relation_bearing)
        similarity = max(s["similarity_score"] for s in relation_bearing)
        primary = 1.0 if any(s["primary"] for s in relation_bearing) else 0.0

        score = (
            rc.SCORE_WEIGHTS["relation"] * relation
            + rc.SCORE_WEIGHTS["position"] * position
            + rc.SCORE_WEIGHTS["similarity"] * similarity
            + rc.SCORE_WEIGHTS["primary"] * primary
        ) * 100
        return round(max(0.0, min(100.0, score)), 1)
