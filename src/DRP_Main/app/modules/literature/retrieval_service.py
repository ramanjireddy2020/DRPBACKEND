"""
LitMineX Tool 1 — Query Expansion and Retrieval (spec §3).

Turns the agent's input into a proper PubMed search and pulls back candidate
articles:

1. Term extraction — scispaCy NER when the input is a raw query string; skipped when
   structured `target` / `disease` fields are already present.
2. MeSH expansion — the NLM thesaurus via E-utilities, per term.
3. Query construction — field-tagged term groups, target AND disease, with an
   optional publication-type filter.
4. Retrieval — esearch for PMIDs (capped), batched efetch for full records, plus PMC
   open-access full text when available.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.literature.mesh_service import MeSHService
from DRP_Main.app.modules.literature.ner_service import NERService
from DRP_Main.app.modules.literature.pubmed_service import PubMedService

logger = get_logger(__name__)


class MissingEntityError(ValueError):
    """Raised when the query names neither a target nor a disease we can act on.

    Carries the fields still needed so the agent can ask the supervisor to prompt the
    user once, rather than guessing (spec §2).
    """

    def __init__(self, missing: List[str], extracted: Dict[str, Optional[str]]):
        self.missing = missing
        self.extracted = extracted
        super().__init__(f"LitMineX needs the {' and '.join(missing)} to run")


class RetrievalService:
    def __init__(
        self,
        mesh_service: Optional[MeSHService] = None,
        pubmed_service: Optional[PubMedService] = None,
        ner_service: Optional[NERService] = None,
    ) -> None:
        self.mesh = mesh_service or MeSHService(db_path=settings.LIT_MESH_CACHE_DB)
        self.pubmed = pubmed_service or PubMedService(timeout=settings.LIT_REQUEST_TIMEOUT)
        self.ner = ner_service or NERService()

    # ── step 1 ───────────────────────────────────────────────────────────────
    def resolve_entities(
        self, payload: Dict[str, Any], require_both: bool = True
    ) -> Dict[str, Optional[str]]:
        """
        Normalise either input shape (spec §2) into `{"target", "disease", "query"}`.

        Structured input from the TxKG carry-over skips NER entirely; a fresh user
        query is run through it.

        Raises `MissingEntityError` if *either* concept is still absent afterwards.
        The spec is explicit that a query naming only one of them is not searchable:
        the whole point of Tool 1's query construction is ANDing the two groups, so
        running with one would silently return the "either concept" result set the
        AND is there to prevent. Ask once instead of guessing (§2).

        `require_both=False` is a deliberate departure from the spec's chain, used by
        the `/v1` surface where the user picks targets from a list and may not have
        named a disease at all. Retrieval then runs single-concept and Tool 2 falls
        back to relation detection against the one entity it has, so scores are not
        comparable with a two-concept run.
        """
        target = (payload.get("target") or "").strip() or None
        disease = (payload.get("disease") or "").strip() or None
        query = (payload.get("query") or "").strip()

        if not (target and disease) and query:
            extracted = self.ner.extract_query_entities(query)
            target = target or extracted["target"]
            disease = disease or extracted["disease"]

        missing = [name for name, value in (("target", target), ("disease", disease)) if not value]
        if missing and (require_both or len(missing) == 2):
            raise MissingEntityError(missing, {"target": target, "disease": disease})

        return {
            "target": target,
            "disease": disease,
            "query": query or _synthesise_query(target, disease),
        }

    # ── steps 2–4 ────────────────────────────────────────────────────────────
    @observe(name="litminex_tool1_retrieval")
    def run(
        self,
        payload: Dict[str, Any],
        max_results: Optional[int] = None,
        exclude_reviews: Optional[bool] = None,
        fetch_fulltext: Optional[bool] = None,
        require_both: bool = True,
    ) -> Dict[str, Any]:
        """Execute Tool 1 and return the object Tool 2 consumes."""
        resolved = self.resolve_entities(payload, require_both=require_both)
        target, disease = resolved["target"], resolved["disease"]
        limit = int(max_results or settings.LITMINEX_RETRIEVAL_LIMIT)
        exclude_reviews = (
            settings.LITMINEX_EXCLUDE_REVIEWS if exclude_reviews is None else exclude_reviews
        )
        fetch_fulltext = (
            settings.LITMINEX_FETCH_FULLTEXT if fetch_fulltext is None else fetch_fulltext
        )
        langfuse_context.update_current_observation(
            input={"target": target, "disease": disease, "max_results": limit}
        )

        target_terms = self.mesh.expand_terms([target]) if target else []
        disease_terms = self.mesh.expand_terms([disease]) if disease else []

        pubmed_query = self.pubmed.build_litminex_query(
            target_terms, disease_terms, exclude_reviews=exclude_reviews
        )
        logger.info("LitMineX PubMed query: %s", pubmed_query)

        pmids = self.pubmed.esearch_pmids(pubmed_query, limit)
        articles = self.pubmed.fetch_records(pmids, batch_size=settings.LITMINEX_EFETCH_BATCH)
        if fetch_fulltext:
            self._attach_sections(articles)
        else:
            for article in articles:
                article["sections"] = _abstract_only_sections(article)

        langfuse_context.update_current_observation(
            output={"pmids": len(pmids), "articles": len(articles)}
        )
        return {
            "query": resolved["query"],
            "target": target,
            "disease": disease,
            "pubmed_query": pubmed_query,
            "query_terms": {"target": target_terms, "disease": disease_terms},
            "articles": articles,
        }

    def _attach_sections(self, articles: List[Dict[str, Any]]) -> None:
        """
        Fetch PMC open-access sections in parallel. Articles outside the OA subset
        keep title + abstract only, which Tool 2 handles at reduced positional
        resolution rather than skipping.
        """
        oa = [a for a in articles if a.get("pmcid")]
        if oa:
            with ThreadPoolExecutor(max_workers=settings.LIT_MAX_WORKERS) as pool:
                fetched = list(
                    pool.map(lambda a: self.pubmed.fetch_fulltext_sections(a["pmcid"]), oa)
                )
            for article, sections in zip(oa, fetched):
                article["sections"] = {**_abstract_only_sections(article), **sections}
                article["full_text_available"] = bool(sections)
        for article in articles:
            if not article.get("sections"):
                article["sections"] = _abstract_only_sections(article)
                article["full_text_available"] = False


def _abstract_only_sections(article: Dict[str, Any]) -> Dict[str, str]:
    sections = {}
    if article.get("title"):
        sections["title"] = article["title"]
    if article.get("abstract"):
        sections["abstract"] = article["abstract"]
    return sections


def _synthesise_query(target: Optional[str], disease: Optional[str]) -> str:
    """The question a TxKG carry-over implies, used to anchor semantic similarity."""
    if target and disease:
        return f"What is the involvement of {target} in {disease}?"
    return f"What is known about {target or disease}?"
