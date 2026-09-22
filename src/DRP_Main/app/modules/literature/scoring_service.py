"""
LLM relevance scoring service for Literature Mining.
Split from app/api/v1/endpoints/literature_mining.py.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from fastapi import HTTPException
from langfuse.decorators import observe, langfuse_context

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.literature.mesh_service import MeSHService
from DRP_Main.app.modules.literature.pubmed_service import PubMedService

logger = get_logger(__name__)

MAX_WORKERS = 10
MAX_ABSTRACT_LENGTH = 500


class LiteratureService:
    def __init__(self):
        self.mesh_service = MeSHService()
        self.pubmed_service = PubMedService()
        self.max_workers = MAX_WORKERS

    @observe(name="literature_search")
    def search_keywords(
        self,
        article_keywords: List[str],
        search_keywords: List[str],
        max_results: int = 20,
    ) -> List[Dict]:
        """Orchestrate: MeSH lookup → PubMed search → keyword filter → LLM scoring."""
        import time
        langfuse_context.update_current_observation(
            input={
                "article_keywords": article_keywords,
                "search_keywords": search_keywords,
                "max_results": max_results,
            }
        )
        start = time.time()
        try:
            mesh_data = self.mesh_service.get_mesh_terms(article_keywords)
            search_query = self.pubmed_service.build_search_query(article_keywords, mesh_data)
            articles = self.pubmed_service.search(search_query, max_results * 2)
            logger.info("Retrieved %d articles in %.2fs", len(articles), time.time() - start)

            if not articles:
                langfuse_context.update_current_observation(output={"count": 0})
                return []

            articles_with_proteins = self._filter_by_mandatory_keywords(articles, article_keywords)
            if not articles_with_proteins:
                logger.warning("No articles contained the required protein names")
                langfuse_context.update_current_observation(output={"count": 0})
                return []

            scored = self._batch_score_articles(articles_with_proteins, article_keywords, search_keywords)
            filtered = sorted(
                [a for a in scored if a.get("relevance_score", 0) > 0],
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )[:max_results]

            results = [
                self._to_output_format(a, article_keywords, search_keywords)
                for a in filtered
            ]
            langfuse_context.update_current_observation(output={"count": len(results)})
            return results

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Literature search failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def _filter_by_mandatory_keywords(
        self, articles: List[Dict], article_keywords: List[str]
    ) -> List[Dict]:
        filtered = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
            if any(kw.lower() in text for kw in article_keywords):
                article["found_protein_keywords"] = [
                    kw for kw in article_keywords if kw.lower() in text
                ]
                filtered.append(article)
        return filtered

    @observe(name="article_batch_scoring")
    def _batch_score_articles(
        self,
        articles: List[Dict],
        article_keywords: List[str],
        search_keywords: List[str],
    ) -> List[Dict]:
        langfuse_context.update_current_observation(input={"article_count": len(articles)})
        scored = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_article = {
                executor.submit(
                    self._calculate_llm_relevance_score, article, article_keywords, search_keywords
                ): article
                for article in articles
            }
            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    score = future.result()
                except Exception as e:
                    logger.warning("Scoring failed, falling back: %s", e)
                    score = self._fallback_scoring(article, article_keywords, search_keywords)
                if score > 0:
                    article["relevance_score"] = score
                    article["matched_keywords"] = self._find_matched_keywords(
                        article, article_keywords, search_keywords
                    )
                    scored.append(article)
        langfuse_context.update_current_observation(output={"scored_count": len(scored)})
        return scored

    @observe(as_type="generation", name="relevance_scoring")
    def _calculate_llm_relevance_score(
        self,
        article: Dict,
        article_keywords: List[str],
        search_keywords: List[str],
    ) -> float:
        langfuse_context.update_current_observation(model=settings.DATABRICKS_LLM_ENDPOINT)
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        if not title and not abstract:
            return 0.0
        abstract_short = abstract[:MAX_ABSTRACT_LENGTH]
        messages = [
            {
                "role": "system",
                "content": "Score relevance. Protein keywords are mandatory, context keywords are bonus. Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    f"Title: {title}\nAbstract: {abstract_short}\n\n"
                    f"REQUIRED Protein Keywords: {', '.join(article_keywords)}\n"
                    f"OPTIONAL Context Keywords: {', '.join(search_keywords)}\n\n"
                    'Score 0-100. Return JSON: {"score": <number>}'
                ),
            },
        ]
        try:
            from DRP_Main.app.core.llm import llm_client

            response = llm_client.databricks(
                messages=messages,
                temperature=0.1,
                max_tokens=50,
                return_raw=True,
            )
            result = json.loads(response["choices"][0]["message"]["content"])
            score = max(0.0, min(100.0, float(result.get("score", 0))))
            usage = response.get("usage") or {}
            langfuse_context.update_current_observation(
                usage={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                }
            )
            return score
        except Exception as e:
            logger.debug("LLM scoring failed, falling back: %s", e)
            return self._fallback_scoring(article, article_keywords, search_keywords)

    def _fallback_scoring(
        self,
        article: Dict,
        article_keywords: List[str],
        search_keywords: List[str],
    ) -> float:
        text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
        title = article.get("title", "").lower()
        if not text.strip():
            return 0.0
        protein_matches = sum(1 for kw in article_keywords if kw.lower() in text)
        if protein_matches == 0:
            return 0.0
        protein_title_matches = sum(1 for kw in article_keywords if kw.lower() in title)
        protein_score = (protein_matches / len(article_keywords)) * 50
        protein_title_bonus = protein_title_matches * 10
        search_score = search_title_bonus = 0
        if search_keywords:
            search_matches = sum(1 for kw in search_keywords if kw.lower() in text)
            search_title_matches = sum(1 for kw in search_keywords if kw.lower() in title)
            search_score = (search_matches / len(search_keywords)) * 20
            search_title_bonus = search_title_matches * 10
        return min(100.0, protein_score + protein_title_bonus + search_score + search_title_bonus)

    def _find_matched_keywords(
        self,
        article: Dict,
        article_keywords: List[str],
        search_keywords: List[str],
    ) -> List[str]:
        text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
        matched = [kw for kw in article_keywords if kw.lower() in text]
        matched += [kw for kw in search_keywords if kw.lower() in text]
        return list(set(matched))

    def _to_output_format(
        self,
        article: Dict,
        article_keywords: List[str],
        search_keywords: List[str],
    ) -> Dict:
        pmcid = article.get("pmcid")
        doi = article.get("doi")
        if pmcid:
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        elif doi:
            pdf_url = f"https://doi.org/{doi}"
        else:
            pdf_url = "PDF not available"
        abstract = article.get("abstract", "")
        return {
            "protein_name": ", ".join(article_keywords) if article_keywords else "N/A",
            "title": article.get("title", "Title not available"),
            "score": round(article.get("relevance_score", 0), 1),
            "pdf_file_path": pdf_url,
            "found_keywords": article.get("matched_keywords", []),
            "preview": abstract if abstract.strip() else "Abstract not available",
            # Citation metadata — consumed by the DRP /v1 article endpoints. Not
            # part of ArticleResult, so it is dropped from the legacy response.
            "pmid": article.get("pmid", ""),
            "pmcid": article.get("pmcid"),
            "doi": article.get("doi"),
            "year": article.get("year"),
            "authors": article.get("authors", ""),
            "journal": article.get("journal", ""),
            "abstract": abstract,
            "keywords": article.get("keywords", []),
            "pubmed_url": article.get("pubmed_url", ""),
        }
