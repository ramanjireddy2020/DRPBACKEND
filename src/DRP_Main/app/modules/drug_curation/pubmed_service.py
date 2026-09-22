"""
PubMed search service for Drug Curation module.
Split from app/services/drug_curation_services.py.
"""
import time
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import requests
from langfuse.decorators import observe, langfuse_context

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.shared import eutils

logger = get_logger(__name__)

REQUEST_TIMEOUT = 10


class PubMedService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

    @observe(name="pubmed_search_drug")
    def search_pubmed(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search PubMed for drug-related articles."""
        langfuse_context.update_current_observation(input={"query": query, "max_results": max_results})
        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            }
            data = self._get_json(search_url, params)
            if data is None:
                return []
            if "esearchresult" not in data or not data["esearchresult"]["idlist"]:
                return []
            ids = data["esearchresult"]["idlist"][:max_results]
            articles = self._fetch_article_details(ids)
            langfuse_context.update_current_observation(output={"articles_found": len(articles)})
            return articles
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    def _get(self, url: str, params: Dict):
        """
        One throttled, retrying eutils GET.

        This service is called once per candidate compound — twenty-odd requests
        back to back. Unthrottled, that alone exhausts NCBI's per-IP budget and
        starves every other module in the process, which is how a pipeline run
        left LitMineX with zero articles for a query that works on its own.
        """
        for attempt in range(eutils.MAX_RETRIES):
            eutils.throttle_sync()
            response = self.session.get(
                url, params=eutils.with_api_key(params), timeout=REQUEST_TIMEOUT
            )
            if not eutils.is_rate_limited(response.status_code):
                response.raise_for_status()
                return response
            wait = eutils.backoff_seconds(attempt)
            logger.warning(
                "PubMed 429 (attempt %d/%d) — backing off %.1fs",
                attempt + 1, eutils.MAX_RETRIES, wait,
            )
            time.sleep(wait)
        logger.error("PubMed still rate-limited after %d attempts", eutils.MAX_RETRIES)
        return None

    def _get_json(self, url: str, params: Dict) -> Optional[Dict]:
        response = self._get(url, params)
        return response.json() if response is not None else None

    def _fetch_article_details(self, ids: List[str]) -> List[Dict]:
        if not ids:
            return []
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        try:
            response = self._get(fetch_url, params)
            if response is None:
                return []
            root = ET.fromstring(response.content)
            articles = []
            for article in root.findall(".//PubmedArticle"):
                parsed = self._parse_article_xml(article)
                if parsed:
                    articles.append(parsed)
            return articles
        except Exception as e:
            logger.error(f"PubMed fetch details error: {e}")
            return []

    def _parse_article_xml(self, article_xml) -> Optional[Dict]:
        try:
            pmid_elem = article_xml.find(".//PMID")
            title_elem = article_xml.find(".//ArticleTitle")
            abstract_elem = article_xml.find(".//AbstractText")
            if pmid_elem is None or title_elem is None:
                return None
            pmid = pmid_elem.text
            title = title_elem.text or ""
            abstract = abstract_elem.text if abstract_elem is not None else ""
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        except Exception as e:
            logger.error(f"Error parsing article XML: {e}")
            return None
