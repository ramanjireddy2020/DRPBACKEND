"""
Literature mining agent — batched PubMed retrieval with keyword-gated scoring.

A port of the standalone `LiteratureMiningAgent` design into this codebase. The
scoring contract is kept exactly as specified:

    protein keywords (article_keywords)  MANDATORY  base score up to 60
    context keywords (search_keywords)   OPTIONAL   bonus     up to 30
    title matches                                   +10 per keyword type

and so is the shape of each returned record (``protein_name``, ``title``,
``score``, ``pdf_file_path``, ``found_keywords``, ``preview``).

Two things had to change for it to run on Databricks Apps at all:

1. **Async httpx, not ``requests.Session``.** Runners are coroutines on the
   server's own event loop; a synchronous HTTP call there freezes every request
   in the process, not just its own. That is the exact bug that made
   ``POST /v1/sessions`` take three minutes.
2. **Databricks Model Serving, not Azure AI Projects.** ``DefaultAzureCredential``
   has no managed identity to bind to inside an Apps container, so the original
   ``AIProjectClient`` would fail at construction. ``llm_client.databricks`` is
   the endpoint the rest of the platform already scores with.

Why this reduces NCBI 429s without an API key: the original per-candidate pattern
issues one ``esearch`` **and** one ``efetch`` per protein. This batches instead —
one ``esearch`` for the whole keyword set, then a single ``efetch`` carrying every
returned PMID — so a 20-article search costs 2 requests rather than 40. Requests
also pass through ``app/shared/eutils.py``, the one process-wide gate every
module shares, because NCBI's limit is per IP rather than per caller.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import httpx

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.shared import eutils

logger = get_logger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
REQUEST_TIMEOUT = 15.0

#: Restrict to open access so `pdf_file_path` resolves to something a researcher
#: can actually open, rather than a paywall.
OPEN_ACCESS_FILTER = '("loattrfree full text"[sb] OR "pmc"[Filter])'

#: PubMed's `term` is capped in practice; five keywords is what the agent spec uses.
MAX_QUERY_KEYWORDS = 5


async def _eutils_get(client: httpx.AsyncClient, url: str, params: Dict[str, Any]):
    """
    One throttled, retrying eutils GET through the process-wide gate.

    NCBI's limit is per IP, so pacing has to be shared across every module in the
    process — a private limiter per module is how the pipeline exhausted the
    quota while each module believed it was behaving.
    """
    for attempt in range(eutils.MAX_RETRIES):
        await eutils.throttle_async()
        response = await client.get(url, params=eutils.with_api_key(params))
        if not eutils.is_rate_limited(response.status_code):
            response.raise_for_status()
            return response
        wait = eutils.backoff_seconds(attempt)
        logger.warning(
            "PubMed 429 (attempt %d/%d) — backing off %.1fs",
            attempt + 1, eutils.MAX_RETRIES, wait,
        )
        await asyncio.sleep(wait)
    logger.error("PubMed still rate-limited after %d attempts", eutils.MAX_RETRIES)
    return None


def build_query(article_keywords: List[str]) -> str:
    """`("A"[Title/Abstract] OR "B"[Title/Abstract]) AND (open access)`."""
    terms = " OR ".join(
        f'"{k}"[Title/Abstract]' for k in article_keywords[:MAX_QUERY_KEYWORDS] if k
    )
    return f"({terms}) AND {OPEN_ACCESS_FILTER}"


async def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """Return PMIDs for a query. Over-fetches, since local filtering drops some."""
    params = {
        "db": "pubmed",
        "term": query,
        # 2x, because the keyword gate below discards articles that merely matched
        # the open-access filter without carrying a protein term.
        "retmax": max_results * 2,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await _eutils_get(client, ESEARCH_URL, params)
            if response is None:
                return []
            return response.json().get("esearchresult", {}).get("idlist", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("PubMed search failed for %r: %s", query[:80], exc)
        return []


async def fetch_article_details(pmids: List[str]) -> List[Dict[str, Any]]:
    """Fetch every article in ONE efetch call — the batching that cuts request count."""
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await _eutils_get(client, EFETCH_URL, params)
            if response is None:
                return []
            content = response.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("PubMed efetch failed for %d id(s): %s", len(pmids), exc)
        return []

    # XML parsing is CPU-bound over a large document; keep it off the event loop.
    return await asyncio.to_thread(_parse_articles, content)


def _parse_articles(content: bytes) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        logger.warning("PubMed returned unparseable XML: %s", exc)
        return []
    articles = []
    for node in root.findall(".//PubmedArticle"):
        parsed = _parse_article_xml(node)
        if parsed:
            articles.append(parsed)
    return articles


def _parse_article_xml(xml) -> Optional[Dict[str, Any]]:
    """Pull pmid / pmcid / doi / title / abstract out of one PubmedArticle node."""
    try:
        pmid_elem = xml.find(".//PMID")
        if pmid_elem is None:
            return None

        title_elem = xml.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

        # Structured abstracts split across several AbstractText nodes; a label
        # attribute ("METHODS", "RESULTS") is worth keeping for readability.
        parts = []
        for text_elem in xml.findall(".//AbstractText"):
            text = "".join(text_elem.itertext()).strip()
            if not text:
                continue
            label = text_elem.get("Label")
            parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(parts)

        # `or` is wrong on ElementTree nodes: an Element with no children is
        # falsy, so a present <Year> would be skipped in favour of MedlineDate.
        # Test against None explicitly.
        year_elem = xml.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = xml.find(".//PubDate/MedlineDate")  # e.g. "2023 Jan-Feb"
        if year_elem is None:
            year_elem = xml.find(".//ArticleDate/Year")
        year = None
        if year_elem is not None and year_elem.text:
            digits = "".join(c for c in year_elem.text.strip()[:4] if c.isdigit())
            year = int(digits) if len(digits) == 4 else None

        authors = ""
        first = xml.find(".//Author/LastName")
        if first is not None and first.text:
            authors = f"{first.text} et al."

        pmc_elem = xml.find('.//ArticleId[@IdType="pmc"]')
        doi_elem = xml.find('.//ArticleId[@IdType="doi"]')

        return {
            "pmid": pmid_elem.text,
            "pmcid": pmc_elem.text if pmc_elem is not None else None,
            "doi": doi_elem.text if doi_elem is not None else None,
            "title": title,
            "abstract": abstract,
            "year": year,
            "authors": authors,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/",
        }
    except Exception as exc:  # noqa: BLE001 — one bad record must not lose the batch
        logger.debug("Could not parse a PubMed record: %s", exc)
        return None


def _searchable(article: Dict[str, Any]) -> str:
    return f"{article.get('title', '')} {article.get('abstract', '')}".lower()


def fallback_score(
    article: Dict[str, Any], article_keywords: List[str], search_keywords: List[str]
) -> float:
    """
    Deterministic keyword score, used whenever the LLM cannot be reached.

    Without this an LLM outage would drop articles entirely; with it the ranking
    degrades to keyword relevance instead of disappearing.
    """
    title = article.get("title", "").lower()
    text = _searchable(article)

    if not article_keywords:
        return 0.0

    protein_matches = sum(1 for k in article_keywords if k.lower() in text)
    protein_title = sum(1 for k in article_keywords if k.lower() in title)
    score = (protein_matches / len(article_keywords)) * 50 + protein_title * 10

    if search_keywords:
        context_matches = sum(1 for k in search_keywords if k.lower() in text)
        context_title = sum(1 for k in search_keywords if k.lower() in title)
        score += (context_matches / len(search_keywords)) * 20 + context_title * 10

    return min(100.0, score)


#: Pay-per-token Model Serving endpoints reject bursts. Scoring a 20-article page
#: fired 20 calls at once, every one 429'd, and every article silently fell back to
#: the keyword score — which is why a whole page came back at an identical 90.0.
_MAX_SCORING_CONCURRENCY = 4
_scoring_gates: Dict[int, Any] = {}


def _scoring_gate() -> Any:
    """Per-event-loop semaphore bounding concurrent Model Serving calls."""
    try:
        loop_key = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_key = 0
    gate = _scoring_gates.get(loop_key)
    if gate is None:
        gate = asyncio.Semaphore(_MAX_SCORING_CONCURRENCY)
        if len(_scoring_gates) > 32:
            _scoring_gates.clear()
        _scoring_gates[loop_key] = gate
    return gate


_warned: set = set()


def _warn_once(message: str) -> None:
    """Log a repeated per-article failure once per distinct message."""
    if message not in _warned:
        if len(_warned) > 64:
            _warned.clear()
        _warned.add(message)
        logger.warning("literature agent: %s", message)


async def score_article(
    article: Dict[str, Any], article_keywords: List[str], search_keywords: List[str]
) -> float:
    """Score one article 0-100 with the LLM, falling back to keyword scoring."""
    prompt = f"""Score this article's relevance (0-100):

Title: {article.get('title', '')}
Abstract: {article.get('abstract', '')[:500]}

REQUIRED Protein Keywords: {', '.join(article_keywords)}
OPTIONAL Context Keywords: {', '.join(search_keywords)}

Scoring:
- Base score for protein relevance (up to 60 points)
- Bonus for context keywords (up to 30 points)
- Title matches get +10 points

Return ONLY a JSON object: {{"score": <number>}}"""

    def _call() -> str:
        from DRP_Main.app.core.llm import llm_client

        return llm_client.databricks(
            messages=[
                {
                    "role": "system",
                    "content": "You are a Literature Mining Specialist scoring article relevance. "
                               "Return only the requested JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=64,
            temperature=0.0,
        )

    try:
        async with _scoring_gate():
            raw = await asyncio.to_thread(_call)
        score = float(json.loads(raw).get("score", 0))
        # A model can return anything; keep it inside the contract's range.
        return max(0.0, min(100.0, score))
    except Exception as exc:  # noqa: BLE001
        # Logged at WARNING, not DEBUG: a silent fall-through to keyword scoring
        # makes every article score the same, which reads as a working ranking
        # rather than as a failure. `_warn_once` keeps a 20-article batch from
        # producing 20 identical lines.
        _warn_once(f"LLM scoring unavailable ({type(exc).__name__}: {exc}); using keyword score")
        return fallback_score(article, article_keywords, search_keywords)


def pdf_url(article: Dict[str, Any]) -> str:
    """Best available full-text link: PMC PDF, else DOI, else nothing."""
    if article.get("pmcid"):
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article['pmcid']}/pdf/"
    if article.get("doi"):
        return f"https://doi.org/{article['doi']}"
    return "PDF not available"


def found_keywords(
    article: Dict[str, Any], article_keywords: List[str], search_keywords: List[str]
) -> List[str]:
    text = _searchable(article)
    matched = [k for k in article_keywords if k.lower() in text]
    matched += [k for k in search_keywords if k.lower() in text]
    # dict.fromkeys dedupes while keeping the order the keywords were given in.
    return list(dict.fromkeys(matched))


async def process_search(
    article_keywords: List[str],
    search_keywords: Optional[List[str]] = None,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search, filter and score literature for a set of protein/compound keywords.

    Protein keywords are mandatory: an article that mentions none of them is
    discarded before scoring, so the LLM is never asked about articles that
    cannot qualify. That local gate is also what keeps request volume down.
    """
    article_keywords = [k for k in (article_keywords or []) if k]
    search_keywords = [k for k in (search_keywords or []) if k]
    if not article_keywords:
        return []

    query = build_query(article_keywords)
    pmids = await search_pubmed(query, max_results)
    articles = await fetch_article_details(pmids)
    logger.info("literature agent: retrieved %d article(s) for %s", len(articles), article_keywords)

    filtered = [
        a for a in articles
        if any(k.lower() in _searchable(a) for k in article_keywords)
    ]
    logger.info("literature agent: %d article(s) carry a protein keyword", len(filtered))

    candidates = filtered[:max_results]
    scores = await asyncio.gather(
        *(score_article(a, article_keywords, search_keywords) for a in candidates)
    )

    scored = [
        {
            "pmid": article.get("pmid"),
            "protein_name": ", ".join(article_keywords),
            "title": article.get("title", ""),
            "score": score,
            "pdf_file_path": pdf_url(article),
            "found_keywords": found_keywords(article, article_keywords, search_keywords),
            "preview": (article.get("abstract") or "Abstract not available")[:500],
            "year": article.get("year"),
            "authors": article.get("authors", ""),
            "pubmed_url": article.get("pubmed_url", ""),
        }
        for article, score in zip(candidates, scores)
        if score > 0
    ]
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:max_results]
