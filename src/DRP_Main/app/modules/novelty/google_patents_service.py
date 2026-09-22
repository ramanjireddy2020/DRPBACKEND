"""
NovSearch — Google Patents client via SerpAPI (spec §3 step 2, §4 step 2).

Replaces `uspto_odp_service.py`: the USPTO Open Data Portal requires an
ID.me-verified MyUSPTO account to issue a key, which was never obtained, so
NovSearch could never actually retrieve a patent. SerpAPI's `google_patents`
engine needs only `SERPAPI_API_KEY` (already used elsewhere in the platform —
TxKG's patent lookups, the legacy novelty agent) and covers the same two needs:

  * search — SerpAPI's `google_patents` engine returns bibliographic hits
    (title, snippet, assignee, filing/publication dates) directly, in its own
    relevance order (`api_rank` for Tool 1's reciprocal rank fusion).
  * full content — SerpAPI does not include claims text in search results, and
    a per-patent detail call for every indexed patent is not worth metering.
    Full text is instead scraped from `patents.google.com/patent/<id>/en`
    (public, unauthenticated) and parsed into the same
    title/abstract/background/summary/description/claims shape
    `uspto_odp_service.fetch_patent_content()` produced, so
    `indexing_service.py` and `synthesis_service.py` need no changes.

Function names and return shapes intentionally mirror `uspto_odp_service.py`
exactly (`search_patents`, `search_terms`, `normalize_patent_id`,
`display_patent_id`, `fetch_patent_content`) — every call site imports this
module `as pv`, so switching is a one-line change per file.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class GooglePatentsError(RuntimeError):
    """SerpAPI or patents.google.com rejected a request or could not be reached."""


def _require_key() -> str:
    key = settings.SERPAPI_API_KEY
    if not key:
        raise GooglePatentsError(
            "SERPAPI_API_KEY is not configured. NovSearch retrieves patents via "
            "SerpAPI's google_patents engine — set SERPAPI_API_KEY."
        )
    return key


# ══════════════════════════════════════════════════════════════════════════════
#  §3 step 2 — search
# ══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "is", "are", "there", "existing", "for", "the", "a", "an", "of", "in", "on",
    "any", "patent", "patents", "coverage", "and", "or", "to", "with", "does",
    "do", "what", "which", "how", "novelty", "prior", "art",
}


def search_terms(query: str) -> List[str]:
    """Drop question scaffolding so the search sees the drug/target/disease terms."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query or "")
    kept = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    return kept or tokens


async def search_patents(query: str, size: int = 40) -> List[Dict[str, Any]]:
    """
    Search for the query terms and return candidates in SerpAPI's relevance order.

    That ordering is one of the two inputs to the reciprocal rank fusion in Tool 1,
    so it is preserved as `api_rank` rather than being re-sorted here.
    """
    terms = search_terms(query)
    if not terms:
        return []

    key = _require_key()
    params = {
        "engine": "google_patents",
        "q": " ".join(terms),
        "num": min(size, 100),  # SerpAPI's google_patents caps per-page results at 100
        "api_key": key,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.NOVSEARCH_REQUEST_TIMEOUT) as client:
            resp = await client.get("https://serpapi.com/search", params=params)
    except httpx.HTTPError as exc:
        raise GooglePatentsError(f"SerpAPI google_patents search unreachable: {exc}") from exc
    if resp.status_code >= 400:
        raise GooglePatentsError(f"SerpAPI returned {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    if "error" in payload:
        raise GooglePatentsError(f"SerpAPI error: {payload['error']}")

    results: List[Dict[str, Any]] = []
    for rank, record in enumerate(payload.get("organic_results", []), start=1):
        pid = normalize_patent_id(_extract_patent_id(record))
        if not pid:
            continue
        snippet = (record.get("snippet") or "").strip()
        results.append(
            {
                "patent_id": pid,
                "title": (record.get("title") or "").strip(),
                "abstract": snippet,
                "abstract_snippet": snippet[:500],
                "assignee": _first(record.get("assignee")),
                "filing_date": record.get("filing_date"),
                "publication_date": record.get("publication_date") or record.get("grant_date"),
                "api_rank": rank,
            }
        )
    logger.info("SerpAPI google_patents search '%s' → %d candidates", query, len(results))
    return results[:size]


def _first(value: Any) -> str:
    """SerpAPI's `assignee`/`inventor` fields are sometimes a list, sometimes a string."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "").strip()


def _extract_patent_id(record: Dict[str, Any]) -> str:
    pid = record.get("patent_id") or record.get("publication_number")
    if pid:
        return str(pid)
    for key in ("patent_link", "link", "pdf"):
        url = record.get(key)
        if isinstance(url, str):
            match = re.search(r"/patent/([A-Z0-9]+)", url, re.IGNORECASE)
            if match:
                return match.group(1)
    title = record.get("title", "")
    match = re.search(r"\b([A-Z]{2}(?:RE)?\d+[A-Z0-9]*)\b", title.upper())
    return match.group(1) if match else ""


def normalize_patent_id(patent_id: str) -> str:
    """Canonical storage form: `US10123456B2` — uppercase, no URL/whitespace noise."""
    pid = (patent_id or "").strip()
    pid = re.sub(r"^/?patent/", "", pid, flags=re.IGNORECASE)
    pid = re.sub(r"^https?://patents\.google\.com/patent/", "", pid, flags=re.IGNORECASE)
    pid = pid.upper()
    pid = re.sub(r"/[A-Z]{2}$", "", pid)  # trailing language suffix, e.g. "/EN"
    pid = re.sub(r"[?#].*$", "", pid)
    pid = re.sub(r"[\s.,\-]+", "", pid)
    return pid


def display_patent_id(patent_id: str) -> str:
    """Already in the platform's `US…` display form — kept for interface parity with uspto_odp_service."""
    return normalize_patent_id(patent_id)


# ══════════════════════════════════════════════════════════════════════════════
#  §4 step 2 — structured full content (scraped from patents.google.com)
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_patent_content(patent_id: str) -> Optional[Dict[str, Any]]:
    """
    One patent's structured content: title, abstract, background/summary/
    description, and claims as individual objects with dependency information.

    Returns None when the page can't be fetched/parsed, so the caller skips
    that patent rather than indexing an empty shell.
    """
    pid = normalize_patent_id(patent_id)
    html = await _fetch_html(pid)
    if not html:
        return None

    parsed = _parse_patent_html(html, pid)
    claims = parsed.get("claims", [])
    if not claims:
        # Loud, because a patent indexed without claims scores its novelty on
        # description prose alone — which is exactly what §4's weights avoid.
        logger.warning("No claims parsed for patent %s — indexing without claims", pid)

    return {
        "patent_id": pid,
        "display_id": display_patent_id(pid),
        "title": parsed.get("title", ""),
        "abstract": parsed.get("abstract", ""),
        "summary": parsed.get("summary", ""),
        "description": parsed.get("description", ""),
        "background": parsed.get("background", ""),
        "assignee": parsed.get("assignee", ""),
        "inventors": parsed.get("inventors", []),
        "filing_date": parsed.get("filing_date"),
        "publication_date": parsed.get("publication_date"),
        "independent_claims": [c for c in claims if not c["dependent"]],
        "dependent_claims": [c for c in claims if c["dependent"]],
    }


async def _fetch_html(patent_id: str) -> Optional[str]:
    url = f"https://patents.google.com/patent/{patent_id}/en"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.text
        logger.warning("Google Patents HTTP %s fetching %s", resp.status_code, patent_id)
    except httpx.HTTPError as exc:
        logger.warning("Google Patents fetch failed for %s: %s", patent_id, exc)
    return None


_SECTION_PATTERNS = (
    ("background", r"(?:BACKGROUND(?:\s+OF\s+THE\s+INVENTION)?|FIELD\s+OF\s+THE\s+INVENTION)"),
    ("summary", r"(?:BRIEF\s+SUMMARY|SUMMARY(?:\s+OF\s+THE\s+INVENTION)?)"),
    ("description", r"(?:DETAILED\s+DESCRIPTION|DESCRIPTION\s+OF\s+THE\s+(?:PREFERRED\s+)?EMBODIMENTS?)"),
)

_DEP_PATTERNS = (
    r"claims?\s+(\d+)\s*(?:-|to|or|and)\s*(\d+)",
    r"claims?\s+(\d+)",
)


def _depends_on(text: str) -> List[int]:
    numbers: set = set()
    for pattern in _DEP_PATTERNS:
        for match in re.findall(pattern, text, re.IGNORECASE):
            values = match if isinstance(match, tuple) else (match,)
            numbers.update(int(v) for v in values if str(v).isdigit())
    return sorted(numbers)


def _parse_patent_html(html: str, patent_id: str) -> Dict[str, Any]:
    """Parse title/abstract/sections/claims out of a Google Patents page."""
    from bs4 import BeautifulSoup  # lazy: keeps bs4 off the hot import path

    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, Any] = {}

    title_el = soup.find("span", {"itemprop": "title"}) or soup.find("meta", {"name": "DC.title"})
    if title_el:
        data["title"] = (title_el.get("content") if title_el.name == "meta" else title_el.get_text(strip=True)) or ""

    abstract_el = soup.find("section", {"itemprop": "abstract"}) or soup.find("div", class_="abstract")
    if abstract_el:
        data["abstract"] = re.sub(r"^Abstract\s*", "", abstract_el.get_text(" ", strip=True), flags=re.I)

    full_text = soup.get_text("\n", strip=True)
    _fill_sections_from_text(data, full_text)

    assignee_el = soup.find("dd", {"itemprop": "assigneeCurrentOriginal"}) or soup.find("dd", {"itemprop": "assigneeOriginal"})
    if assignee_el:
        data["assignee"] = assignee_el.get_text(" ", strip=True)

    data["inventors"] = [
        el.get_text(" ", strip=True) for el in soup.find_all("dd", {"itemprop": "inventor"})
    ]

    filing_el = soup.find("time", {"itemprop": "filingDate"})
    if filing_el:
        data["filing_date"] = filing_el.get("datetime") or filing_el.get_text(strip=True)
    pub_el = soup.find("time", {"itemprop": "publicationDate"})
    if pub_el:
        data["publication_date"] = pub_el.get("datetime") or pub_el.get_text(strip=True)

    data["claims"] = _parse_claims(soup, full_text)
    return data


def _fill_sections_from_text(data: Dict[str, Any], full_text: str) -> None:
    matches = []
    for name, pattern in _SECTION_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            matches.append((match.start(), match.end(), name))
    matches.sort()
    if not matches:
        return
    for i, (_start, end, name) in enumerate(matches):
        stop = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
        data[name] = full_text[end:stop].strip(" :.—-").strip()[:20000]


def _parse_claims(soup: Any, full_text: str) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    seen: set = set()

    claims_section = None
    for tag, attrs in (
        ("section", {"itemprop": "claims"}),
        ("div", {"class": "claims"}),
        ("section", {"class": "claims"}),
    ):
        claims_section = soup.find(tag, attrs)
        if claims_section:
            break

    if claims_section:
        for el in claims_section.find_all("div", class_=lambda c: c and "claim" in c):
            number = el.get("num")
            if not number:
                match = re.match(r"^\s*(\d+)[.)]\s*", el.get_text(" ", strip=True))
                number = match.group(1) if match else None
            if not number:
                continue
            try:
                num_int = int(re.search(r"\d+", str(number)).group())
            except (AttributeError, ValueError):
                continue
            if num_int in seen:
                continue
            seen.add(num_int)
            text = re.sub(r"^\s*\d+[.)]\s*", "", el.get_text(" ", strip=True))
            if not text:
                continue
            depends_on = _depends_on(text)
            claims.append(
                {"number": num_int, "text": text, "dependent": bool(depends_on), "depends_on": depends_on}
            )

    if not claims:
        # Fallback: regex over the plain-text claims block when the page's markup
        # doesn't match the expected itemprop/class structure.
        start = -1
        for kw in ("claims", "what is claimed", "the invention claimed"):
            pos = full_text.lower().find(kw)
            if pos != -1:
                start = pos
                break
        claims_text = full_text[start:] if start != -1 else ""
        for cn, ctxt in re.findall(r"(?:^|\n)\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.\s|$)", claims_text, re.DOTALL):
            num_int = int(cn)
            if num_int in seen:
                continue
            seen.add(num_int)
            text = re.sub(r"\s+", " ", ctxt.strip())
            if not text:
                continue
            depends_on = _depends_on(text)
            claims.append(
                {"number": num_int, "text": text, "dependent": bool(depends_on), "depends_on": depends_on}
            )

    claims.sort(key=lambda c: c["number"])
    return claims
