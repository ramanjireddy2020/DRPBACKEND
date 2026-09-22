"""
MeSH term expansion service with SQLite cache.

Two expansion paths share the same cache table:

* `expand_term()` — LitMineX Tool 1 (spec §3 step 2). Queries the NLM MeSH thesaurus
  through E-utilities (`esearch`/`efetch` on `db=mesh`) for a term's official MeSH
  heading plus its listed synonyms and entry terms. This is the authoritative path.
* `get_mesh_terms()` — the legacy keyword endpoint's LLM-assisted expansion, kept
  because `scoring_service.LiteratureService` and the `/v1` runners still call it.
"""
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import requests
from langfuse.decorators import observe, langfuse_context

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

MESH_CACHE_DAYS = 30

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI allows 3 requests/second without an API key, 10 with one.
_MIN_INTERVAL_NO_KEY = 0.34
_MIN_INTERVAL_WITH_KEY = 0.11

# Entry-term lists run long (AMPK alone has 20+ subunit variants). Keeping the whole
# list here is fine — the query builder is what caps how many reach PubMed.
MAX_ENTRY_TERMS = 40


class MeSHService:
    def __init__(self, db_path: str = "data/mesh_cache.db", cache_days: int = MESH_CACHE_DAYS):
        self.db_path = db_path
        self.cache_days = cache_days
        self.api_key = os.environ.get("NCBI_API_KEY", "")
        self._min_interval = _MIN_INTERVAL_WITH_KEY if self.api_key else _MIN_INTERVAL_NO_KEY
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "InnoDD-LitMineX/1.0"})
        self._initialize_db()

    def _initialize_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mesh_cache (
                query TEXT PRIMARY KEY,
                mesh_terms TEXT,
                created_date TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_date ON mesh_cache(created_date)")
        conn.commit()
        conn.close()

    # ── LitMineX Tool 1: NLM MeSH thesaurus expansion (spec §3 step 2) ───────
    @observe(name="mesh_thesaurus_expansion")
    def expand_term(self, term: str) -> Dict:
        """
        Expand one term against the NLM MeSH thesaurus.

        Returns::

            {"input": "AMPK",
             "heading": "AMP-Activated Protein Kinases",
             "entry_terms": ["5' AMP-Activated Protein Kinase", ...],
             "terms": [heading + entry terms + the input, deduped],
             "source": "mesh" | "unmapped"}

        A term with no MeSH record (many gene symbols) comes back as ``unmapped``
        with the input as its only term — the query builder still field-tags it into
        the ``[Title/Abstract]`` half of the group, so nothing is lost.
        """
        term = (term or "").strip()
        if not term:
            return {"input": term, "heading": None, "entry_terms": [], "terms": [], "source": "empty"}

        cache_key = f"mesh_thesaurus::{term.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug("MeSH thesaurus cache hit for %s", term)
            return cached

        result = self._fetch_mesh_record(term)
        self._cache_result(cache_key, result)
        langfuse_context.update_current_observation(input={"term": term}, output=result)
        return result

    def expand_terms(self, terms: List[str]) -> List[str]:
        """Flatten `expand_term` over several inputs into one deduped term list."""
        out: List[str] = []
        seen: set = set()
        for term in terms:
            for expanded in self.expand_term(term).get("terms", []):
                key = expanded.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(expanded)
        return out

    def _fetch_mesh_record(self, term: str) -> Dict:
        unmapped = {
            "input": term,
            "heading": None,
            "entry_terms": [],
            "terms": [term],
            "source": "unmapped",
        }
        try:
            uids = self._esearch_mesh(term)
            if not uids:
                return unmapped
            heading, entry_terms = self._efetch_mesh(uids[0])
            if not heading:
                return unmapped
            terms = _dedupe([heading, term, *entry_terms])
            return {
                "input": term,
                "heading": heading,
                "entry_terms": entry_terms,
                "terms": terms,
                "source": "mesh",
            }
        except Exception as e:
            logger.warning("MeSH thesaurus lookup failed for %r: %s", term, e)
            return unmapped

    def _esearch_mesh(self, term: str) -> List[str]:
        params = {"db": "mesh", "term": term, "retmode": "json", "retmax": 1}
        response = self._request("esearch.fcgi", params)
        return response.json().get("esearchresult", {}).get("idlist", [])

    def _efetch_mesh(self, uid: str):
        """
        Fetch one MeSH record. `db=mesh` serves ASCII text (`retmode=text`), not XML.

        The record's shape is::

            1: AMP-Activated Protein Kinases
            <scope note paragraph>

            Entry Terms:
                  AMP Activated Protein Kinases
                  AMP-Activated Kinase

            All MeSH Categories
               Chemicals and Drugs Category
            ...
            Tree Number(s): D08.811...

        Only the indented block under "Entry Terms:" is taken — everything after the
        blank line that closes it is the tree hierarchy, which is *not* a synonym set
        and would otherwise drag the query out to terms like "Enzymes".
        """
        params = {"db": "mesh", "id": uid, "retmode": "text", "rettype": "full"}
        body = self._request("efetch.fcgi", params).text
        heading = None
        entry_terms: List[str] = []
        in_entry_block = False

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if heading is None:
                match = re.match(r"^\d+:\s*(.+)$", line)
                if match:
                    heading = match.group(1).strip()
                continue
            if not in_entry_block:
                if line.lower().startswith("entry term"):
                    in_entry_block = True
                continue
            if not line:
                break  # blank line closes the entry-term block
            if line.endswith(":") or re.match(r"^[A-Za-z ()]+:\s", line):
                break  # a new labelled section started
            entry_terms.append(line)

        entry_terms = [
            term for term in _dedupe(entry_terms)
            if term.lower() != (heading or "").lower() and len(term) <= 80
        ][:MAX_ENTRY_TERMS]
        return heading, entry_terms

    def _request(self, endpoint: str, params: Dict) -> requests.Response:
        """Rate-limited E-utilities GET honouring NCBI's per-second cap."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        query = dict(params)
        query["tool"] = "InnoDD-LitMineX"
        if self.api_key:
            query["api_key"] = self.api_key
        response = self._session.get(
            f"{EUTILS_BASE}/{endpoint}", params=query, timeout=settings.LIT_REQUEST_TIMEOUT
        )
        self._last_request = time.monotonic()
        response.raise_for_status()
        return response

    # ── Legacy LLM-assisted expansion (keyword endpoint) ────────────────────
    @observe(name="mesh_term_lookup")
    def get_mesh_terms(self, keywords: List[str]) -> Dict:
        """Return MeSH expansion dict; uses SQLite cache to avoid redundant LLM calls."""
        langfuse_context.update_current_observation(input={"keywords": keywords})
        cache_key = ",".join(sorted(k.lower() for k in keywords))
        cached = self._get_cached(cache_key)
        if cached:
            logger.info(f"MeSH cache hit for: {keywords}")
            langfuse_context.update_current_observation(output={"cached": True})
            return cached

        result = self._call_llm_for_mesh(keywords)
        self._cache_result(cache_key, result)
        langfuse_context.update_current_observation(output=result)
        return result

    @observe(as_type="generation", name="mesh_llm_generation")
    def _call_llm_for_mesh(self, keywords: List[str]) -> Dict:
        from DRP_Main.app.core.llm import llm_client
        langfuse_context.update_current_observation(model=settings.DATABRICKS_LLM_ENDPOINT)
        prompt = (
            f"Keywords: {', '.join(keywords)}\n"
            "Return JSON with mesh_terms (max 5 terms), related_concepts (max 3), "
            "search_strategies (max 2). Be concise. Respond with JSON only, no prose."
        )
        try:
            response = llm_client.databricks(
                messages=[
                    {"role": "system", "content": "Return brief JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=400,
                return_raw=True,
            )
            result = json.loads(response["choices"][0]["message"]["content"])
            usage = response.get("usage") or {}
            langfuse_context.update_current_observation(
                usage={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                }
            )
            return result
        except Exception as e:
            logger.error(f"MeSH LLM call failed: {e}")
            return self._fallback_mesh(keywords)

    def _fallback_mesh(self, keywords: List[str]) -> Dict:
        return {
            "mesh_terms": keywords[:5],
            "related_concepts": keywords[:3],
            "search_strategies": [],
        }

    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT mesh_terms FROM mesh_cache WHERE query = ? '
            f'AND date(created_date) >= date("now", "-{self.cache_days} days")',
            (cache_key,),
        )
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else None

    def _cache_result(self, cache_key: str, result: Dict) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO mesh_cache (query, mesh_terms, created_date) VALUES (?, ?, ?)",
            (cache_key, json.dumps(result), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()


def _dedupe(values: List[str]) -> List[str]:
    seen, out = set(), []
    for value in values:
        value = (value or "").strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out
