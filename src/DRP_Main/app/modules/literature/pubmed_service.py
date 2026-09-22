"""
PubMed search and article fetch service for Literature Mining.
Split from app/api/v1/endpoints/literature_mining.py.
"""
import os
import re

import requests
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

from langfuse.decorators import observe, langfuse_context

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT = 15
MAX_ABSTRACT_LENGTH = 500
# Terms per concept group in a LitMineX query — see build_litminex_query().
MAX_TERMS_PER_GROUP = 12


class PubMedService:
    def __init__(self, timeout: int = REQUEST_TIMEOUT):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

    @observe(name="pubmed_search")
    def search(self, query: str, max_results: int) -> List[Dict]:
        """Search PubMed and return a list of article dicts."""
        langfuse_context.update_current_observation(
            input={"query": query, "max_results": max_results}
        )
        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            }
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            ids = response.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                langfuse_context.update_current_observation(output={"articles_found": 0})
                return []
            articles = self.fetch_details(ids)
            langfuse_context.update_current_observation(output={"articles_found": len(articles)})
            return articles
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    @observe(name="pubmed_fetch_details")
    def fetch_details(self, ids: List[str]) -> List[Dict]:
        """Fetch article details for a list of PubMed IDs."""
        langfuse_context.update_current_observation(input={"id_count": len(ids)})
        if not ids:
            return []
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            articles = [
                self._parse_article(a)
                for a in root.findall(".//PubmedArticle")
                if self._parse_article(a) is not None
            ]
            langfuse_context.update_current_observation(output={"parsed": len(articles)})
            return articles
        except Exception as e:
            logger.error(f"PubMed fetch details error: {e}")
            return []

    def _parse_article(self, xml) -> Optional[Dict]:
        try:
            pmid_elem = xml.find(".//PMID")
            if pmid_elem is None:
                return None
            title_elem = xml.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            abstract_parts = [
                t.text for t in xml.findall(".//AbstractText") if t.text
            ]
            abstract = " ".join(abstract_parts)
            pmc_elem = xml.find('.//ArticleId[@IdType="pmc"]')
            doi_elem = xml.find('.//ArticleId[@IdType="doi"]')
            return {
                "pmid": pmid_elem.text,
                "pmcid": pmc_elem.text if pmc_elem is not None else None,
                "doi": doi_elem.text if doi_elem is not None else None,
                "title": title,
                "abstract": abstract,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/",
                "year": self._parse_year(xml),
                "pub_date": self._parse_pub_date(xml),
                "authors": self._parse_authors(xml),
                "authors_list": self._parse_author_list(xml),
                "journal": self._parse_text(xml, ".//Journal/Title"),
                "keywords": self._parse_keywords(xml),
            }
        except Exception as e:
            logger.debug(f"Article parse error: {e}")
            return None

    # ── Citation metadata helpers ─────────────────────────────────────────────
    @staticmethod
    def _parse_text(xml, path: str) -> str:
        elem = xml.find(path)
        return (elem.text or "").strip() if elem is not None else ""

    def _parse_year(self, xml) -> Optional[int]:
        """First 4-digit year found in the publication date elements."""
        for path in (
            ".//JournalIssue/PubDate/Year",
            ".//JournalIssue/PubDate/MedlineDate",
            ".//PubMedPubDate/Year",
        ):
            raw = self._parse_text(xml, path)
            match = re.search(r"\d{4}", raw)
            if match:
                return int(match.group())
        return None

    _MONTHS = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    def _parse_pub_date(self, xml) -> Optional[str]:
        """ISO-8601 publication date; falls back to the parts PubMed actually carries."""
        for path in (".//JournalIssue/PubDate", ".//ArticleDate", ".//PubMedPubDate"):
            node = xml.find(path)
            if node is None:
                continue
            year = self._parse_text(node, "Year")
            if not year:
                medline = self._parse_text(node, "MedlineDate")
                match = re.search(r"\d{4}", medline)
                if not match:
                    continue
                return f"{match.group()}-01-01"
            raw_month = self._parse_text(node, "Month") or "1"
            month = self._MONTHS.get(raw_month[:3].lower(), 0) or _safe_int(raw_month, 1)
            day = _safe_int(self._parse_text(node, "Day"), 1)
            return f"{int(year):04d}-{min(max(month, 1), 12):02d}-{min(max(day, 1), 31):02d}"
        return None

    @staticmethod
    def _parse_author_list(xml) -> List[str]:
        """Structured author names — the spec's `authors` array."""
        names = []
        for author in xml.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            collective = author.find("CollectiveName")
            if last is not None and last.text:
                names.append(
                    f"{fore.text} {last.text}".strip()
                    if fore is not None and fore.text
                    else last.text.strip()
                )
            elif collective is not None and collective.text:
                names.append(collective.text.strip())
        return names

    @staticmethod
    def _parse_authors(xml) -> str:
        """Render as 'Chen, S. et al.' — the display form the frontend expects."""
        names = []
        for author in xml.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            initials = author.find("Initials")
            if last is None or not last.text:
                continue
            initial_part = f", {initials.text[0]}." if initials is not None and initials.text else ""
            names.append(f"{last.text}{initial_part}")
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return " and ".join(names)
        return f"{names[0]} et al."

    @staticmethod
    def _parse_keywords(xml) -> List[str]:
        keywords = [k.text.strip() for k in xml.findall(".//KeywordList/Keyword") if k.text]
        mesh = [
            m.text.strip()
            for m in xml.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if m.text
        ]
        seen, out = set(), []
        for term in keywords + mesh:
            if term.lower() not in seen:
                seen.add(term.lower())
                out.append(term)
        return out[:20]

    def build_search_query(self, keywords: List[str], mesh_data: Dict) -> str:
        """Build optimised PubMed query combining keywords and MeSH terms."""
        keyword_query = " OR ".join(
            [f'"{k}"[Title/Abstract]' for k in keywords[:5]]
        )
        mesh_terms = mesh_data.get("mesh_terms", [])[:3]
        query_parts = [f"({keyword_query})"]
        if mesh_terms and mesh_terms != keywords:
            mesh_query = " OR ".join([f'"{t}"[MeSH Terms]' for t in mesh_terms])
            query_parts.append(f"({mesh_query})")
        base_query = " OR ".join(query_parts)
        open_access = '("loattrfree full text"[sb] OR "pmc"[Filter])'
        return f"({base_query}) AND {open_access}"

    # ── LitMineX Tool 1 (spec §3 steps 3–4) ─────────────────────────────────
    def build_litminex_query(
        self,
        target_terms: List[str],
        disease_terms: List[str],
        exclude_reviews: bool = False,
        max_terms: int = MAX_TERMS_PER_GROUP,
    ) -> str:
        """
        Build the spec's PubMed query: every expanded term is field-tagged into both
        ``[Title/Abstract]`` and ``[MeSH Terms]``, OR'd within its concept group, and
        the target group is **AND**ed with the disease group so both concepts must
        appear rather than either alone.

        `exclude_reviews` adds the optional publication-type filter for users who
        want primary evidence only.

        Each group is capped at `max_terms`. MeSH entry-term lists reach 30+ synonyms
        per concept, and the resulting query overruns E-utilities' URI limit; the
        heading and the caller's own term lead the list, so the cap drops only the
        long tail of subunit and inversion variants.
        """
        target_group = self._field_group(target_terms, max_terms)
        disease_group = self._field_group(disease_terms, max_terms)
        if not target_group and not disease_group:
            raise ValueError("LitMineX query needs at least one target or disease term")
        if not target_group:
            query = disease_group
        elif not disease_group:
            query = target_group
        else:
            query = f"({target_group}) AND ({disease_group})"
        if exclude_reviews:
            query = f"({query}) NOT (review[Publication Type])"
        return query

    @staticmethod
    def _field_group(terms: List[str], max_terms: int = MAX_TERMS_PER_GROUP) -> str:
        clauses: List[str] = []
        seen: set = set()
        for term in terms:
            term = (term or "").strip().replace('"', "")
            # Scope-note sentences occasionally leak out of a MeSH record; they are
            # not search terms and would only add noise.
            if not term or term.lower() in seen or len(term) > 60 or term.count(" ") > 6:
                continue
            seen.add(term.lower())
            clauses.append(f'"{term}"[Title/Abstract]')
            clauses.append(f'"{term}"[MeSH Terms]')
            if len(seen) >= max_terms:
                break
        return " OR ".join(clauses)

    @observe(name="pubmed_esearch_pmids")
    def esearch_pmids(self, query: str, retmax: int) -> List[str]:
        """Submit the constructed query and return matching PMIDs, relevance-sorted."""
        langfuse_context.update_current_observation(input={"query": query, "retmax": retmax})
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
            "tool": "InnoDD-LitMineX",
        }
        api_key = os.environ.get("NCBI_API_KEY", "")
        if api_key:
            params["api_key"] = api_key
        try:
            # POST, not GET: a fully MeSH-expanded two-concept query runs to several
            # kilobytes and E-utilities answers a GET of that size with 414.
            response = self.session.post(url, data=params, timeout=self.timeout)
            response.raise_for_status()
            ids = response.json().get("esearchresult", {}).get("idlist", [])
            langfuse_context.update_current_observation(output={"pmids": len(ids)})
            return ids
        except Exception as e:
            logger.error("PubMed esearch failed: %s", e)
            return []

    @observe(name="pubmed_fetch_records")
    def fetch_records(self, pmids: List[str], batch_size: int = 50) -> List[Dict]:
        """
        Batch-fetch full records via efetch and parse out the fields Tool 2 needs:
        title, abstract, authors, journal, publication date, PMID, PMCID/DOI and the
        full-text link when the article is in the PMC open-access subset.
        """
        records: List[Dict] = []
        for start in range(0, len(pmids), max(1, batch_size)):
            for raw in self.fetch_details(pmids[start : start + batch_size]):
                records.append(self._to_litminex_record(raw))
        langfuse_context.update_current_observation(output={"records": len(records)})
        return records

    def _to_litminex_record(self, article: Dict) -> Dict:
        pmcid = article.get("pmcid")
        doi = article.get("doi")
        if pmcid:
            pdf_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        elif doi:
            pdf_link = f"https://doi.org/{doi}"
        else:
            pdf_link = article.get("pubmed_url", "")
        return {
            "pmid": article.get("pmid", ""),
            "pmcid": pmcid,
            "doi": doi,
            "title": article.get("title", ""),
            "abstract": article.get("abstract", ""),
            "authors": article.get("authors_list") or _author_list(article.get("authors", "")),
            "authors_display": article.get("authors", ""),
            "journal": article.get("journal", ""),
            "year": article.get("year"),
            "pub_date": article.get("pub_date") or _year_as_date(article.get("year")),
            "pdf_link": pdf_link,
            "pubmed_url": article.get("pubmed_url", ""),
            "keywords": article.get("keywords", []),
            # Populated by fetch_fulltext_sections() when the article is PMC OA.
            "sections": {},
        }

    @observe(name="pmc_fulltext_sections")
    def fetch_fulltext_sections(self, pmcid: str) -> Dict[str, str]:
        """
        Pull the PMC open-access full text and bucket it into the sections the
        positional-weighting table names. Returns ``{}` when the article is not in
        the OA subset — Tool 2 then scores off title and abstract alone, at reduced
        positional-weighting resolution (spec §3, closing note).
        """
        if not pmcid:
            return {}
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pmc", "id": pmcid.replace("PMC", ""), "retmode": "xml"}
        api_key = os.environ.get("NCBI_API_KEY", "")
        if api_key:
            params["api_key"] = api_key
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception as e:
            logger.debug("PMC full-text fetch failed for %s: %s", pmcid, e)
            return {}

        sections: Dict[str, List[str]] = {}
        for sec in root.findall(".//body//sec"):
            title_elem = sec.find("title")
            label = _canonical_section((title_elem.text or "") if title_elem is not None else "")
            text = " ".join(
                "".join(p.itertext()).strip() for p in sec.findall(".//p")
            ).strip()
            if text:
                sections.setdefault(label, []).append(text)

        refs = " ".join(
            "".join(ref.itertext()).strip() for ref in root.findall(".//ref-list//ref")
        ).strip()
        if refs:
            sections.setdefault("references", []).append(refs)
        return {name: " ".join(parts) for name, parts in sections.items() if parts}


# ── module helpers ───────────────────────────────────────────────────────────
_SECTION_ALIASES = (
    ("result", "results"),
    ("discussion", "discussion"),
    ("conclusion", "conclusion"),
    ("introduction", "introduction"),
    ("background", "introduction"),
    ("method", "methods"),
    ("material", "methods"),
    ("reference", "references"),
    ("abstract", "abstract"),
)


def _canonical_section(title: str) -> str:
    lowered = (title or "").strip().lower()
    for needle, canonical in _SECTION_ALIASES:
        if needle in lowered:
            return canonical
    return lowered or "body"


def _author_list(authors: str) -> List[str]:
    """`_parse_authors` renders a display string; the spec's shape is a list."""
    if not authors:
        return []
    text = authors.replace(" et al.", "").replace(" and ", ", ")
    return [part.strip() for part in text.split(",") if part.strip() and part.strip() != "."]


def _year_as_date(year) -> Optional[str]:
    return f"{year}-01-01" if year else None


def _safe_int(value: str, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default
