"""
URL validation and compound verification service.
Split from app/api/v1/endpoints/drug_curation.py.
"""
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import quote

from langfuse.decorators import observe, langfuse_context

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class URLValidator:
    BLOCKED_DOMAINS = ["medscape.com", "fda.gov", "accessdata.fda.gov", "reference.medscape.com"]

    @staticmethod
    def is_blocked_url(url: str) -> bool:
        if not url:
            return True
        url_lower = url.lower()
        return any(domain in url_lower for domain in URLValidator.BLOCKED_DOMAINS)

    @staticmethod
    def normalize_drug_name(drug_name: str) -> str:
        normalized = re.sub(r"[^\w\s-]", "", drug_name.lower())
        normalized = re.sub(r"\s+", "-", normalized.strip())
        return re.sub(r"-+", "-", normalized)

    @staticmethod
    def construct_drugs_com_url(drug_name: str) -> str:
        return f"https://www.drugs.com/{URLValidator.normalize_drug_name(drug_name)}.html"

    @staticmethod
    def construct_pubmed_url(pmid: str = None, query: str = None) -> str:
        if pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if query:
            return f"https://pubmed.ncbi.nlm.nih.gov/?term={quote(query)}"
        return "https://pubmed.ncbi.nlm.nih.gov/"

    @staticmethod
    def validate_url_fast(url: str, timeout: int = 3) -> Dict:
        if not url or not url.startswith(("http://", "https://")):
            return {"valid": False, "status_code": None, "final_url": url, "error": "Invalid URL format"}
        if URLValidator.is_blocked_url(url):
            return {"valid": False, "status_code": None, "final_url": url, "error": "Blocked domain"}
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return {
                "valid": response.status_code == 200,
                "status_code": response.status_code,
                "final_url": response.url,
                "error": None,
            }
        except requests.RequestException as e:
            return {"valid": False, "status_code": None, "final_url": url, "error": str(e)}

    @staticmethod
    def fix_common_url_issues(url: str, drug_name: str = None) -> Optional[str]:
        if not url or URLValidator.is_blocked_url(url):
            return None
        url_lower = url.lower()
        if "drugs.com" in url_lower and drug_name:
            return URLValidator.construct_drugs_com_url(drug_name)
        if "pubmed" in url_lower:
            pmid_match = re.search(r"/(\d{8})/?", url)
            if pmid_match:
                return URLValidator.construct_pubmed_url(pmid=pmid_match.group(1))
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        url = re.sub(r"(?<!:)//+", "/", url)
        return url

    @observe(name="url_validation")
    @staticmethod
    def validate_and_fix_urls(urls: List[str], drug_name: str = None, max_workers: int = 5) -> List[str]:
        if not urls:
            return []
        urls_to_validate = []
        for url in urls:
            if not url or URLValidator.is_blocked_url(url):
                continue
            fixed = URLValidator.fix_common_url_issues(url, drug_name)
            if fixed:
                urls_to_validate.append(fixed)
        if not urls_to_validate:
            return []
        fixed_urls = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(URLValidator.validate_url_fast, url): url
                for url in urls_to_validate
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    validation = future.result()
                    if validation["valid"]:
                        fixed_urls.append(validation["final_url"])
                    elif drug_name and "drugs.com" in url.lower():
                        fixed_urls.append(
                            f"https://www.drugs.com/search.php?searchterm={quote(drug_name)}"
                        )
                except Exception as e:
                    logger.debug(f"URL validation exception for {url}: {e}")
        return list(set(fixed_urls))


class VerificationService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

    @observe(name="compound_verification")
    def verify_compounds(self, compounds: List[Dict]) -> Dict:
        langfuse_context.update_current_observation(input={"compound_count": len(compounds)})
        verification_results = []

        for compound in compounds:
            name = compound.get("compound_name", "Unknown Compound")
            websites = compound.get("websites", [])
            if isinstance(websites, str):
                websites = [websites]

            compound_verification = {
                "compound": name,
                "verification_details": [],
                "overall_status": "Unknown",
            }
            valid_verifications = 0
            total_mentions = 0

            for website in websites[:5]:
                if not self._is_valid_url(website):
                    compound_verification["verification_details"].append(
                        {"website": website, "status": "Invalid URL", "mentions": 0}
                    )
                    continue

                page_data = self._fetch_page_content(website)
                if page_data:
                    variations = [
                        name.lower(),
                        name.replace(" ", "-").lower(),
                        name.replace("-", " ").lower(),
                        name.replace(" ", "").lower(),
                    ]
                    total_compound_mentions = sum(
                        page_data.lower().count(v) for v in variations
                    )
                    if total_compound_mentions > 0:
                        valid_verifications += 1
                        total_mentions += total_compound_mentions
                    compound_verification["verification_details"].append(
                        {
                            "website": website,
                            "status": (
                                f"Found {total_compound_mentions} mentions"
                                if total_compound_mentions > 0
                                else "Not Found"
                            ),
                            "mentions": total_compound_mentions,
                            "content_length": len(page_data),
                        }
                    )
                else:
                    compound_verification["verification_details"].append(
                        {"website": website, "status": "Failed to retrieve content", "mentions": 0}
                    )
                time.sleep(3)

            valid_urls = sum(1 for w in websites if self._is_valid_url(w))
            if valid_verifications > 0 and valid_urls > 0:
                ratio = valid_verifications / valid_urls
                compound_verification["overall_status"] = (
                    "Verified" if ratio >= 0.5 else "Partially Verified"
                )
            else:
                compound_verification["overall_status"] = "Not Verified"

            compound_verification["summary"] = {
                "valid_sources": valid_verifications,
                "total_sources": len(websites),
                "total_mentions": total_mentions,
            }
            verification_results.append(compound_verification)

        result = {
            "verification_results": verification_results,
            "summary": {
                "total_compounds": len(compounds),
                "verified_compounds": sum(
                    1 for r in verification_results if r["overall_status"] == "Verified"
                ),
                "partially_verified": sum(
                    1 for r in verification_results if r["overall_status"] == "Partially Verified"
                ),
                "not_verified": sum(
                    1 for r in verification_results if r["overall_status"] == "Not Verified"
                ),
            },
        }
        langfuse_context.update_current_observation(output=result["summary"])
        return result

    def _is_valid_url(self, url: str) -> bool:
        return isinstance(url, str) and bool(re.match(r"^https?://", url))

    def _fetch_page_content(self, url: str, retries: int = 3, delay: int = 5) -> Optional[str]:
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.content, "html.parser")
                        return soup.get_text(separator="\n", strip=True)
                    except ImportError:
                        return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                time.sleep(delay)
        return None
