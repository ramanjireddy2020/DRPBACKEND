"""
Drug curation service — Groq-powered compound generation + PubMed enhancement.
Split from app/services/drug_curation_services.py.
"""
import json
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List

from fastapi import HTTPException
from langfuse.decorators import observe, langfuse_context

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.pubmed_service import PubMedService

logger = get_logger(__name__)

OUTPUT_DIR = "data/drug_curation"
MIN_RESULTS = 15
MAX_RESULTS = 50
PUBMED_MAX_RESULTS = 3


class DrugCurationService:
    def __init__(self):
        self.pubmed_service = PubMedService()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    @observe(name="drug_curation")
    def curate_compounds(self, criteria: str, num_results: int = 20, skip_pubmed: bool = False) -> Dict:
        """Curate drug compounds via LLM, optionally enhanced with PubMed articles."""
        langfuse_context.update_current_observation(
            input={"criteria": criteria[:100], "num_results": num_results, "skip_pubmed": skip_pubmed}
        )
        try:
            num_results = max(MIN_RESULTS, min(num_results, MAX_RESULTS))
            prompt = self._create_prompt(criteria, num_results)
            result_dict = self._retrieve_compounds_with_groq(prompt)

            if "error" in result_dict:
                if "Invalid JSON" in result_dict.get("error", ""):
                    retry_count = max(MIN_RESULTS, num_results // 2)
                    result_dict = self._retrieve_compounds_with_groq(
                        self._create_prompt(criteria, retry_count)
                    )
                    if "error" in result_dict:
                        raise HTTPException(status_code=500, detail=result_dict["error"])
                else:
                    raise HTTPException(status_code=500, detail=result_dict["error"])

            compounds = result_dict.get("compounds", [])
            try:
                compounds.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
                confidence_scores = [c.get("confidence_score", 0) for c in compounds]
                max_score = max(confidence_scores) if confidence_scores else 0
                min_score = min(confidence_scores) if confidence_scores else 0
                avg_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            except Exception as e:
                logger.warning("Could not sort compounds: %s", e)
                max_score = min_score = avg_score = 0

            enhanced_compounds = (
                compounds if skip_pubmed
                else self._enhance_compounds_with_pubmed(compounds, criteria)
            )
            try:
                enhanced_compounds.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
            except Exception:
                pass

            final_result = {
                "compounds": enhanced_compounds,
                "total_compounds": len(enhanced_compounds),
                "criteria": criteria,
                "search_metadata": {
                    "requested_count": num_results,
                    "delivered_count": len(enhanced_compounds),
                    "api_used": "Groq API",
                    "pubmed_enhanced": not skip_pubmed,
                    "sorted_by_confidence": True,
                    "confidence_score_range": {
                        "min": min_score,
                        "max": max_score,
                        "average": round(avg_score, 1),
                    },
                    "timestamp": datetime.now().isoformat(),
                },
            }
            try:
                self._save_to_json(final_result)
            except Exception as e:
                logger.warning("Failed to save JSON (non-critical): %s", e)

            langfuse_context.update_current_observation(
                output={"delivered_count": len(enhanced_compounds)}
            )
            return final_result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Drug curation failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Curation failed: {str(e)}")

    def _create_prompt(self, criteria: str, num_results: int) -> str:
        return f"""You are a pharmaceutical expert. Return EXACTLY {num_results} drug compounds matching these criteria in JSON format.

CRITERIA: {criteria}

Return JSON with this EXACT structure (be concise in descriptions):
{{
    "compounds": [
        {{
            "compound_name": "Drug name",
            "content": {{
                "drug_profile": "Brief profile (max 200 chars)",
                "mechanism_of_action": "Brief MOA (max 150 chars)",
                "criteria_match": {{"key1": "match explanation"}},
                "clinical_data": "Key data (max 150 chars)",
                "side_effects": "Main side effects (max 100 chars)",
                "dosage_forms": "Forms available"
            }},
            "confidence_score": 95,
            "exact_criteria_match": {{
                "total_criteria": 3,
                "matched_criteria": 3,
                "match_percentage": 100,
                "unmatched_criteria": []
            }},
            "sources": {{
                "pubmed_query": "search terms",
                "additional_databases": ["Drugs.com"]
            }}
        }}
    ]
}}

RULES:
1. EXACTLY {num_results} compounds
2. Keep descriptions concise
3. Valid JSON only
4. FDA-approved drugs preferred
5. Assign realistic confidence scores (60-100)
6. Best matches: 90-100, Good: 75-89, Acceptable: 60-74"""

    @observe(as_type="generation", name="compound_generation")
    def _retrieve_compounds_with_groq(self, prompt: str) -> dict:
        from DRP_Main.app.core.llm import llm_client
        langfuse_context.update_current_observation(model=settings.DATABRICKS_LLM_ENDPOINT)
        try:
            chat_completion = llm_client.databricks(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a pharmaceutical research expert. Always respond with valid JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=8000,
                return_raw=True,
            )
            content = chat_completion["choices"][0]["message"]["content"]
            finish_reason = chat_completion["choices"][0].get("finish_reason")
            if finish_reason == "length":
                content = self._repair_truncated_json(content)
            usage = chat_completion.get("usage") or {}
            langfuse_context.update_current_observation(
                usage={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                }
            )
            return json.loads(content)
        except json.JSONDecodeError as e:
            try:
                repaired = self._repair_truncated_json(content)
                return json.loads(repaired)
            except Exception:
                return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            logger.error("Groq API error: %s", e, exc_info=True)
            return {"error": f"Error communicating with Groq API: {str(e)}"}

    def _repair_truncated_json(self, content: str) -> str:
        content = content.rstrip()
        open_braces = content.count("{")
        close_braces = content.count("}")
        open_brackets = content.count("[")
        close_brackets = content.count("]")
        if open_brackets > close_brackets:
            content += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            content += "}" * (open_braces - close_braces)
        last_complete = content.rfind("},")
        if last_complete > 0:
            content = content[: last_complete + 1] + "]}"
        return content

    @observe(name="pubmed_enhancement")
    def _enhance_compounds_with_pubmed(self, compounds: List[Dict], criteria: str) -> List[Dict]:
        langfuse_context.update_current_observation(input={"compound_count": len(compounds)})

        def enhance_single(compound_with_index):
            index, compound = compound_with_index
            try:
                compound_name = compound.get("compound_name", "")
                pubmed_query = compound.get("sources", {}).get("pubmed_query", compound_name)
                if not pubmed_query:
                    keywords = self._extract_key_criteria(criteria)
                    pubmed_query = f"{compound_name} AND ({' OR '.join(keywords)})"
                pubmed_articles = self.pubmed_service.search_pubmed(pubmed_query, max_results=PUBMED_MAX_RESULTS)
                additional_sources = self._search_additional_sources(compound_name, criteria)
                enhanced = compound.copy()
                enhanced["websites"] = [a["url"] for a in pubmed_articles] + additional_sources
                enhanced["pubmed_articles"] = pubmed_articles
                enhanced["pubmed_query_used"] = pubmed_query
                return index, enhanced
            except Exception as e:
                logger.error("Error enhancing compound %s: %s", compound.get("compound_name"), e)
                return index, compound

        enhanced_compounds = [None] * len(compounds)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(enhance_single, (i, compound))
                for i, compound in enumerate(compounds)
            ]
            for future in futures:
                try:
                    index, result = future.result()
                    enhanced_compounds[index] = result
                except Exception as e:
                    logger.error("Failed to enhance compound: %s", e)

        result = [c for c in enhanced_compounds if c is not None]
        langfuse_context.update_current_observation(output={"enhanced_count": len(result)})
        return result

    def _extract_key_criteria(self, criteria: str) -> List[str]:
        medical_terms = [
            "efficacy", "safety", "toxicity", "side effects", "mechanism",
            "clinical trial", "FDA approved", "treatment", "therapy",
        ]
        found = [t for t in medical_terms if t in criteria.lower()]
        words = [
            w.lower()
            for w in criteria.split()
            if len(w) > 4 and w.lower() not in {"with", "that", "have", "been", "this", "they"}
        ]
        return list(set(found + words))[:5]

    def _search_additional_sources(self, compound_name: str, criteria: str) -> List[str]:
        encoded = urllib.parse.quote(compound_name)
        return [
            f"https://www.drugs.com/search.php?searchterm={encoded}",
            f"https://www.rxlist.com/script/main/srchcont_rxlist.asp?src={encoded}",
            f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={encoded}",
        ]

    def _save_to_json(self, data: Dict, filename: str = "drug_curation_output.json") -> None:
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
