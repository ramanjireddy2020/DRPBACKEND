"""
CurateX Tool 3 (§7) — evidence retrieval and validation.

Per ranked candidate: pull literature scoped to the drug + target/disease
combination, check that the retrieved articles actually reference the drug and
the target/disease *together* rather than co-occurring by accident, reuse the
regulatory/clinical status already fetched in Tool 2, and flag high-scoring
candidates with no direct literature support as `score-only`.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.candidate_service import Candidate
from DRP_Main.app.modules.drug_curation.pubmed_service import PubMedService
from DRP_Main.app.modules.drug_curation.scoring_service import score_drivers

logger = get_logger(__name__)

EVIDENCE_BACKED = "evidence-backed"
SCORE_ONLY = "score-only"

CHEMBL_DRUG_URL = "https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"
OPEN_TARGETS_DRUG_URL = "https://platform.opentargets.org/drug/{chembl_id}"
DAILYMED_SPL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"


@dataclass
class CandidateEvidence:
    chembl_id: str
    name: str
    articles: List[Dict[str, Any]] = field(default_factory=list)
    validated_articles: List[Dict[str, Any]] = field(default_factory=list)
    status_links: List[Dict[str, str]] = field(default_factory=list)
    strength: str = SCORE_ONLY
    query_used: str = ""
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chemblId": self.chembl_id,
            "name": self.name,
            "evidenceStrength": self.strength,
            "queryUsed": self.query_used,
            "validatedCount": len(self.validated_articles),
            "retrievedCount": len(self.articles),
            "articles": self.validated_articles or self.articles,
            "statusLinks": self.status_links,
            "note": self.note,
        }


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]


def _mentions(haystack: str, term: str) -> bool:
    """Whole-word-ish containment; a drug named 'ACE' must not match 'space'."""
    if not term:
        return False
    return re.search(rf"\b{re.escape(term.lower())}", (haystack or "").lower()) is not None


def _context_terms(target: Optional[str], disease: Optional[str]) -> List[str]:
    """
    The target/disease terms an article must also mention to count as validation.

    Multi-word names are split, because a title rarely repeats the full
    'Diabetes Mellitus, Type 2' string that EFO or MeSH stores.
    """
    terms: List[str] = []
    for phrase in (target, disease):
        if not phrase:
            continue
        terms.append(phrase)
        terms.extend(t for t in _tokens(phrase) if len(t) > 3)
    return list(dict.fromkeys(terms))


def build_query(drug_name: str, target: Optional[str], disease: Optional[str]) -> str:
    """§7 step 1 — literature scoped to the drug plus the target/disease combination."""
    parts = [f'"{drug_name}"']
    context = [p for p in (target, disease) if p]
    if context:
        parts.append("(" + " OR ".join(f'"{c}"' for c in context) + ")")
    return " AND ".join(parts)


def _status_links(candidate: Candidate) -> List[Dict[str, str]]:
    """
    §7 step 1 — regulatory/clinical status, reused from Tool 2 rather than re-queried.

    Every link is paired with the value it backs, so a score stays traceable to
    the record it came from.
    """
    links: List[Dict[str, str]] = [
        {
            "label": "ChEMBL compound record",
            "url": CHEMBL_DRUG_URL.format(chembl_id=candidate.chembl_id),
            "supports": "physicochemical, drug-likeness and regulatory fields",
        },
        {
            "label": "Open Targets drug profile",
            "url": OPEN_TARGETS_DRUG_URL.format(chembl_id=candidate.chembl_id),
            "supports": "clinical phase, withdrawal, black-box and adverse-event signal",
        },
    ]
    if candidate.label and candidate.label.found and candidate.label.setid:
        links.append(
            {
                "label": f"DailyMed label — {candidate.label.title or candidate.name}",
                "url": DAILYMED_SPL_URL.format(setid=candidate.label.setid),
                "supports": "dosage form, dosing frequency, absorption, half-life, metabolism",
            }
        )
    return links


@observe(name="curatex_evidence_candidate")
def gather_for_candidate(
    candidate: Candidate,
    target: Optional[str],
    disease: Optional[str],
    service: PubMedService,
    max_results: int = 5,
) -> CandidateEvidence:
    """Steps 1-2 for one candidate: retrieve, then validate co-reference."""
    query = build_query(candidate.name, target, disease)
    articles = service.search_pubmed(query, max_results=max_results) or []

    context = _context_terms(target, disease)
    validated: List[Dict[str, Any]] = []
    for article in articles:
        haystack = f"{article.get('title', '')} {article.get('abstract', '')}"
        drug_hit = _mentions(haystack, candidate.name)
        context_hit = any(_mentions(haystack, term) for term in context) if context else False
        if drug_hit and (context_hit or not context):
            validated.append({**article, "validated": True})

    evidence = CandidateEvidence(
        chembl_id=candidate.chembl_id,
        name=candidate.name,
        articles=articles,
        validated_articles=validated,
        status_links=_status_links(candidate),
        query_used=query,
    )
    if validated:
        evidence.strength = EVIDENCE_BACKED
    else:
        evidence.strength = SCORE_ONLY
        evidence.note = (
            "No retrieved article references this drug together with the target/disease; "
            "the score rests on property and regulatory data alone."
            if articles
            else "No literature retrieved for this drug in this target/disease context."
        )
    return evidence


@observe(name="curatex_tool3_evidence")
def gather_evidence(
    candidates: Sequence[Candidate],
    target: Optional[str] = None,
    disease: Optional[str] = None,
    max_results: int = 5,
    max_workers: int = 3,
) -> Dict[str, CandidateEvidence]:
    """
    §7 steps 1-2 across the ranked candidate list.

    Concurrency is deliberately low: E-utilities allows 3 requests/second without
    an API key, and each candidate costs an esearch *and* an efetch. Running this
    wider returns 429s, which would silently degrade every candidate to
    `score-only` — a data-collection failure dressed up as a finding.
    """
    if not candidates:
        return {}
    service = PubMedService()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda c: gather_for_candidate(c, target, disease, service, max_results),
                candidates,
            )
        )
    by_id = {evidence.chembl_id: evidence for evidence in results}
    langfuse_context.update_current_observation(
        output={
            "candidates": len(results),
            "evidenceBacked": sum(1 for e in results if e.strength == EVIDENCE_BACKED),
        }
    )
    return by_id


# ── Step 3: recommendation ────────────────────────────────────────────────────
def build_recommendation(
    candidates: Sequence[Candidate],
    evidence: Dict[str, CandidateEvidence],
    target: Optional[str],
    disease: Optional[str],
    exclusion: Optional[Dict[str, Any]] = None,
    top_n: int = 3,
) -> str:
    """
    §7 step 3 — a short synthesis naming the top candidates, their score drivers
    and their evidence strength, with the score-only candidates called out
    separately rather than mixed in with the evidence-backed ones.
    """
    if not candidates:
        return (
            "No candidate survived the exclusion filter and scoring. Widen the candidate "
            "universe (lower the minimum clinical phase) or relax the criteria weights."
        )

    scope = target or "the selected target"
    if disease:
        scope = f"{scope} in {disease}"

    backed = [c for c in candidates if evidence.get(c.chembl_id, None) and
              evidence[c.chembl_id].strength == EVIDENCE_BACKED]
    only = [c for c in candidates if c not in backed]

    lines: List[str] = []
    leaders = (backed or candidates)[:top_n]
    headline = ", ".join(
        f"{c.name} ({c.score.composite:.0f}/100)" for c in leaders if c.score
    )
    lines.append(f"Top repurposing candidates for {scope}: {headline}.")

    for candidate in leaders:
        if not candidate.score:
            continue
        drivers = score_drivers(candidate.score)
        support = evidence.get(candidate.chembl_id)
        support_text = (
            f"{len(support.validated_articles)} validated publication(s)"
            if support and support.validated_articles
            else "no validated publication"
        )
        completeness = candidate.score.data_completeness
        lines.append(
            f"- {candidate.name} scored {candidate.score.composite:.1f} on "
            f"{completeness * 100:.0f}% of the active criteria weight, driven by "
            f"{'; '.join(drivers) if drivers else 'no positively scoring criterion'} — "
            f"{support_text}."
        )

    if backed and only:
        names = ", ".join(c.name for c in only[:top_n])
        lines.append(
            f"Score-only (ranked on data, no direct literature tying the drug to "
            f"{scope}): {names}. Treat these as hypotheses to check before prioritising."
        )
    elif not backed:
        lines.append(
            "Every candidate in this list is score-only — none has literature linking it "
            f"to {scope}. The ranking reflects property, safety and regulatory data only."
        )

    if exclusion and exclusion.get("active") and exclusion.get("count"):
        lines.append(
            f"{exclusion['count']} drug(s) already indicated for the disease (or a descendant "
            "term) were removed before scoring and are not eligible for repurposing here."
        )
    return "\n".join(lines)
