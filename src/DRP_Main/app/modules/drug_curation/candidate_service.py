"""
CurateX Tool 2 (§6) — candidate matching and scoring.

Steps 1-5: search the broader candidate universe → apply Tool 1's exclusion flag
*before* scoring → extract the same three-source field set on the survivors →
score against the profile's baselines and current weights → rank.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation import dailymed_service, ligand_sources
from DRP_Main.app.modules.drug_curation.profile_service import (
    DrugProfile,
    OPEN_TARGETS_AUTHORITATIVE,
)
from DRP_Main.app.modules.drug_curation.scoring_service import (
    CandidateScore,
    score_candidate,
    score_drivers,
)

logger = get_logger(__name__)


@dataclass
class Candidate:
    """One scored repurposing candidate."""

    chembl_id: str
    name: str
    smiles: str = ""
    inchikey: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)
    field_sources: Dict[str, str] = field(default_factory=dict)
    secondary_values: Dict[str, Any] = field(default_factory=dict)
    label: Optional[dailymed_service.LabelExtract] = None
    score: Optional[CandidateScore] = None
    enriched: bool = False

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chemblId": self.chembl_id,
            "name": self.name,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "fields": dict(self.fields),
            "fieldSources": dict(self.field_sources),
            "secondaryValues": dict(self.secondary_values),
            "enriched": self.enriched,
            "labelFound": bool(self.label and self.label.found),
            "labelNote": self.label.note if self.label else "",
        }
        if self.score:
            payload.update(self.score.as_dict())
            payload["scoreDrivers"] = score_drivers(self.score)
        return payload


@dataclass
class CandidateSet:
    candidates: List[Candidate]
    excluded: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self.candidates),
            "candidates": [c.as_dict() for c in self.candidates],
            "excludedCount": len(self.excluded),
            "excluded": self.excluded,
            "metadata": dict(self.metadata),
        }


# ── Step 1: candidate universe ────────────────────────────────────────────────
@observe(name="curatex_candidate_universe")
def search_candidate_universe(
    limit: int = 500, min_phase: int = 1, seed_records: Optional[Sequence[Dict[str, Any]]] = None
) -> List[Candidate]:
    """
    §6 step 1 — ChEMBL molecules with any populated max clinical phase.

    Approved *and* investigational: an investigational compound already carries a
    development data package, which is the head start repurposing exploits, and
    the Regulatory status category already scores maturity, so restricting the
    pool would apply that filter twice.
    """
    records = list(seed_records or ligand_sources.chembl_candidate_universe(limit, min_phase))
    candidates: List[Candidate] = []
    seen: set = set()
    for record in records:
        chembl_id = record.get("chembl_id") or record.get("record_id") or ""
        if not chembl_id or chembl_id in seen:
            continue
        seen.add(chembl_id)
        fields = dict(record.get("fields") or {})
        candidates.append(
            Candidate(
                chembl_id=chembl_id,
                name=record.get("name") or chembl_id,
                smiles=record.get("smiles", "") or "",
                inchikey=record.get("inchikey", "") or "",
                fields=fields,
                field_sources={key: "chembl" for key in fields},
            )
        )
    langfuse_context.update_current_observation(output={"universe": len(candidates)})
    return candidates


# ── Step 2: exclusion filter ──────────────────────────────────────────────────
def apply_exclusion_filter(
    candidates: Sequence[Candidate], exclusion: Dict[str, Any]
) -> tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    §6 step 2 — drop excluded candidates from the pool entirely, before scoring.

    Not scored-then-hidden: a drug already indicated for the disease (or one of
    its EFO descendants) is not a repurposing candidate, so spending extraction
    and scoring on it is wasted work and its score would be misleading if it ever
    leaked into a response.
    """
    if not exclusion.get("active"):
        return list(candidates), []

    excluded_drugs: Dict[str, Any] = exclusion.get("drugs") or {}
    kept: List[Candidate] = []
    removed: List[Dict[str, Any]] = []
    for candidate in candidates:
        hit = excluded_drugs.get(candidate.chembl_id)
        if hit:
            removed.append(
                {
                    "chemblId": candidate.chembl_id,
                    "name": candidate.name,
                    "matchedDiseases": hit.get("matchedDiseaseNames", []),
                    "matchedEfoIds": hit.get("matchedEfoIds", []),
                    "reason": "already indicated for the linked disease or a descendant term",
                }
            )
        else:
            kept.append(candidate)
    return kept, removed


# ── Step 3: field extraction on survivors ─────────────────────────────────────
def _apply_open_targets(candidate: Candidate, detail: Optional[Dict[str, Any]]) -> None:
    if not detail:
        return
    for key, value in (detail.get("fields") or {}).items():
        if value is None or value == "":
            continue
        existing = candidate.field_sources.get(key)
        if existing and existing != "open_targets" and key in OPEN_TARGETS_AUTHORITATIVE:
            candidate.secondary_values[key] = candidate.fields.get(key)
        candidate.fields[key] = value
        candidate.field_sources[key] = "open_targets"


def _apply_label(candidate: Candidate, extract: dailymed_service.LabelExtract) -> None:
    candidate.label = extract
    if not extract.found:
        return
    for key, value in extract.fields.items():
        if key == "black_box_warning" and candidate.field_sources.get(key) == "open_targets":
            candidate.secondary_values.setdefault("black_box_warning_dailymed", value)
            continue
        candidate.fields[key] = value
        candidate.field_sources[key] = "dailymed"


@observe(name="curatex_candidate_extraction")
def enrich_candidates(candidates: Sequence[Candidate], max_workers: int = 8) -> None:
    """§6 step 3 — Open Targets fields + DailyMed labels for the post-filter set."""
    if not candidates:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        details = executor.map(
            ligand_sources.open_targets_drug, [c.chembl_id for c in candidates]
        )
        labels = executor.map(dailymed_service.enrich, [c.name for c in candidates])
        for candidate, detail in zip(candidates, details):
            _apply_open_targets(candidate, detail)
        for candidate, extract in zip(candidates, labels):
            _apply_label(candidate, extract)
            candidate.enriched = True


# ── Steps 4-5: scoring and ranking ────────────────────────────────────────────
@observe(name="curatex_tool2_candidates")
def match_and_score(
    profile: DrugProfile,
    limit: Optional[int] = None,
    top_n: int = 25,
    enrich_top_n: Optional[int] = None,
    min_phase: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
) -> CandidateSet:
    """
    Run Tool 2 end to end against a (possibly user-edited) profile.

    `enrich_top_n` bounds the expensive part. Open Targets and DailyMed are one
    HTTP round trip *per candidate*, so a 500-molecule pool would be 1000 calls.
    The pool is therefore pre-scored on the ChEMBL fields alone — which cover 13
    of the 19 criteria including the two heaviest, max clinical phase and QED —
    and only the leaders are enriched and rescored. The cap is reported in the
    result metadata rather than left implicit, because a candidate that never got
    enriched is not the same as one that scored poorly.
    """
    limit = int(limit or getattr(settings, "CURATEX_UNIVERSE_LIMIT", 500) or 500)
    enrich_top_n = int(enrich_top_n or getattr(settings, "CURATEX_ENRICH_TOP_N", 60) or 60)
    if min_phase is None:
        min_phase = int(getattr(settings, "CURATEX_MIN_CLINICAL_PHASE", 1) or 1)

    active_weights = dict(profile.weights)
    if weights:
        active_weights.update({k: float(v) for k, v in weights.items() if k in active_weights})

    universe = search_candidate_universe(limit=limit, min_phase=min_phase)
    survivors, removed = apply_exclusion_filter(universe, profile.exclusion)

    # Pre-score on ChEMBL-only fields to decide what is worth enriching.
    for candidate in survivors:
        candidate.score = score_candidate(
            candidate.fields, profile.baselines, active_weights, candidate.field_sources
        )
    survivors.sort(key=lambda c: c.score.composite if c.score else 0.0, reverse=True)

    shortlist = survivors[: max(top_n, enrich_top_n)]
    enrich_candidates(shortlist)

    for candidate in shortlist:
        candidate.score = score_candidate(
            candidate.fields, profile.baselines, active_weights, candidate.field_sources
        )

    ranked = sorted(
        survivors, key=lambda c: c.score.composite if c.score else 0.0, reverse=True
    )[:top_n]

    metadata = {
        "universeSize": len(universe),
        "universeNote": (
            f"The candidate pool is the {limit} most clinically mature ChEMBL molecules with "
            f"max phase ≥ {min_phase}, not the whole of ChEMBL. Raise universeLimit to widen it."
        ),
        "excludedCount": len(removed),
        "scoredCount": len(survivors),
        "enrichedCount": sum(1 for c in shortlist if c.enriched),
        "enrichmentCap": len(shortlist),
        "enrichmentNote": (
            f"Open Targets and DailyMed extraction ran on the top {len(shortlist)} of "
            f"{len(survivors)} surviving candidates (ranked on ChEMBL fields first); the "
            "remainder are scored on ChEMBL data only."
        ),
        "weights": active_weights,
        "exclusionActive": bool(profile.exclusion.get("active")),
        "minPhase": min_phase,
    }
    langfuse_context.update_current_observation(output={k: metadata[k] for k in
                                                        ("universeSize", "excludedCount", "enrichedCount")})
    return CandidateSet(candidates=ranked, excluded=removed, metadata=metadata)
