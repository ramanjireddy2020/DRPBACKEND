"""
CurateX — the agent chain (§1, §5-§8).

Tool 1 builds an editable weighted profile from the target's known ligands, Tool
2 matches and scores a broader candidate universe against the submitted profile,
Tool 3 gathers and validates evidence. The output is a table plus text: a ranked
candidate table with per-criterion breakdown, evidence links per candidate and a
recommendation string. No visual artifact.

Invocation (§1): CurateX runs when a target and/or disease is available, either
carried in from target identification or entered fresh. Given a disease with no
target, it resolves the disease identifiers and hands the upstream target
shortlist back for confirmation — it does not select a target itself. Target
scoring belongs to TxKG, and the choice of target changes every downstream
profile and score, so it stays a human decision.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langfuse.decorators import observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation import (
    candidate_service,
    evidence_service,
    profile_service,
)
from DRP_Main.app.modules.drug_curation.criteria_config import (
    DAILYMED_CRITERIA,
    normalize_weight_table,
    validate_weight_table,
)
from DRP_Main.app.modules.drug_curation.identifier_service import (
    ResolutionError,
    resolve_disease,
)
from DRP_Main.app.modules.drug_curation.profile_service import DrugProfile

logger = get_logger(__name__)

AWAITING_TARGET = "awaiting_target_confirmation"
PROFILE_READY = "profile_ready"
COMPLETE = "complete"


class CurateXError(RuntimeError):
    """A CurateX run failed for a reportable reason."""


@dataclass
class SessionState:
    """What a session needs to answer a follow-up without rerunning Tool 1."""

    session_id: str
    profiles: Dict[str, DrugProfile] = field(default_factory=dict)
    last_result: Optional[Dict[str, Any]] = None
    stage: str = ""


_SESSIONS: Dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str) -> Optional[SessionState]:
    with _sessions_lock:
        return _SESSIONS.get(session_id)


def _store_session(state: SessionState) -> None:
    with _sessions_lock:
        _SESSIONS[state.session_id] = state


def clear_sessions() -> None:
    with _sessions_lock:
        _SESSIONS.clear()


class CurateXService:
    """The three-tool chain plus the §1 invocation rules."""

    # ── Tool 1 ────────────────────────────────────────────────────────────────
    @observe(name="curatex_build_profile")
    def build_profile(
        self,
        target: Optional[str] = None,
        disease: Optional[str] = None,
        targets: Optional[Sequence[str]] = None,
        candidate_targets: Optional[Sequence[Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, Any]] = None,
        skip_dailymed: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        §5 — resolve inputs and build one profile object per target.

        With a disease but no target, this returns `stage: awaiting_target_confirmation`
        and the upstream shortlist instead of guessing a target.
        """
        payload, _profiles = self._build_profiles(
            target=target,
            disease=disease,
            targets=targets,
            candidate_targets=candidate_targets,
            weights=weights,
            values=values,
            skip_dailymed=skip_dailymed,
            session_id=session_id,
        )
        return payload

    def _build_profiles(
        self,
        target: Optional[str] = None,
        disease: Optional[str] = None,
        targets: Optional[Sequence[str]] = None,
        candidate_targets: Optional[Sequence[Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, Any]] = None,
        skip_dailymed: bool = False,
        session_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], List[DrugProfile]]:
        """Tool 1, returning the wire payload *and* the live profile objects.

        `run()` needs the objects to feed Tool 2 without a session round trip,
        which is why this is split out rather than read back from `_SESSIONS`.
        """
        target_names = [t for t in (list(targets or []) + ([target] if target else [])) if t]
        target_names = list(dict.fromkeys(t.strip() for t in target_names if t.strip()))

        if not target_names:
            return self._await_target_confirmation(disease, candidate_targets, session_id), []

        if weights:

            # The profile screen edits rows rendered from `label`, so it sends

            # labels back rather than keys. Re-key before validating, and before

            # anything downstream reads the table.

            weights = normalize_weight_table(weights)

            problems = validate_weight_table(weights)
            if problems:
                raise CurateXError("; ".join(problems))

        try:
            profiles = profile_service.build_profiles(
                target_names, disease, weights, skip_dailymed, values
            )
        except ResolutionError as exc:
            raise CurateXError(str(exc)) from exc

        state = SessionState(session_id=session_id or "", stage=PROFILE_READY)
        state.profiles = {
            (p.target.gene_symbol or p.target.query if p.target else str(index)): p
            for index, p in enumerate(profiles)
        }
        if session_id:
            _store_session(state)

        return {
            "module": "CurateX",
            "stage": PROFILE_READY,
            "profiles": [p.as_dict() for p in profiles],
            "editable": True,
            "message": (
                "Profile built from the target's known ligands. Adjust the weights or "
                "exclude criteria, then submit it for candidate scoring."
            ),
        }, profiles

    def _await_target_confirmation(
        self,
        disease: Optional[str],
        candidate_targets: Optional[Sequence[Dict[str, Any]]],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        §1/§5 step 1 — disease only, no target resolved yet.

        The disease identifiers are resolved here (they are needed for the
        exclusion filter either way), but the target shortlist is surfaced for
        confirmation rather than auto-selecting the top-scored target.
        """
        if not disease:
            raise CurateXError(
                "CurateX needs a target or a disease. Provide a target (gene symbol or "
                "protein name), or a disease to carry a target shortlist over from "
                "target identification."
            )
        try:
            resolved = resolve_disease(disease)
        except ResolutionError as exc:
            raise CurateXError(str(exc)) from exc

        shortlist = [
            {
                "name": t.get("name") or t.get("geneName") or t.get("uniprotId", ""),
                "uniprotId": t.get("uniprotId") or t.get("uniprot_id", ""),
                "score": t.get("score"),
                "category": t.get("category"),
            }
            for t in (candidate_targets or [])
        ]
        state = SessionState(session_id=session_id or "", stage=AWAITING_TARGET)
        if session_id:
            _store_session(state)

        return {
            "module": "CurateX",
            "stage": AWAITING_TARGET,
            "disease": resolved.as_dict(),
            "candidateTargets": shortlist,
            "message": (
                f"Resolved {resolved.efo_label or disease}"
                f"{f' (EFO {resolved.efo_id})' if resolved.efo_id else ''}. "
                + (
                    "Confirm which of these targets to curate against — target choice "
                    "changes every downstream profile and score, so CurateX does not pick "
                    "one for you."
                    if shortlist
                    else "Run target identification first, or name a target directly."
                )
            ),
        }

    # ── Tools 2 + 3 ───────────────────────────────────────────────────────────
    @observe(name="curatex_score_candidates")
    def score_candidates(
        self,
        profile: DrugProfile,
        weights: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, Any]] = None,
        top_n: int = 25,
        universe_limit: Optional[int] = None,
        min_phase: Optional[int] = None,
        evidence_top_n: Optional[int] = None,
        skip_evidence: bool = False,
    ) -> Dict[str, Any]:
        """§6 + §7 — score against the submitted profile, then validate evidence."""
        evidence_top_n = int(
            evidence_top_n or getattr(settings, "CURATEX_EVIDENCE_TOP_N", 10) or 10
        )
        if weights:
            # The profile screen edits rows rendered from `label`, so it sends
            # labels back rather than keys. Re-key before validating, and before
            # anything downstream reads the table.
            weights = normalize_weight_table(weights)
            problems = validate_weight_table(weights)
            if problems:
                raise CurateXError("; ".join(problems))

        candidate_set = candidate_service.match_and_score(
            profile,
            limit=universe_limit,
            top_n=top_n,
            min_phase=min_phase,
            weights=weights,
            values=values,
        )

        target_name = profile.target.gene_symbol if profile.target else None
        disease_name = (
            (profile.disease.efo_label or profile.disease.query) if profile.disease else None
        )

        evidence: Dict[str, evidence_service.CandidateEvidence] = {}
        if not skip_evidence:
            evidence = evidence_service.gather_evidence(
                candidate_set.candidates[:evidence_top_n], target_name, disease_name
            )

        recommendation = evidence_service.build_recommendation(
            candidate_set.candidates, evidence, target_name, disease_name, profile.exclusion
        )
        return self._assemble(profile, candidate_set, evidence, recommendation)

    @observe(name="curatex_run")
    def run(
        self,
        target: Optional[str] = None,
        disease: Optional[str] = None,
        targets: Optional[Sequence[str]] = None,
        candidate_targets: Optional[Sequence[Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, Any]] = None,
        top_n: int = 25,
        universe_limit: Optional[int] = None,
        min_phase: Optional[int] = None,
        skip_dailymed: bool = False,
        skip_evidence: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        The full chain. Stops at `awaiting_target_confirmation` when only a
        disease is available, and at `profile_ready` for nothing — the profile is
        editable, but a caller asking for a run gets the scored result built from
        the default (or supplied) weights.
        """
        profile_payload, profiles = self._build_profiles(
            target=target,
            disease=disease,
            targets=targets,
            candidate_targets=candidate_targets,
            weights=weights,
            values=values,
            skip_dailymed=skip_dailymed,
            session_id=session_id,
        )
        if profile_payload["stage"] == AWAITING_TARGET:
            return profile_payload
        if not profiles:
            raise CurateXError("Profile build produced no usable profile")

        state = get_session(session_id) if session_id else None

        results = [
            self.score_candidates(
                profile,
                weights=weights,
                values=values,
                top_n=top_n,
                universe_limit=universe_limit,
                min_phase=min_phase,
                skip_evidence=skip_evidence,
            )
            for profile in profiles
        ]
        payload = results[0] if len(results) == 1 else {
            "module": "CurateX",
            "stage": COMPLETE,
            "perTarget": results,
            "recommendation": "\n\n".join(r["recommendation"] for r in results),
        }
        if state:
            state.last_result = payload
            state.stage = COMPLETE
            _store_session(state)
        return payload

    # ── §8 output assembly ────────────────────────────────────────────────────
    def _assemble(
        self,
        profile: DrugProfile,
        candidate_set: candidate_service.CandidateSet,
        evidence: Dict[str, evidence_service.CandidateEvidence],
        recommendation: str,
    ) -> Dict[str, Any]:
        """
        §8 — the presentation object.

        `candidateTable` is the primary output: one row per candidate with the
        composite score, the per-criterion breakdown that produced it, the
        evidence-strength flag and per-field provenance. `profile` rides along so
        the UI can re-show the editable criteria table next to the scores.
        """
        rows: List[Dict[str, Any]] = []
        for rank, candidate in enumerate(candidate_set.candidates, start=1):
            support = evidence.get(candidate.chembl_id)
            row = candidate.as_dict()
            row.update(
                {
                    "rank": rank,
                    "evidenceStrength": support.strength if support else "not-assessed",
                    "evidenceLinks": (
                        [
                            {"label": a.get("title", ""), "url": a.get("url", ""), "pmid": a.get("pmid", "")}
                            for a in (support.validated_articles or support.articles)
                        ]
                        + support.status_links
                        if support
                        else []
                    ),
                    # §4.3 re-shown per candidate at scoring time.
                    "coverage": {
                        key: {
                            "value": candidate.fields.get(key),
                            "source": candidate.field_sources.get(key),
                            "available": candidate.fields.get(key) is not None,
                        }
                        for key in DAILYMED_CRITERIA
                    },
                }
            )
            rows.append(row)

        score_only = [r["name"] for r in rows if r["evidenceStrength"] == evidence_service.SCORE_ONLY]
        return {
            "module": "CurateX",
            "stage": COMPLETE,
            "target": profile.target.as_dict() if profile.target else None,
            "disease": profile.disease.as_dict() if profile.disease else None,
            "profile": profile.as_dict(),
            "candidateTable": rows,
            "excludedCandidates": candidate_set.excluded,
            "recommendation": recommendation,
            "scoreOnlyCandidates": score_only,
            "metadata": candidate_set.metadata,
            "warnings": profile.warnings,
            "artifact": None,  # §7 step 4 / §8 — table and text only, by design
        }
