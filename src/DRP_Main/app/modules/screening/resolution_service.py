"""
ScreenSuite resolution step — names/identifiers in, one resolved list out.

Runs as ordinary code inside the ScreenSuite agent *before* the pipeline is
submitted. It is deliberately not an LLM-selectable step: it must happen before
docking every time, and there is no scenario where skipping or reordering it
makes sense.

Two rules shape the whole module:

* **Every resolved record carries its source.** Structure retrieval branches on
  source (PDB vs. AlphaFold, PubChem vs. ZINC), so a source-less identifier is
  not enough to act on unambiguously.
* **Ambiguity pauses, it never guesses.** Docking is expensive; silently taking
  the top match and screening against the wrong structure is a worse failure
  than one confirmation round-trip. This mirrors CurateX's
  `awaiting_target_confirmation` rather than inventing a second mechanism.

An identifier supplied by the caller is trusted as-is — no redundant lookup.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.http_client import get_json, get_session
from DRP_Main.app.modules.screening.enums import ResolutionStage
from DRP_Main.app.modules.screening.schemas import (
    AmbiguousEntity,
    CandidateMatch,
    DrugQuery,
    DrugSource,
    ProteinQuery,
    ProteinSource,
    ResolutionResult,
    ResolvedDrug,
    ResolvedProtein,
    UnresolvedEntity,
)

logger = get_logger(__name__)

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_PREDICTION = "https://alphafold.ebi.ac.uk/api/prediction"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

#: A 4-character PDB code — one digit followed by three alphanumerics.
_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
#: A UniProt accession, which is what AlphaFold keys its models on.
_UNIPROT_ID = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")


def _timeout() -> int:
    return int(settings.SCREENING_RESOLVE_TIMEOUT or 20)


def _max_matches() -> int:
    return max(int(settings.SCREENING_RESOLVE_MAX_MATCHES or 5), 1)


# ── Protein resolution ────────────────────────────────────────────────────────

def _rcsb_search(name: str, rows: int) -> List[str]:
    """Full-text search RCSB for `name`, returning PDB entry ids in rank order."""
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": name},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_verbosity": "compact",
        },
    }
    try:
        response = get_session().post(RCSB_SEARCH, json=payload, timeout=_timeout())
        # RCSB answers 204 with no body when a query matches nothing.
        if response.status_code == 204:
            return []
        response.raise_for_status()
        body = response.json()
    except Exception as exc:  # noqa: BLE001 — a dead source must not raise here
        logger.warning("RCSB search for %r failed: %s", name, exc)
        return []

    result_set = body.get("result_set") or []
    identifiers: List[str] = []
    for item in result_set:
        # `compact` verbosity returns bare id strings; the default returns dicts.
        identifier = item if isinstance(item, str) else item.get("identifier")
        if identifier:
            identifiers.append(str(identifier).upper())
    return identifiers


def _rcsb_title(pdb_id: str) -> Optional[str]:
    """The entry's structure title, for display in a confirmation shortlist."""
    body = get_json(f"{RCSB_ENTRY}/{quote(pdb_id)}", timeout=_timeout())
    if not isinstance(body, dict):
        return None
    return (body.get("struct") or {}).get("title")


def _uniprot_accession(name: str) -> Optional[Tuple[str, str]]:
    """Best reviewed-human UniProt hit for `name`, as (accession, label)."""
    body = get_json(
        UNIPROT_SEARCH,
        {
            "query": f"{name} AND (reviewed:true)",
            "fields": "accession,protein_name,id",
            "format": "json",
            "size": 1,
        },
        timeout=_timeout(),
    )
    results = (body or {}).get("results") or []
    if not results:
        return None
    entry = results[0]
    accession = entry.get("primaryAccession")
    if not accession:
        return None
    label = (
        ((entry.get("proteinDescription") or {}).get("recommendedName") or {})
        .get("fullName", {})
        .get("value")
    ) or entry.get("uniProtkbId") or name
    return str(accession), str(label)


def _alphafold_exists(accession: str) -> bool:
    body = get_json(f"{ALPHAFOLD_PREDICTION}/{quote(accession)}", timeout=_timeout())
    return bool(body)


def resolve_protein(query: ProteinQuery) -> Dict[str, Any]:
    """
    Resolve one protein.

    Returns `{"resolved": ResolvedProtein}`, `{"ambiguous": AmbiguousEntity}` or
    `{"unresolved": UnresolvedEntity}` — exactly one key.
    """
    name = (query.name or "").strip()
    if not name:
        return {"unresolved": UnresolvedEntity(name=query.name or "", kind="protein", reason="empty name")}

    # Already resolved upstream — trust it, look nothing up.
    if query.identifier:
        identifier = query.identifier.strip()
        source = query.source
        if source is None:
            # Infer only from identifier shape, which is unambiguous between
            # a 4-character PDB code and a UniProt accession.
            if _PDB_ID.match(identifier):
                source = ProteinSource.pdb
            elif _UNIPROT_ID.match(identifier):
                source = ProteinSource.alphafold
            else:
                return {
                    "unresolved": UnresolvedEntity(
                        name=name,
                        kind="protein",
                        reason=f"identifier {identifier!r} matches neither a PDB code nor a UniProt accession; "
                               "supply `source` explicitly",
                    )
                }
        return {
            "resolved": ResolvedProtein(
                name=name,
                identifier=identifier.upper() if source == ProteinSource.pdb else identifier,
                source=source,
            )
        }

    # Experimental structures first — they dock better than predicted models.
    hits = _rcsb_search(name, _max_matches())
    if len(hits) == 1:
        return {"resolved": ResolvedProtein(name=name, identifier=hits[0], source=ProteinSource.pdb)}
    if len(hits) > 1:
        return {
            "ambiguous": AmbiguousEntity(
                name=name,
                kind="protein",
                candidates=[
                    CandidateMatch(identifier=h, source=ProteinSource.pdb.value, title=_rcsb_title(h))
                    for h in hits
                ],
            )
        }

    # No experimental structure — fall back to a predicted AlphaFold model.
    uniprot = _uniprot_accession(name)
    if uniprot and _alphafold_exists(uniprot[0]):
        accession, label = uniprot
        return {
            "resolved": ResolvedProtein(
                name=name, identifier=accession, source=ProteinSource.alphafold
            )
        }

    return {
        "unresolved": UnresolvedEntity(
            name=name,
            kind="protein",
            reason="no RCSB structure and no AlphaFold model found",
        )
    }


# ── Drug resolution ───────────────────────────────────────────────────────────

def _pubchem_cids(name: str) -> List[str]:
    body = get_json(f"{PUBCHEM_BASE}/name/{quote(name)}/cids/JSON", timeout=_timeout())
    cids = ((body or {}).get("IdentifierList") or {}).get("CID") or []
    return [str(c) for c in cids]


def _pubchem_titles(cids: List[str]) -> Dict[str, str]:
    """Compound titles for a shortlist. Best-effort — absence is not fatal."""
    if not cids:
        return {}
    body = get_json(
        f"{PUBCHEM_BASE}/cid/{','.join(cids)}/property/Title/JSON", timeout=_timeout()
    )
    rows = ((body or {}).get("PropertyTable") or {}).get("Properties") or []
    return {str(r.get("CID")): r.get("Title", "") for r in rows if r.get("CID") is not None}


def resolve_drug(query: DrugQuery) -> Dict[str, Any]:
    """
    Resolve one drug. Same three-way return as `resolve_protein`.

    ZINC is supported only when the caller supplies a ZINC identifier directly:
    resolution never *chooses* it, because ZINC15's substance search is no
    longer a dependable lookup surface.
    """
    name = (query.name or "").strip()
    if not name:
        return {"unresolved": UnresolvedEntity(name=query.name or "", kind="drug", reason="empty name")}

    if query.identifier:
        identifier = query.identifier.strip()
        source = query.source
        if source is None:
            source = DrugSource.zinc if identifier.upper().startswith("ZINC") else DrugSource.pubchem
        return {"resolved": ResolvedDrug(name=name, identifier=identifier, source=source)}

    cids = _pubchem_cids(name)
    if len(cids) == 1:
        return {"resolved": ResolvedDrug(name=name, identifier=cids[0], source=DrugSource.pubchem)}
    if len(cids) > 1:
        shortlist = cids[: _max_matches()]
        titles = _pubchem_titles(shortlist)
        return {
            "ambiguous": AmbiguousEntity(
                name=name,
                kind="drug",
                candidates=[
                    CandidateMatch(
                        identifier=c, source=DrugSource.pubchem.value, title=titles.get(c) or None
                    )
                    for c in shortlist
                ],
            )
        }

    return {
        "unresolved": UnresolvedEntity(
            name=name, kind="drug", reason="no PubChem compound found for this name"
        )
    }


# ── The step itself ───────────────────────────────────────────────────────────

def _confirmation_message(ambiguous: List[AmbiguousEntity], unresolved: List[UnresolvedEntity]) -> str:
    names = ", ".join(f"{a.name} ({len(a.candidates)} matches)" for a in ambiguous)
    message = (
        f"Confirm which structure to screen for: {names}. "
        "Docking is expensive and the choice changes every downstream result, "
        "so ScreenSuite does not pick one for you."
    )
    if unresolved:
        message += " Not resolved at all: " + ", ".join(u.name for u in unresolved) + "."
    return message


@observe(name="screensuite_resolution")
def resolve(
    proteins: List[ProteinQuery], drugs: List[DrugQuery], *, allow_partial: bool = False
) -> ResolutionResult:
    """
    Resolve a whole batch.

    `stage` is `awaiting_structure_confirmation` while anything is ambiguous.
    With `allow_partial=True` an ambiguous batch still reports whatever resolved
    cleanly, so a caller can screen the unambiguous part and confirm the rest —
    the stage still says confirmation is outstanding.
    """
    langfuse_context.update_current_observation(
        input={"protein_count": len(proteins), "drug_count": len(drugs)}
    )

    resolved_proteins: List[ResolvedProtein] = []
    resolved_drugs: List[ResolvedDrug] = []
    ambiguous: List[AmbiguousEntity] = []
    unresolved: List[UnresolvedEntity] = []

    for query in proteins:
        outcome = resolve_protein(query)
        if "resolved" in outcome:
            resolved_proteins.append(outcome["resolved"])
        elif "ambiguous" in outcome:
            ambiguous.append(outcome["ambiguous"])
        else:
            unresolved.append(outcome["unresolved"])

    for query in drugs:
        outcome = resolve_drug(query)
        if "resolved" in outcome:
            resolved_drugs.append(outcome["resolved"])
        elif "ambiguous" in outcome:
            ambiguous.append(outcome["ambiguous"])
        else:
            unresolved.append(outcome["unresolved"])

    if ambiguous:
        result = ResolutionResult(
            stage=ResolutionStage.awaiting_confirmation,
            proteins=resolved_proteins if allow_partial else [],
            drugs=resolved_drugs if allow_partial else [],
            ambiguous=ambiguous,
            unresolved=unresolved,
            message=_confirmation_message(ambiguous, unresolved),
        )
    else:
        parts = []
        if resolved_proteins:
            parts.append(f"{len(resolved_proteins)} protein(s)")
        if resolved_drugs:
            parts.append(f"{len(resolved_drugs)} drug(s)")
        message = "Resolved " + (" and ".join(parts) if parts else "nothing") + "."
        if unresolved:
            message += " Could not resolve: " + ", ".join(u.name for u in unresolved) + "."
        result = ResolutionResult(
            stage=ResolutionStage.resolved,
            proteins=resolved_proteins,
            drugs=resolved_drugs,
            ambiguous=[],
            unresolved=unresolved,
            message=message,
        )

    langfuse_context.update_current_observation(
        output={
            "stage": result.stage,
            "resolved_proteins": len(result.proteins),
            "resolved_drugs": len(result.drugs),
            "ambiguous": len(result.ambiguous),
            "unresolved": len(result.unresolved),
        }
    )
    return result


def confirm(
    selections: List[Dict[str, Any]],
    pending_proteins: Optional[List[ProteinQuery]] = None,
    pending_drugs: Optional[List[DrugQuery]] = None,
) -> ResolutionResult:
    """
    Apply a user's shortlist choices, then re-resolve.

    Each selection is `{"name", "kind", "identifier", "source"}`. Once confirmed,
    a choice is treated exactly like any other resolved entity from here on.
    """
    protein_overrides: Dict[str, Dict[str, str]] = {}
    drug_overrides: Dict[str, Dict[str, str]] = {}
    for selection in selections or []:
        name = str(selection.get("name", "")).strip()
        if not name:
            continue
        target = protein_overrides if selection.get("kind") == "protein" else drug_overrides
        target[name.lower()] = {
            "identifier": str(selection.get("identifier", "")).strip(),
            "source": str(selection.get("source", "")).strip(),
        }

    proteins: List[ProteinQuery] = []
    for query in pending_proteins or []:
        override = protein_overrides.get(query.name.strip().lower())
        if override and override["identifier"]:
            proteins.append(
                ProteinQuery(
                    name=query.name,
                    identifier=override["identifier"],
                    source=override["source"] or None,
                )
            )
        else:
            proteins.append(query)

    drugs: List[DrugQuery] = []
    for query in pending_drugs or []:
        override = drug_overrides.get(query.name.strip().lower())
        if override and override["identifier"]:
            drugs.append(
                DrugQuery(
                    name=query.name,
                    identifier=override["identifier"],
                    source=override["source"] or None,
                )
            )
        else:
            drugs.append(query)

    return resolve(proteins, drugs)
