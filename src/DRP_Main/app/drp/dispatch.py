"""
Composer-query → agent-job routing.

`POST /v1/sessions` accepts a free-text query with an optional `@module` mention.
This module decides which agent runs and builds its parameters.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from DRP_Main.app.drp.catalog import canonical_module
from DRP_Main.app.drp.jobs import create_job, enqueue
from DRP_Main.app.drp.models import DrpJob
from DRP_Main.app.drp.runners import _strip_query_noise

# Keyword → module, checked in order. First match wins.
_INTENT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("patent", "novelty", "freedom to operate", "fto", "prior art"), "NovSearch"),
    (("dock", "docking", "screen", "screening", "binding affinity", "library"), "ScreenSuite"),
    (("curate", "compound", "drug candidate", "candidate profile"), "CurateX"),
    (("literature", "article", "pubmed", "paper", "publication", "mine"), "LitMineX"),
    (("target", "knowledge graph", "subgraph", "pathway", "protein", "repurpos"), "TxKG"),
]

_MODULE_MENTION = re.compile(r"@([A-Za-z][\w-]*)")

JOB_KIND_BY_MODULE = {
    "TxKG": "txkg.query",
    "LitMineX": "litminex.query",
    "CurateX": "curatex.compounds",
    "ScreenSuite": "screensuite.screen",
    "NovSearch": "novsearch.assess",
    # The end-to-end chain the module catalogue advertises. Without this entry
    # `/v1/modules` offered "SaaS Pipeline" and selecting it returned 422.
    "SaaS Pipeline": "pipeline.run",
}


def extract_module_mention(query: str) -> Optional[str]:
    """Pull an '@TxKG'-style module mention out of the composer text."""
    for match in _MODULE_MENTION.finditer(query or ""):
        candidate = canonical_module(match.group(1))
        if candidate in JOB_KIND_BY_MODULE:
            return candidate
    return None


def infer_module(query: str, explicit: Optional[str] = None) -> str:
    """Resolve the module for a query: explicit > @mention > keywords > TxKG."""
    resolved = canonical_module(explicit)
    if resolved in JOB_KIND_BY_MODULE:
        return resolved
    mentioned = extract_module_mention(query)
    if mentioned:
        return mentioned
    lowered = (query or "").lower()
    for keywords, module in _INTENT_RULES:
        if any(keyword in lowered for keyword in keywords):
            return module
    return "TxKG"


#: Gene symbols are written in caps with an optional digit — JAK2, EGFR, TP53,
#: SGLT2, STAT5B. Matching on that shape lets an instruction like "create drug
#: profile for JAK2" yield "JAK2" instead of sending the whole sentence to UniProt.
_GENE_SYMBOL = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

#: All-caps words that look like symbols but are not. Without this, "DNA", "FDA"
#: or a shouted "FIND" would be taken as the target.
_NOT_A_GENE = {
    "DNA", "RNA", "FDA", "EMA", "PDB", "USA", "AND", "OR", "NOT", "FOR", "THE",
    "WITH", "FIND", "GET", "ALL", "NEW", "TOP", "API", "ID", "IDS", "AI", "ML",
    "MOA", "ADME", "IC50", "EC50", "PK", "PD", "QSAR", "SAR", "HTS",
}


def extract_gene_symbol(query: str) -> str:
    """
    Pull a gene symbol out of a free-text instruction, or return "".

    Case matters: only the original casing distinguishes a symbol from an ordinary
    word, so this runs on the raw query rather than the lower-cased one.
    """
    for token in _GENE_SYMBOL.findall(query or ""):
        if token not in _NOT_A_GENE:
            return token
    return ""


def resolve_target(query: str, subject: str) -> str:
    """
    The target a module should act on: an explicit gene symbol wins over the
    leftover phrase, which may still be prose after noise-stripping.
    """
    return extract_gene_symbol(query) or subject


def split_target_disease(query: str) -> Tuple[str, str]:
    """Split 'HER2 in Breast Cancer' / 'HER2 for Breast Cancer' into its parts."""
    cleaned = _strip_query_noise(query)
    for separator in (" in ", " for ", " against ", " — ", " - ", ","):
        if separator in cleaned:
            left, right = cleaned.split(separator, 1)
            if left.strip() and right.strip():
                return left.strip(), right.strip()
    return cleaned, ""


def build_params(
    module: str, query: str, selections: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Derive runner parameters for a module from the composer query.

    `selections` are what the researcher chose on the *previous* step — the ticked
    target ids, the compounds sent on from CurateX. They override what can be
    inferred from the query text, because an explicit choice is always better
    evidence of intent than a phrase parsed out of free text. Passing none keeps
    the original text-only behaviour, which is what the first step of a session
    uses.
    """
    selections = selections or {}
    subject = _strip_query_noise(_MODULE_MENTION.sub("", query or "")).strip()

    if module == "TxKG":
        return {
            "query": query,
            "limit": int(selections.get("limit", 10)),
            "maxHops": int(selections.get("maxHops", 3)),
        }

    if module == "LitMineX":
        # Targets ticked on the TxKG step are the whole point of the hand-off;
        # fall back to the query subject only when nothing was selected.
        target_ids = selections.get("targetIds") or ([subject] if subject else [])
        return {
            "query": query or selections.get("query", ""),
            "targetIds": target_ids,
            "maxResults": int(selections.get("maxResults", 20)),
        }

    if module == "CurateX":
        return {
            "target": selections.get("target") or resolve_target(query, subject),
            "disease": selections.get("disease"),
            "numResults": int(selections.get("numResults", 20)),
            # The researcher's edited scoring weights, set on the CurateX profile screen.
            "weights": selections.get("weights"),
        }

    if module == "ScreenSuite":
        return {
            "target": selections.get("target") or resolve_target(query, subject),
            "compoundLibrary": selections.get("compoundLibrary"),
            # CurateX's "View in ScreenSuite" sends the chosen compounds here.
            "compounds": selections.get("compounds") or [],
        }

    if module == "SaaS Pipeline":
        # The chain derives each stage's inputs from the previous stage's output,
        # so it needs only the opening question plus how far to carry it.
        return {
            "query": query,
            "limit": int(selections.get("limit", 10)),
            "carryTargets": int(selections.get("carryTargets", 3)),
            "maxResults": int(selections.get("maxResults", 20)),
            "stages": selections.get("stages")
            or ["TxKG", "LitMineX", "CurateX", "ScreenSuite", "NovSearch"],
        }

    if module == "NovSearch":
        target, disease = split_target_disease(query)
        return {
            "target": selections.get("target") or target,
            "disease": selections.get("disease") or disease,
            "drug": selections.get("drug"),
            # Present on a ScreenSuite carry-over so the report ties back to the
            # specific screening result the researcher was looking at.
            "candidateId": selections.get("candidateId"),
            "dockingId": selections.get("dockingId"),
            "numResults": int(selections.get("numResults", 5)),
        }

    return {"query": query}


def start_module_job(
    db: Session,
    *,
    user_id: int,
    module: str,
    query: str,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    session_step_id: Optional[str] = None,
    selections: Optional[Dict[str, Any]] = None,
    params_override: Optional[Dict[str, Any]] = None,
) -> DrpJob:
    """
    Create and start the agent job backing one step of a session.

    `params_override` is merged over the derived parameters, which is how a re-run
    applies the researcher's adjusted settings without re-deriving them from the
    original query text.
    """
    kind = JOB_KIND_BY_MODULE.get(module)
    if kind is None:
        raise ValueError(f"No agent job is defined for module '{module}'")
    params = build_params(module, query, selections)
    if params_override:
        params.update(params_override)
    job = create_job(
        db,
        user_id=user_id,
        kind=kind,
        module=module,
        params=params,
        session_id=session_id,
        project_id=project_id,
        session_step_id=session_step_id,
    )
    enqueue(job)
    return job
