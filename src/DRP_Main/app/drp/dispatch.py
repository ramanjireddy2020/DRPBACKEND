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


#: An instruction to run a module, as opposed to a question about the results on
#: screen. Only an explicit `@mention` used to move a conversation between agents,
#: so "create a drug profile for JAK2" was answered out of whatever the current
#: module had already returned — the agent would even describe the other module
#: while refusing to start it. Keyword matching alone cannot replace the mention:
#: "what about the patents on these?" is a question about the current results, not
#: a request to run NovSearch. So a match needs an action verb *and* that module's
#: object, which a question phrased as a question will not satisfy.
#: Verbs that begin an instruction. "find" and "check" were missing, so "find the
#: novelty of the combination jak2 and imatinib" — a direct request to run
#: NovSearch — was answered out of the CurateX results instead of starting it.
_ACTION = (r"(?:create|build|generate|make|produce|run|start|perform|do|find|"
           r"check|search|assess|analyse|analyze|evaluate|get|show)")

#: Verbs that are themselves the instruction: "dock jak2 and imatinib" names the
#: module's work in one word, with no separate object to match against.
_BARE_VERB_INTENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*dock(?:ing)?\b", re.I), "ScreenSuite"),
    (re.compile(r"^\s*screen(?:ing)?\b", re.I), "ScreenSuite"),
    (re.compile(r"^\s*curate\b", re.I), "CurateX"),
]

_MODULE_INTENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{_ACTION}\b[^.?!]*\b(?:drug|target|compound|candidate)\s+profile\b", re.I),
     "CurateX"),
    (re.compile(rf"\b{_ACTION}\b[^.?!]*\bcurat(?:e|ion)\b", re.I), "CurateX"),
    (re.compile(rf"\b{_ACTION}\b[^.?!]*\b(?:novelty|novel|freedom[- ]to[- ]operate|fto|prior art|patent)\b", re.I),
     "NovSearch"),
    (re.compile(rf"\b{_ACTION}\b[^.?!]*\b(?:dock(?:ing)?|screen(?:ing)?|binding affinity)\b", re.I),
     "ScreenSuite"),
    (re.compile(rf"\b{_ACTION}\b[^.?!]*\b(?:literature|pubmed|papers?|publications?)\b", re.I),
     "LitMineX"),
    (re.compile(rf"\b{_ACTION}\b[^.?!]*\b(?:knowledge graph|subgraph)\b", re.I), "TxKG"),
    (re.compile(rf"\b{_ACTION}\b[^.?!]*\b(?:protein )?targets?\s+(?:for|associated with)\b", re.I),
     "TxKG"),
]


def explicit_module_request(message: str) -> Optional[str]:
    """
    The module a message explicitly asks to *run*, or None.

    Deliberately conservative: it answers None for anything that reads as a
    question about the current results, because moving the conversation to
    another agent mid-read is worse than answering in place.
    """
    text = (message or "").strip()
    if not text:
        return None
    # A question is a question however it opens: "do these dock?" is not an
    # instruction to run ScreenSuite, so anything ending in "?" is left alone.
    if text.endswith("?"):
        return None
    for pattern, module in _BARE_VERB_INTENTS:
        if pattern.search(text):
            return module
    for pattern, module in _MODULE_INTENTS:
        if pattern.search(text):
            return module
    return None


def _selected_target(selections: Dict[str, Any]) -> str:
    """
    The single target a step hand-off chose, under any of the keys the UI sends.

    The spec's hand-off payload is `{"targetIds": [...]}` — that is the example on
    `POST /sessions/{id}/steps` and what LitMineX already reads. CurateX and
    ScreenSuite only looked for a singular `target`, found nothing, and fell back
    to parsing the *original composer query*, which on a TxKG-led session is still
    "find protein targets for thrombocytosis". So "Continue to CurateX" after
    picking JAK2 profiled thrombocytosis instead: the researcher's explicit choice
    was discarded in favour of a phrase from a question they asked three steps ago.

    Takes the first id when several are ticked — CurateX and ScreenSuite each act
    on one target at a time.
    """
    single = selections.get("target")
    if isinstance(single, str) and single.strip():
        return single.strip()

    for key in ("targetIds", "targets", "selectedTargets"):
        value = selections.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)):
            for entry in value:
                if isinstance(entry, str) and entry.strip():
                    return entry.strip()
                # Rows are sometimes sent whole rather than as bare ids.
                if isinstance(entry, dict):
                    for field in ("target", "uniprotId", "geneName", "name", "id"):
                        candidate = entry.get(field)
                        if isinstance(candidate, str) and candidate.strip():
                            return candidate.strip()
    return ""


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
    chosen = _selected_target(selections)

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
            "target": chosen or resolve_target(query, subject),
            "disease": selections.get("disease"),
            "numResults": int(selections.get("numResults", 20)),
            # The researcher's edited scoring weights, set on the CurateX profile screen.
            "weights": selections.get("weights"),
        }

    if module == "ScreenSuite":
        return {
            "target": chosen or resolve_target(query, subject),
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
            "target": chosen or target,
            "disease": selections.get("disease") or disease,
            "drug": selections.get("drug"),
            # Present on a ScreenSuite carry-over so the report ties back to the
            # specific screening result the researcher was looking at.
            "candidateId": selections.get("candidateId"),
            "dockingId": selections.get("dockingId"),
            "numResults": int(selections.get("numResults", 5)),
        }

    return {"query": query}


def _curatex_stage(kind: str, params: Dict[str, Any]) -> str:
    """
    Which CurateX job a request wants: build the profile, or score against it.

    CurateX is two steps by design — build an editable profile, let the
    researcher adjust it, then score candidates against what they approved. The
    module→job map points at `curatex.compounds`, which does both at once, so a
    conversational start ("create a drug profile for JAK2") skipped straight to a
    ranked list. The researcher then saw "20 candidates ranked" *before* being
    offered a profile to edit, and the profile screen had nothing pending to
    submit — the edit step existed but nothing ever routed through it.

    Weights or values present mean the researcher has already edited a profile
    and this is the scoring pass; their absence means they are still at the start.
    """
    if kind != "curatex.compounds":
        return kind
    if params.get("weights") or params.get("values"):
        return kind
    return "curatex.target_profile"


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
    kind = _curatex_stage(kind, params)
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
