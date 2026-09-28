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


def infer_module(query: str, explicit: Optional[str] = None) -> Optional[str]:
    """
    Resolve the module for a query: explicit > @mention > keywords > None.

    Returns None when nothing matches, rather than falling back to TxKG. The
    fallback turned every unrecognised input into a disease query: "open
    report.pdf" ran TxKG, whose fuzzy resolver matched it to "Bone Resorption"
    and propagated a knowledge graph for it. Two loose steps in a row produced a
    confident answer to a question nobody asked. The caller decides what to do
    with None — `POST /sessions` asks the researcher what they meant.
    """
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
    return None


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


#: A symbol written in lower case — "jak2", "stat3". Researchers type these as
#: often as they type JAK2, and requiring capitals meant "dock jak2 with
#: imatinib" resolved no target at all: the whole sentence was sent to UniProt as
#: the target and "jak2" was left to be read as a compound. A trailing digit is
#: what separates a symbol from an ordinary word here — "dock" and "imatinib"
#: have none, so neither is mistaken for a gene.
_GENE_SYMBOL_LOWER = re.compile(r"\b[A-Za-z]{2,8}\d{1,3}[A-Za-z]?\b")


def extract_gene_symbol(query: str) -> str:
    """
    Pull a gene symbol out of a free-text instruction, or return "".

    An all-capitals token wins: casing is the strongest signal a word is a symbol
    rather than prose. Failing that, a token carrying a digit is taken as one, so
    a lower-case "jak2" still resolves. A word with neither — "dock", "imatinib",
    "profile" — is never treated as a gene.
    """
    text = query or ""
    for token in _GENE_SYMBOL.findall(text):
        if token not in _NOT_A_GENE:
            return token
    for token in _GENE_SYMBOL_LOWER.findall(text):
        if token.upper() not in _NOT_A_GENE:
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


#: Words that are instruction, not subject. Whatever survives removing these and
#: the target is what the query is actually about.
_INSTRUCTION_WORDS = {
    "a", "an", "the", "of", "for", "in", "on", "with", "to", "and", "or", "me",
    "my", "please", "create", "build", "generate", "make", "produce", "run",
    "start", "perform", "do", "find", "check", "search", "assess", "analyse",
    "analyze", "evaluate", "get", "show", "identify", "list", "give",
    "drug", "target", "targets", "protein", "proteins", "compound", "compounds",
    "candidate", "candidates", "profile", "literature", "novelty", "patent",
    "patents", "knowledge", "graph", "subgraph", "pathway", "pathways",
    "associated", "repurposing", "repurpose", "about", "related",
    "ideal", "best", "good", "new", "dock", "docking", "screen", "screening",
    "mine", "mining", "review", "study", "combination", "combo",
    # Question forms. "is there prior art for ruxolitinib in thrombocytosis?"
    # reached NovSearch with the target "is there prior art for ruxolitinib",
    # because a question is phrased around the subject rather than naming it.
    # Only ever stripped from the edges of a phrase, so a gene that happens to
    # share a spelling is safe anywhere a gene actually appears.
    "is", "are", "was", "were", "be", "been", "can", "could", "would", "should",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "does", "did", "has", "have", "had", "there", "here", "it", "its",
    "prior", "art", "freedom", "operate", "fto", "tell", "explain", "describe",
    "we", "us", "you", "your", "i",
}

#: Words that appear in a docking request but never name a compound. Small-molecule
#: drug names are lower-case and unremarkable in shape, so there is no pattern to
#: match on — the reliable signal is what is left after the instruction is removed.
#: Shares `_INSTRUCTION_WORDS` rather than keeping a second list: the two drifted,
#: and "find targets for thrombocytosis" reached PubChem as the compounds "find"
#: and "targets" because this copy was missing both.
_NOT_A_COMPOUND = _INSTRUCTION_WORDS | {
    "against", "between", "binding", "affinity", "library", "this", "that",
    "these", "those", "using", "via", "from", "into", "onto", "out", "over",
    "please", "also", "then", "next", "some", "any", "all", "both",
}


#: A message that is actually asking for a docking run, rather than a session
#: query being carried forward into one.
_DOCK_INTENT = re.compile(r"\b(dock(?:ing)?|screen(?:ing)?|binding affinity)\b", re.I)


def _asks_to_dock(query: str) -> bool:
    return bool(_DOCK_INTENT.search(query or ""))


def compounds_from_text(query: str, target: str = "") -> list[dict]:
    """
    Compound names read out of a docking instruction, as the runner's payload.

    Returns `[{"drug_name": ...}]` — the shape ScreenSuite's error message itself
    documents. The target is excluded so "dock jak2 with imatinib" does not also
    try to screen JAK2 against itself, and gene symbols are excluded because a
    compound is not one.
    """
    text = _MODULE_MENTION.sub("", query or "")
    target_lower = (target or "").strip().lower()
    out, seen = [], set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text):
        low = token.lower()
        if low in _NOT_A_COMPOUND or low in seen or low == target_lower:
            continue
        # A gene symbol is a target, not something to dock.
        if token.isupper() or extract_gene_symbol(token):
            continue
        seen.add(low)
        out.append({"drug_name": token})
    return out


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




def _disease_from_query(query: str, target: str) -> str:
    """
    The disease a session's own query is about, or "".

    A TxKG-led session opens with "find protein targets for thrombocytosis", and
    what is left after removing the instruction is the disease. Stop-phrase
    stripping alone is not enough to decide this: it matches known phrasings, so
    "create a drug profile for JAK2" came through as the disease "create a drug
    profile". Removing instruction words and the target, and requiring something
    meaningful to survive, rejects that without needing every phrasing listed.
    """
    target_lower = (target or "").strip().lower()
    left, right = split_target_disease(query)

    for part in (right, left):
        candidate = (part or "").strip()
        if not candidate or candidate.lower() == target_lower:
            continue

        words = [w for w in re.split(r"[^\w\-]+", candidate) if w]
        kept = [
            w for w in words
            if w.lower() not in _INSTRUCTION_WORDS and w.lower() != target_lower
        ]
        if not kept:
            continue
        # A bare gene symbol is a target, not a disease.
        if len(kept) == 1 and extract_gene_symbol(kept[0]) == kept[0]:
            continue
        return " ".join(kept)
    return ""


def _drop_instruction_words(phrase: str) -> str:
    """
    Remove leading/trailing instruction words from a split half.

    `_strip_query_noise` removes known noise *phrases*, not stray verbs, so
    "find patents for ruxolitinib in thrombocytosis" survived it intact and then
    split on " in " into ("find patents for ruxolitinib", "thrombocytosis").
    NovSearch searched for the literal string "find patents for ruxolitinib" and
    returned patents on methotrexate adjuvants and trabecular meshwork — nothing
    to do with the request. Only the edges are trimmed: an interior word can be
    part of a real name ("vitamin D receptor"), whereas a leading "find" or a
    trailing "for" never is.
    """
    words = phrase.split()
    while words and words[0].strip(",.").lower() in _INSTRUCTION_WORDS:
        words.pop(0)
    while words and words[-1].strip(",.").lower() in _INSTRUCTION_WORDS:
        words.pop()
    return " ".join(words).strip()


def split_target_disease(query: str) -> Tuple[str, str]:
    """Split 'HER2 in Breast Cancer' / 'HER2 for Breast Cancer' into its parts."""
    cleaned = _strip_query_noise(query)
    for separator in (" in ", " for ", " against ", " — ", " - ", ","):
        if separator in cleaned:
            left, right = cleaned.split(separator, 1)
            left, right = _drop_instruction_words(left), _drop_instruction_words(right)
            if left and right:
                return left, right
            # One side was pure instruction ("find patents for X"): the surviving
            # side is the subject, not a target/disease pair.
            if left or right:
                return (left or right), ""
    return _drop_instruction_words(cleaned), ""


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
        # Targets ticked on the TxKG step are the whole point of the hand-off.
        # Failing that, the gene symbol in the message: "search literature for
        # jak2" left `subject` as "search jak2" — noise-stripping removes the
        # phrase "literature for" but not the verb — and PubMed found nothing
        # for it, so a routed run returned 0 articles where the picker returned
        # 20. Only when neither exists does the leftover phrase stand in, which
        # is what a disease-led query ("literature on thrombocytosis") needs.
        chosen_ids = selections.get("targetIds")
        if not chosen_ids:
            symbol = extract_gene_symbol(query)
            chosen_ids = [symbol] if symbol else ([subject] if subject else [])
        return {
            "query": query or selections.get("query", ""),
            "targetIds": chosen_ids,
            "maxResults": int(selections.get("maxResults", 20)),
        }

    if module == "CurateX":
        target = chosen or resolve_target(query, subject)
        return {
            "target": target,
            # Without a disease the repurposing exclusion filter has nothing to
            # exclude against and reports itself inactive — which is what
            # happened on every "Continue to CurateX": the hand-off sends only
            # targetIds, so `disease` was None even though the session had been
            # about thrombocytosis since its first message. The session's own
            # query still carries it, so it is derived rather than demanded of
            # the caller.
            "disease": selections.get("disease") or _disease_from_query(query, target),
            "numResults": int(selections.get("numResults", 20)),
            # The researcher's edited scoring weights, set on the CurateX profile screen.
            "weights": selections.get("weights"),
        }

    if module == "ScreenSuite":
        # Docking needs a *protein*: the selected target, or a gene symbol named
        # in the message. Never the leftover phrase — on a TxKG-led session that
        # is the disease, and a hand-off with no selection screened compounds
        # against "thrombocytosis". An empty target is better than a wrong one;
        # the runner says what is missing.
        target = chosen or extract_gene_symbol(query)
        # Compounds named in *this* message outrank whatever was carried over.
        # The other order screened the top CurateX candidate no matter what was
        # asked for: "dock jak2 with ruxolitinib" arrived with the CurateX
        # selection still attached, so it docked Omega-3-carboxylic acid and the
        # explicitly named compound was dropped without a word. Naming a compound
        # is the clearest statement of intent available, so it wins; the carried
        # selection is the default for a hand-off that names none.
        #
        # Still gated on `_asks_to_dock`, because the session query carries
        # forward unchanged and a hand-off from "find protein targets for
        # thrombocytosis" would otherwise read the disease as a ligand.
        named = compounds_from_text(query, target) if _asks_to_dock(query) else []
        compounds = named or selections.get("compounds") or []
        return {
            "target": target,
            "compoundLibrary": selections.get("compoundLibrary"),
            # CurateX's "View in ScreenSuite" sends the chosen compounds here.
            "compounds": compounds,
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
