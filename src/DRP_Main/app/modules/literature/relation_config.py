"""
LitMineX Tool 2 — fixed lookup tables.

Everything the relevance scorer treats as a constant lives here so the weights can
be spot-checked against known-good papers and adjusted in one place (spec §4).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ── Relation-phrase list (spec §4, step 2) ────────────────────────────────────
# Order matters only for reporting: the first phrase that matches is recorded.
RELATION_PHRASES: Tuple[str, ...] = (
    "regulates",
    "regulated by",
    "upregulates",
    "downregulates",
    "activates",
    "activation of",
    "inhibits",
    "inhibition of",
    "suppresses",
    "induces",
    "modulates",
    "mediates",
    "phosphorylates",
    "targets",
    "increases",
    "decreases",
    "reduces",
    "improves",
    "impairs",
    "attenuates",
    "ameliorates",
    "protects against",
    "contributes to",
    "causes",
    "leads to",
    "results in",
    "associated with",
    "correlated with",
    "linked to",
    "implicated in",
    "involved in",
    "plays a role in",
    "plays a key role in",
    "is a biomarker for",
    "predisposes to",
)

# Pre-compiled, longest-first so "plays a key role in" wins over "plays a role in".
_RELATION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (phrase, re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", re.I))
    for phrase in sorted(RELATION_PHRASES, key=len, reverse=True)
]

# Dependency-relation labels that count as a syntactic link between two entities.
# Used to confirm the relation phrase actually joins the target and the disease
# rather than merely sharing the sentence with them.
LINKING_DEPS: frozenset = frozenset(
    {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "attr", "acomp", "xcomp", "prep",
     "agent", "conj", "appos", "amod", "compound", "nmod", "obl"}
)

# ── Positional weights (spec §4, step 3) ─────────────────────────────────────
SECTION_WEIGHTS: Dict[str, float] = {
    "title": 1.0,
    "abstract": 1.0,
    "results": 0.6,
    "discussion": 0.6,
    "conclusion": 0.6,
    "methods": 0.3,
    "introduction": 0.3,
    "intro": 0.3,
    "references": 0.3,
}
DEFAULT_SECTION_WEIGHT = 0.3

# ── Evidence-type detection (spec §4, step 4) ────────────────────────────────
# A relation sentence is "cited" when a citation marker sits within this many
# characters of the matched relation phrase; otherwise it is "primary".
CITATION_WINDOW_CHARS = 120
CITATION_MARKERS: Tuple[re.Pattern, ...] = (
    re.compile(r"\[\s*\d+(\s*[,\-–]\s*\d+)*\s*\]"),        # [12]  [12,14]  [12-15]
    re.compile(r"\(\s*\d+(\s*[,\-–]\s*\d+)*\s*\)"),        # (12)
    re.compile(r"\bet\s+al\.?"),                            # Smith et al.
    re.compile(r"\(\s*[A-Z][A-Za-z\-']+[^)]{0,40}?\b(19|20)\d{2}[a-z]?\s*\)"),  # (Smith 2019)
    re.compile(r"\bref(?:s|erence)?\.?\s*\d+", re.I),
)

# ── Final score combination (spec §4, step 6) ────────────────────────────────
SCORE_WEIGHTS: Dict[str, float] = {
    "relation": 0.40,
    "position": 0.25,
    "similarity": 0.20,
    "primary": 0.15,
}
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def match_relation_phrase(sentence: str) -> str | None:
    """Return the first relation phrase present in the sentence, else None."""
    for phrase, pattern in _RELATION_PATTERNS:
        if pattern.search(sentence):
            return phrase
    return None


def find_relation_span(sentence: str, phrase: str) -> Tuple[int, int] | None:
    """Character span of `phrase` inside `sentence` (case-insensitive)."""
    for candidate, pattern in _RELATION_PATTERNS:
        if candidate != phrase:
            continue
        match = pattern.search(sentence)
        return match.span() if match else None
    return None


def section_weight(section: str) -> float:
    """Positional weight for a section name, tolerant of PMC's varied headings."""
    key = (section or "").strip().lower()
    if key in SECTION_WEIGHTS:
        return SECTION_WEIGHTS[key]
    for name, weight in SECTION_WEIGHTS.items():
        if name in key:
            return weight
    return DEFAULT_SECTION_WEIGHT


def is_cited(sentence: str, relation_span: Tuple[int, int] | None) -> bool:
    """
    True when a citation marker sits within CITATION_WINDOW_CHARS of the relation
    phrase — the claim is being attributed to another paper rather than reported.
    """
    if relation_span is None:
        window = sentence
    else:
        start = max(0, relation_span[0] - CITATION_WINDOW_CHARS)
        end = min(len(sentence), relation_span[1] + CITATION_WINDOW_CHARS)
        window = sentence[start:end]
    return any(marker.search(window) for marker in CITATION_MARKERS)
