"""
LitMineX biomedical NER (spec §3 step 1 and §4 step 1).

Wraps the two scispaCy models the spec names — `en_ner_bc5cdr_md` for diseases and
`en_ner_bionlp13cg_md` for genes/proteins — behind a lazy loader, following the same
pattern the screening module uses for PyMOL: heavy models are never imported at module
import time, and a missing model degrades to a deterministic rule-based tagger instead
of breaking the API.
"""
from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional, Tuple

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

# scispaCy entity labels that count as a disease / a gene-or-protein.
DISEASE_LABELS = frozenset({"DISEASE"})
GENE_LABELS = frozenset({"GENE_OR_GENE_PRODUCT", "PROTEIN", "AMINO_ACID"})

# Rule-based fallback signals, used only when the scispaCy models are unavailable.
_DISEASE_HINTS = (
    "disease", "diseases", "syndrome", "disorder", "cancer", "carcinoma", "tumor",
    "tumour", "diabetes", "diabetic", "obesity", "alzheimer", "parkinson",
    "sclerosis", "itis", "emia", "opathy", "oma",
)
_GENE_TOKEN = re.compile(r"^[A-Z][A-Z0-9\-]{1,9}\d?$")   # AMPK, PRKAA1, TP53, IL-6
_STOPWORDS = frozenset(
    {"what", "is", "the", "of", "in", "a", "an", "and", "or", "for", "with", "to",
     "role", "involvement", "involved", "does", "do", "how", "why", "between",
     "type", "show", "me", "find", "about", "literature", "articles", "papers",
     "evidence", "on"}
)
# Verbs that join a target to a disease — never part of the disease name itself.
_RELATION_VERBS = frozenset(
    {"drive", "drives", "cause", "causes", "regulate", "regulates", "activate",
     "activates", "inhibit", "inhibits", "affect", "affects", "influence",
     "influences", "contribute", "contributes", "promote", "promotes", "modulate",
     "modulates", "mediate", "mediates", "worsen", "worsens", "trigger", "triggers"}
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


class _ModelCache:
    """Process-wide lazy cache for the scispaCy pipelines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, object] = {}
        self._failed: set[str] = set()

    def get(self, name: str):
        if name in self._failed:
            return None
        model = self._models.get(name)
        if model is not None:
            return model
        with self._lock:
            if name in self._models:
                return self._models[name]
            if name in self._failed:
                return None
            try:
                import spacy  # noqa: PLC0415 — heavy, load on first use only
                self._models[name] = spacy.load(name)
                logger.info("LitMineX: loaded scispaCy model %s", name)
            except Exception as exc:  # model or spaCy not installed
                logger.warning(
                    "LitMineX: scispaCy model %s unavailable (%s); "
                    "falling back to rule-based entity tagging",
                    name,
                    exc,
                )
                self._failed.add(name)
                return None
            return self._models[name]


_MODELS = _ModelCache()


class NERService:
    """Entity extraction and mention tagging for targets and diseases."""

    def __init__(
        self,
        disease_model: Optional[str] = None,
        gene_model: Optional[str] = None,
    ) -> None:
        self.disease_model = disease_model or settings.LITMINEX_NER_DISEASE_MODEL
        self.gene_model = gene_model or settings.LITMINEX_NER_GENE_MODEL

    # ── availability ─────────────────────────────────────────────────────────
    @property
    def models_available(self) -> bool:
        return bool(_MODELS.get(self.disease_model) or _MODELS.get(self.gene_model))

    # ── §3 step 1: pull target + disease out of a free-text query ────────────
    def extract_query_entities(self, query: str) -> Dict[str, Optional[str]]:
        """
        Return {"target": ..., "disease": ...}; either may be None when the query
        does not name it. The agent asks the supervisor to prompt for whatever is
        missing rather than guessing (spec §2).
        """
        text = (query or "").strip()
        if not text:
            return {"target": None, "disease": None}

        target = self._first(self._entities(text, self.gene_model, GENE_LABELS))
        disease = self._first(self._entities(text, self.disease_model, DISEASE_LABELS))

        if target is None:
            target = self._fallback_target(text)
        if disease is None:
            disease = self._fallback_disease(text)
        return {"target": target, "disease": disease}

    # ── §4 step 1: tag mentions inside a section of article text ────────────
    def tag_mentions(
        self,
        text: str,
        target_terms: List[str],
        disease_terms: List[str],
    ) -> Dict[str, List[str]]:
        """
        Tag target and disease mentions in `text`.

        The expanded MeSH term sets from Tool 1 are always matched literally; the
        scispaCy models add anything the term lists missed (synonyms, spelled-out
        forms). Returns the surface forms found, per entity class.
        """
        if not text:
            return {"target": [], "disease": []}

        targets = self._literal_matches(text, target_terms)
        diseases = self._literal_matches(text, disease_terms)
        targets += self._entities(text, self.gene_model, GENE_LABELS)
        diseases += self._entities(text, self.disease_model, DISEASE_LABELS)
        return {"target": _dedupe(targets), "disease": _dedupe(diseases)}

    # ── sentence segmentation, shared by the scorer ─────────────────────────
    def sentences(self, text: str) -> List[str]:
        """Split into sentences with spaCy when loaded, else on terminal punctuation."""
        if not text:
            return []
        nlp = _MODELS.get(self.disease_model) or _MODELS.get(self.gene_model)
        if nlp is not None:
            try:
                return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
            except Exception as exc:
                logger.debug("LitMineX: spaCy sentence split failed (%s)", exc)
        return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    def dependency_link(
        self, sentence: str, target_terms: List[str], disease_terms: List[str]
    ) -> bool:
        """
        True when the target and disease mentions are connected through the
        dependency parse — spec §4 step 2's check that the relation phrase actually
        joins the two entities. Without a parser, sentence co-occurrence stands in.
        """
        nlp = _MODELS.get(self.gene_model) or _MODELS.get(self.disease_model)
        if nlp is None:
            return True
        try:
            doc = nlp(sentence)
        except Exception:
            return True

        from DRP_Main.app.modules.literature.relation_config import LINKING_DEPS

        target_idx = _token_indices(doc, target_terms)
        disease_idx = _token_indices(doc, disease_terms)
        if not target_idx or not disease_idx:
            return False
        for t in target_idx:
            ancestors = {t} | {a.i for a in doc[t].ancestors}
            for d in disease_idx:
                if d in ancestors:
                    return True
                if {d} | {a.i for a in doc[d].ancestors} & ancestors:
                    return True
                if doc[d].dep_ in LINKING_DEPS and doc[t].dep_ in LINKING_DEPS:
                    if doc[d].head.i == doc[t].head.i:
                        return True
        return False

    # ── internals ────────────────────────────────────────────────────────────
    def _entities(self, text: str, model_name: str, labels: frozenset) -> List[str]:
        nlp = _MODELS.get(model_name)
        if nlp is None:
            return []
        try:
            doc = nlp(text[:100_000])
        except Exception as exc:
            logger.debug("LitMineX: NER failed on %s (%s)", model_name, exc)
            return []
        return [ent.text.strip() for ent in doc.ents if ent.label_ in labels and ent.text.strip()]

    @staticmethod
    def _literal_matches(text: str, terms: List[str]) -> List[str]:
        lowered = text.lower()
        return [t for t in terms if t and t.lower() in lowered]

    @staticmethod
    def _first(values: List[str]) -> Optional[str]:
        return values[0] if values else None

    @staticmethod
    def _fallback_target(text: str) -> Optional[str]:
        for raw in re.findall(r"[A-Za-z0-9\-]+", text):
            if raw.lower() in _STOPWORDS:
                continue
            if _GENE_TOKEN.match(raw):
                return raw
        return None

    @staticmethod
    def _fallback_disease(text: str) -> Optional[str]:
        lowered = text.lower()
        for hint in _DISEASE_HINTS:
            idx = lowered.find(hint)
            if idx == -1:
                continue
            # Walk left over the qualifying words ("type 2 diabetes", "breast cancer"),
            # stopping at anything that clearly is not part of the disease name: a
            # stopword, a gene symbol, or a verb linking the two concepts.
            words = re.findall(r"[A-Za-z0-9]+", text[:idx].rstrip())
            phrase: List[str] = []
            for word in reversed(words[-2:]):
                low = word.lower()
                if low != "type" and (
                    low in _STOPWORDS or low in _RELATION_VERBS or _GENE_TOKEN.match(word)
                ):
                    break
                phrase.insert(0, low)
            end = idx + len(hint)
            candidate = " ".join(phrase + [lowered[idx:end]]).strip()
            return candidate or None
        return None


def _token_indices(doc, terms: List[str]) -> List[int]:
    lowered = [t.lower() for t in terms if t]
    return [
        token.i
        for token in doc
        if any(term == token.text.lower() or term in token.text.lower() for term in lowered)
    ]


def _dedupe(values: List[str]) -> List[str]:
    seen, out = set(), []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out
