"""
LitMineX semantic similarity (spec §4 step 5).

Embeds the original query and every relation-bearing sentence with SapBERT via
sentence-transformers, then takes cosine similarity. The model is loaded lazily and
once per process; if it cannot be loaded (no weights cached, no network, CPU-only
build missing torch) the service degrades to a lexical Jaccard similarity so scoring
still produces a defensible number instead of failing the request.
"""
from __future__ import annotations

import re
import threading
from typing import List, Optional

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingService:
    _lock = threading.Lock()
    _model = None
    _load_failed = False

    def __init__(self, model_name: Optional[str] = None, enabled: Optional[bool] = None) -> None:
        self.model_name = model_name or settings.LITMINEX_EMBEDDING_MODEL
        self.enabled = settings.LITMINEX_ENABLE_EMBEDDINGS if enabled is None else enabled

    @property
    def backend(self) -> str:
        """Which similarity path is actually in use — reported in the scoring output."""
        return "sapbert" if self._get_model() is not None else "lexical"

    def _get_model(self):
        if not self.enabled or EmbeddingService._load_failed:
            return None
        if EmbeddingService._model is not None:
            return EmbeddingService._model
        with EmbeddingService._lock:
            if EmbeddingService._model is not None:
                return EmbeddingService._model
            if EmbeddingService._load_failed:
                return None
            try:
                from sentence_transformers import SentenceTransformer  # heavy: lazy
                EmbeddingService._model = SentenceTransformer(self.model_name)
                logger.info("LitMineX: loaded embedding model %s", self.model_name)
            except Exception as exc:
                logger.warning(
                    "LitMineX: embedding model %s unavailable (%s); "
                    "using lexical similarity instead",
                    self.model_name,
                    exc,
                )
                EmbeddingService._load_failed = True
                return None
            return EmbeddingService._model

    def similarities(self, query: str, sentences: List[str]) -> List[float]:
        """Cosine similarity of `query` against each sentence, clamped to [0, 1]."""
        if not sentences:
            return []
        model = self._get_model()
        if model is None:
            return [_lexical_similarity(query, s) for s in sentences]
        try:
            from sentence_transformers import util

            embeddings = model.encode(
                [query, *sentences], convert_to_tensor=True, normalize_embeddings=True
            )
            scores = util.cos_sim(embeddings[0], embeddings[1:])[0]
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except Exception as exc:
            logger.warning("LitMineX: embedding similarity failed (%s); using lexical", exc)
            return [_lexical_similarity(query, s) for s in sentences]


def _lexical_similarity(query: str, sentence: str) -> float:
    """Token Jaccard — the deterministic stand-in when SapBERT is unavailable."""
    a = set(_TOKEN.findall((query or "").lower()))
    b = set(_TOKEN.findall((sentence or "").lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
