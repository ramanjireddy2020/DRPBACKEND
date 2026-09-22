"""
NovSearch Tool 1 — patent retrieval and reranking (spec §3).

Turns the normalized query into a ranked candidate patent list:

  1. title-query detection
  2. PatentsView full-text search
  3. BM25Okapi over title + abstract
  4. reciprocal rank fusion of the API order and the BM25 order (k=60)
  5. title-pin scoring when title-query mode is active
  6. MiniLM cross-encoder rerank on Databricks, normalised to 1-10
  7. per-query caching — re-running the same query reuses the cache, and asking
     for more results fetches only the difference
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.novelty import google_patents_service as pv
from DRP_Main.app.modules.novelty.databricks_clients import cross_encoder_client

logger = get_logger(__name__)

_QUESTION_STARTS = {
    "what", "how", "why", "where", "when", "does", "is", "are", "can",
    "will", "should", "would", "could", "do", "did", "has", "have",
}


# ══════════════════════════════════════════════════════════════════════════════
#  §3 step 1 — title-query detection
# ══════════════════════════════════════════════════════════════════════════════

def looks_like_title(query: str) -> bool:
    """
    Quoted text, capitalised multi-word phrases, or no question-word start flag
    the query as a title-match search rather than a general query.
    """
    text = (query or "").strip()
    if not text:
        return False
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return True
    words = text.split()
    if len(words) < 4:
        return False
    if words[0].lower() in _QUESTION_STARTS:
        return False
    if any("-" in w for w in words):
        return True
    return any(w[0].isupper() for w in words if len(w) > 2)


def _title_similarity(query: str, title: str) -> float:
    return difflib.SequenceMatcher(
        None,
        (query or "").lower().strip().strip("\"'"),
        (title or "").lower().strip(),
    ).ratio()


# ══════════════════════════════════════════════════════════════════════════════
#  §3 steps 3-4 — BM25 and reciprocal rank fusion
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Hyphenated terms are indexed whole *and* split — patent titles are full of them."""
    tokens: List[str] = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", (text or "").lower()):
        tokens.append(word)
        if "-" in word:
            tokens.extend(part for part in word.split("-") if part)
    return tokens


def _combined_text(candidate: Dict[str, Any]) -> str:
    return f"{candidate.get('title', '')} {candidate.get('abstract_snippet', '')}"


def bm25_ranking(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    index = BM25Okapi([_tokenize(_combined_text(c)) for c in candidates])
    scores = index.get_scores(_tokenize(query))
    order = np.argsort(scores)[::-1]
    return [
        {"patent_id": candidates[i]["patent_id"], "rank": rank}
        for rank, i in enumerate(order, start=1)
    ]


def reciprocal_rank_fusion(
    rankings: List[List[Dict[str, Any]]], k: Optional[int] = None
) -> List[str]:
    """Sum 1/(k+rank) per document across the supplied rankings; k defaults to 60."""
    k = settings.NOVSEARCH_RRF_K if k is None else k
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for item in ranking:
            fused[item["patent_id"]] = fused.get(item["patent_id"], 0.0) + 1.0 / (
                k + item["rank"]
            )
    return [pid for pid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


# ══════════════════════════════════════════════════════════════════════════════
#  §3 steps 5-6 — title pin + cross-encoder rerank
# ══════════════════════════════════════════════════════════════════════════════

async def cross_encoder_rerank(
    query: str, candidates: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """
    Score (query, candidate) pairs on the hosted MiniLM cross-encoder, add the
    title-pin bonus, then min-max the result onto the 1-10 scale the candidate
    table reports.
    """
    if not candidates:
        return []

    pairs = [(query, _combined_text(c)[:512]) for c in candidates]
    try:
        scores = await cross_encoder_client.score(pairs)
    except Exception as exc:  # noqa: BLE001
        # The RRF order is already a usable ranking; degrading to it beats failing
        # the whole assessment because one serving endpoint is cold.
        logger.warning("Cross-encoder rerank unavailable (%s) — keeping RRF order", exc)
        scores = [float(len(candidates) - i) for i in range(len(candidates))]

    ranked: List[Dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        base = (float(score) + 10.0) / 2.0
        ranked.append(
            {**candidate, "_raw_score": base + candidate.get("_title_pin", 0.0)}
        )
    ranked.sort(key=lambda c: c["_raw_score"], reverse=True)

    low = ranked[-1]["_raw_score"]
    span = (ranked[0]["_raw_score"] - low) or 1.0
    output: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(ranked[:top_k], start=1):
        record = {
            k: v
            for k, v in candidate.items()
            if not k.startswith("_") and k != "api_rank"
        }
        record["relevance_score"] = round(
            1.0 + 9.0 * (candidate["_raw_score"] - low) / span, 3
        )
        record["rank"] = rank
        output.append(record)
    return output


# ══════════════════════════════════════════════════════════════════════════════
#  §3 — the tool
# ══════════════════════════════════════════════════════════════════════════════

async def retrieve_candidates(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Run the full Tool 1 chain and return the top `num_results` candidates."""
    is_title = looks_like_title(query)
    fetch_size = max(
        20, num_results * max(1, settings.NOVSEARCH_SEARCH_FETCH_MULTIPLIER)
    )

    candidates = await pv.search_patents(query, size=fetch_size)
    if not candidates:
        return []

    api_ranking = [
        {"patent_id": c["patent_id"], "rank": c["api_rank"]} for c in candidates
    ]
    fused_order = reciprocal_rank_fusion([api_ranking, bm25_ranking(query, candidates)])
    by_id = {c["patent_id"]: c for c in candidates}
    ordered = [by_id[pid] for pid in fused_order if pid in by_id][:30]

    if is_title:
        pin_weight = settings.NOVSEARCH_TITLE_PIN_WEIGHT
        for candidate in ordered:
            candidate["_title_pin"] = round(
                _title_similarity(query, candidate.get("title", "")) * pin_weight, 4
            )

    return await cross_encoder_rerank(query, ordered, top_k=num_results)


# ══════════════════════════════════════════════════════════════════════════════
#  §3 step 7 — query cache
# ══════════════════════════════════════════════════════════════════════════════

class QueryCache:
    """
    Results cached per normalized query string.

    A repeat of the same query with the same or fewer results is served straight
    from the cache; asking for more fetches only the patents beyond what is held.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _key(query: str) -> str:
        return " ".join((query or "").lower().split())

    async def get(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        key = self._key(query)
        cached = self._entries.get(key, [])
        if len(cached) >= num_results:
            logger.info("NovSearch query cache hit for '%s'", key)
            return cached[:num_results]

        fresh = await retrieve_candidates(query, num_results)
        if not cached:
            self._entries[key] = fresh
            return fresh

        # Partial hit — keep what we have and append only the genuinely new ones.
        seen = {c["patent_id"] for c in cached}
        combined = cached + [c for c in fresh if c["patent_id"] not in seen]
        for rank, candidate in enumerate(combined, start=1):
            candidate["rank"] = rank
        self._entries[key] = combined
        return combined[:num_results]

    def clear(self) -> None:
        self._entries.clear()

    def keys(self) -> List[str]:
        return list(self._entries)


query_cache = QueryCache()
