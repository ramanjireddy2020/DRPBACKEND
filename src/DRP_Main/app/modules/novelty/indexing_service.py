"""
NovSearch Tool 2 — indexing (spec §4).

Fetch full structured patent content, chunk it, embed it, and store it, skipping
anything already indexed:

  1. deduplication check against Databricks Vector Search — an already-indexed
     patent is skipped entirely, no fetch and no re-embedding
  2. structured content fetch from PatentsView (Patent + Claims endpoints)
  3. section-aware, weighted chunking, with claim chunks tagged by claim number
  4. BGE-large embeddings via Databricks Model Serving
  5. upsert into Databricks Vector Search
  6. cap enforcement — the oldest indexed patent is evicted past the cap

`databricks.vector_search` is imported lazily, like PyMOL elsewhere in this repo:
a workspace without Vector Search must fail one job with a readable message, not
break API import.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.novelty import google_patents_service as pv
from DRP_Main.app.modules.novelty.databricks_clients import embedding_client

logger = get_logger(__name__)


class VectorStoreError(RuntimeError):
    """Databricks Vector Search is unavailable or rejected an operation."""


# ══════════════════════════════════════════════════════════════════════════════
#  §4 step 3 — section-aware chunking
# ══════════════════════════════════════════════════════════════════════════════

SECTION_WEIGHTS: Dict[str, float] = {
    "independent_claims": 1.00,
    "abstract": 0.95,
    "summary": 0.90,
    "dependent_claims": 0.85,
    "metadata": 0.85,
    "description": 0.80,
    "background": 0.70,
}


class PatentChunker:
    """Splits a structured patent record into weighted, section-tagged chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, max_chunks: Optional[int] = None) -> List[str]:
        words = (text or "").split()
        if not words:
            return []
        stride = max(1, self.chunk_size - self.overlap)
        chunks: List[str] = []
        for start in range(0, len(words), stride):
            chunks.append(" ".join(words[start : start + self.chunk_size]))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def chunks_for(self, patent: Dict[str, Any]) -> List[Dict[str, Any]]:
        pid = patent.get("patent_id", "UNKNOWN")
        chunks: List[Dict[str, Any]] = []

        def add(section: str, text: str, index: int, **extra: Any) -> None:
            if not (text or "").strip():
                return
            chunks.append(
                {
                    "chunk_id": f"{pid}_{section}_{index}",
                    "patent_id": pid,
                    "section_type": section,
                    "chunk_text": text,
                    "chunk_index": index,
                    "section_weight": SECTION_WEIGHTS.get(section, 0.75),
                    "is_claim": extra.get("is_claim", False),
                    "claim_number": extra.get("claim_number"),
                    "claim_type": extra.get("claim_type"),
                    "depends_on": extra.get("depends_on") or [],
                }
            )

        add("abstract", patent.get("abstract", ""), 0)
        # Claim chunks are tagged by claim number so a specific claim is directly
        # retrievable and citable in the report (§4 step 3, §5).
        for claim in patent.get("independent_claims", []):
            add(
                "independent_claims",
                claim["text"],
                claim["number"],
                is_claim=True,
                claim_type="independent",
                claim_number=claim["number"],
            )
        for claim in patent.get("dependent_claims", []):
            add(
                "dependent_claims",
                claim["text"],
                claim["number"],
                is_claim=True,
                claim_type="dependent",
                claim_number=claim["number"],
                depends_on=claim.get("depends_on", []),
            )
        for i, text in enumerate(self._split(patent.get("summary", ""), max_chunks=3)):
            add("summary", text, i)
        for i, text in enumerate(self._split(patent.get("description", ""), max_chunks=10)):
            add("description", text, i)
        for i, text in enumerate(self._split(patent.get("background", ""), max_chunks=3)):
            add("background", text, i)

        meta_parts = []
        if patent.get("assignee"):
            meta_parts.append(f"Assignee: {patent['assignee']}")
        if patent.get("inventors"):
            meta_parts.append(f"Inventors: {', '.join(patent['inventors'])}")
        if patent.get("filing_date"):
            meta_parts.append(f"Filing Date: {patent['filing_date']}")
        if patent.get("publication_date"):
            meta_parts.append(f"Publication Date: {patent['publication_date']}")
        if patent.get("title"):
            meta_parts.append(f"Title: {patent['title']}")
        add("metadata", ". ".join(meta_parts), 0)
        return chunks


chunker = PatentChunker()


# ══════════════════════════════════════════════════════════════════════════════
#  §4 steps 1, 5, 6 — Databricks Vector Search store
# ══════════════════════════════════════════════════════════════════════════════

_PAYLOAD_COLUMNS = [
    "chunk_id",
    "patent_id",
    "section_type",
    "chunk_text",
    "chunk_index",
    "section_weight",
    "is_claim",
    "claim_number",
    "claim_type",
    "depends_on",
]


class VectorStore:
    """
    Databricks Vector Search direct-access index.

    Vectors are supplied by us (BGE-large on Model Serving) rather than computed
    by a delta-sync index, because indexing is driven by an API request, not by a
    table write.

    `_patent_order` tracks insertion order for the cap in §4 step 6. It is
    rebuilt lazily from the index on first use, so a warm restart still knows
    what is already stored and the dedup check in step 1 stays correct.
    """

    def __init__(self) -> None:
        self._index = None
        self._patent_order: "OrderedDict[str, bool]" = OrderedDict()
        self._loaded = False

    # ── connection ───────────────────────────────────────────────────────────
    def _get_index(self):
        if self._index is not None:
            return self._index
        try:
            from databricks.vector_search.client import VectorSearchClient
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise VectorStoreError(
                "databricks-vectorsearch is not installed — NovSearch stores patent "
                "chunks in Databricks Vector Search."
            ) from exc
        # Unlike WorkspaceClient(), VectorSearchClient() has no ambient-credential
        # fallback — it always needs an explicit PAT or service-principal
        # client id/secret. A deployed Databricks App auto-injects its own
        # service principal's OAuth M2M credentials as DATABRICKS_CLIENT_ID /
        # DATABRICKS_CLIENT_SECRET precisely for cases like this one.
        import os

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        if settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN:
            client = VectorSearchClient(
                workspace_url=settings.DATABRICKS_HOST,
                personal_access_token=settings.DATABRICKS_TOKEN,
                disable_notice=True,
            )
        elif client_id and client_secret:
            client = VectorSearchClient(
                workspace_url=settings.DATABRICKS_HOST or None,
                service_principal_client_id=client_id,
                service_principal_client_secret=client_secret,
                disable_notice=True,
            )
        else:
            raise VectorStoreError(
                "No Vector Search credentials available — need DATABRICKS_HOST/"
                "DATABRICKS_TOKEN, or DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET "
                "(auto-injected inside a deployed Databricks App)."
            )
        self._index = client.get_index(
            endpoint_name=settings.NOVSEARCH_VS_ENDPOINT,
            index_name=settings.NOVSEARCH_VS_INDEX,
        )
        logger.info("NovSearch vector index ready (%s)", settings.NOVSEARCH_VS_INDEX)
        return self._index

    def _ensure_loaded(self) -> None:
        """Rebuild `_patent_order` from the index once per process."""
        if self._loaded:
            return
        self._loaded = True
        try:
            index = self._get_index()
            # A zero vector retrieves an arbitrary page; we only need patent ids.
            probe = [0.0] * settings.NOVSEARCH_EMBEDDING_DIM
            probe[0] = 1.0
            response = index.similarity_search(
                query_vector=probe,
                columns=["patent_id"],
                num_results=2000,
            )
            for row in _rows(response, ["patent_id"]):
                pid = row.get("patent_id")
                if pid:
                    self._patent_order.setdefault(pid, True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enumerate indexed patents: %s", exc)

    # ── §4 step 1: deduplication ─────────────────────────────────────────────
    def indexed_patent_ids(self) -> List[str]:
        self._ensure_loaded()
        return list(self._patent_order)

    def is_indexed(self, patent_id: str) -> bool:
        self._ensure_loaded()
        return pv.normalize_patent_id(patent_id) in self._patent_order

    # ── §4 steps 4-6: embed, upsert, cap ─────────────────────────────────────
    async def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        vectors = await embedding_client.embed_passages([c["chunk_text"] for c in chunks])

        rows: List[Dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            rows.append(
                {
                    "id": _row_id(chunk["chunk_id"]),
                    "chunk_id": chunk["chunk_id"],
                    "patent_id": chunk["patent_id"],
                    "section_type": chunk["section_type"],
                    "chunk_text": chunk["chunk_text"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "section_weight": float(chunk["section_weight"]),
                    "is_claim": bool(chunk["is_claim"]),
                    "claim_number": chunk["claim_number"],
                    "claim_type": chunk["claim_type"],
                    "depends_on": ",".join(str(d) for d in chunk["depends_on"]),
                    "embedding": vector,
                }
            )

        index = self._get_index()
        batch = max(1, settings.NOVSEARCH_EMBED_BATCH)
        for start in range(0, len(rows), batch):
            index.upsert(rows[start : start + batch])

        self._ensure_loaded()
        for chunk in chunks:
            self._patent_order.setdefault(chunk["patent_id"], True)
        self._enforce_cap()
        logger.info("Upserted %d chunks into Databricks Vector Search", len(rows))
        return len(rows)

    def _enforce_cap(self) -> None:
        while len(self._patent_order) > settings.NOVSEARCH_MAX_PATENTS:
            oldest = next(iter(self._patent_order))
            logger.info("Evicting oldest indexed patent %s (store cap reached)", oldest)
            self.delete_patent(oldest)

    def delete_patent(self, patent_id: str) -> None:
        pid = pv.normalize_patent_id(patent_id)
        try:
            index = self._get_index()
            keys = [
                row["chunk_id"]
                for row in self._search(
                    [0.0] * settings.NOVSEARCH_EMBEDDING_DIM,
                    patent_ids=[pid],
                    columns=["chunk_id"],
                    num_results=1000,
                )
            ]
            if keys:
                index.delete([_row_id(k) for k in keys])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector Search delete failed for %s: %s", pid, exc)
        self._patent_order.pop(pid, None)

    # ── retrieval (used by Tool 3) ───────────────────────────────────────────
    def _search(
        self,
        query_vector: List[float],
        patent_ids: List[str],
        columns: Optional[List[str]] = None,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        index = self._get_index()
        columns = columns or _PAYLOAD_COLUMNS
        kwargs: Dict[str, Any] = {
            "query_vector": query_vector,
            "columns": columns,
            "num_results": num_results,
        }
        if patent_ids:
            kwargs["filters"] = {"patent_id": patent_ids}
        response = index.similarity_search(**kwargs)
        return _rows(response, columns)

    async def search_by_patent(
        self, query: str, patent_id: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Top-matching chunks for one patent, ordered by section-weighted score so
        an independent claim outranks an equally similar background paragraph.
        """
        pid = pv.normalize_patent_id(patent_id)
        if not self.is_indexed(pid):
            return []
        top_k = top_k or settings.NOVSEARCH_TOP_K_CHUNKS
        vector = await embedding_client.embed_query(query)
        rows = self._search(vector, [pid], num_results=top_k)
        for row in rows:
            weight = float(row.get("section_weight") or 0.75)
            row["weighted_score"] = float(row.get("similarity_score") or 0.0) * weight
        rows.sort(key=lambda r: r["weighted_score"], reverse=True)
        return rows

    async def metadata_chunks(self, patent_id: str) -> List[Dict[str, Any]]:
        """Metadata chunks for ownership / filing-date style questions (§5 QA)."""
        pid = pv.normalize_patent_id(patent_id)
        vector = await embedding_client.embed_query(
            "assignee inventors filing date publication date"
        )
        rows = self._search(vector, [pid], num_results=5)
        return [r for r in rows if r.get("section_type") == "metadata"]

    def statistics(self) -> Dict[str, Any]:
        return {
            "total_patents": len(self.indexed_patent_ids()),
            "patent_ids": self.indexed_patent_ids(),
            "cap": settings.NOVSEARCH_MAX_PATENTS,
            "backend": "databricks_vector_search",
            "index": settings.NOVSEARCH_VS_INDEX,
        }


def _row_id(chunk_id: str) -> str:
    """Stable primary key — re-indexing a patent overwrites rather than duplicates."""
    return hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()


def _rows(response: Any, columns: List[str]) -> List[Dict[str, Any]]:
    """
    Vector Search answers `{"result": {"data_array": [[col...], ...]}}` with the
    manifest naming the columns; the similarity score is appended as the last
    element of each row.
    """
    if not isinstance(response, dict):
        return []
    result = response.get("result") or {}
    data = result.get("data_array") or []
    manifest = response.get("manifest") or {}
    names = [c.get("name") for c in manifest.get("columns") or []] or list(columns)

    rows: List[Dict[str, Any]] = []
    for values in data:
        row = dict(zip(names, values))
        if len(values) > len(names):
            row["similarity_score"] = float(values[-1])
        else:
            row.setdefault("similarity_score", float(row.pop("score", 0.0) or 0.0))
        rows.append(row)
    return rows


vector_store = VectorStore()


# ══════════════════════════════════════════════════════════════════════════════
#  §4 — the tool
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndexingResult:
    indexed_patent_ids: List[str] = field(default_factory=list)
    newly_indexed: List[str] = field(default_factory=list)
    skipped_already_indexed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    total_chunks: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "indexed_patent_ids": self.indexed_patent_ids,
            "newly_indexed": self.newly_indexed,
            "skipped_already_indexed": self.skipped_already_indexed,
            "total_chunks": self.total_chunks,
        }


async def index_candidates(candidates: List[Dict[str, Any]]) -> IndexingResult:
    """Index the Tool 1 candidate list, skipping patents already in the store."""
    result = IndexingResult()
    wanted = [c["patent_id"] for c in candidates if c.get("patent_id")]
    already = set(vector_store.indexed_patent_ids())

    to_index = [pid for pid in wanted if pid not in already]
    result.skipped_already_indexed = [pid for pid in wanted if pid in already]

    for pid in to_index:
        try:
            patent = await pv.fetch_patent_content(pid)
        except Exception as exc:  # noqa: BLE001 — one bad patent must not sink the run
            logger.warning("Content fetch failed for %s: %s", pid, exc)
            result.failed.append(pid)
            continue
        if not patent:
            result.failed.append(pid)
            continue
        chunks = chunker.chunks_for(patent)
        if not chunks:
            result.failed.append(pid)
            continue
        result.total_chunks += await vector_store.add_chunks(chunks)
        result.newly_indexed.append(pid)

    result.indexed_patent_ids = [
        pid for pid in wanted if pid in set(result.newly_indexed) | already
    ]
    return result
