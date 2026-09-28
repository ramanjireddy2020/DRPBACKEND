"""
NovSearch — the agent (spec §2, §6).

Branches on the input shape *before* Tool 1 runs, then calls the three tools in
sequence and assembles the object returned to the supervisor.

  Case A — `screensuite_carryover`: a resolved drug-target-disease combination
           arrives from session state. The query string is built from the carried
           fields; `candidate_id` / `docking_id` ride through the pipeline so the
           report traces back to the screening result that triggered it.
  Case B — `user_direct`: raw user text. Used as typed. NER-based term extraction
           (the platform's shared tagger) isolates drug/target/disease when the
           agent needs them for the title-detection check.

Both converge on one normalized shape. `candidate_id` and `docking_id` are
**absent**, not null, for a fresh query — downstream consumers branch on their
presence to know whether a report is tied to a screening result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.novelty import google_patents_service as pv
from DRP_Main.app.modules.novelty import synthesis_service
from DRP_Main.app.modules.novelty.indexing_service import index_candidates, vector_store
from DRP_Main.app.modules.novelty.retrieval_service import query_cache

logger = get_logger(__name__)

SOURCE_CARRYOVER = "screensuite_carryover"
SOURCE_USER_DIRECT = "user_direct"


class AmbiguousQueryError(ValueError):
    """
    The query names nothing searchable. The agent asks the supervisor to prompt
    the user once for clarification rather than guessing (§2 Case B).
    """


# ══════════════════════════════════════════════════════════════════════════════
#  §2 — input handling
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NormalizedInput:
    query: str
    source: str
    candidate_id: Optional[str] = None
    docking_id: Optional[str] = None
    target: Optional[str] = None
    drug: Optional[str] = None
    disease: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """The normalized shape passed to Tool 1 — omitting, not null-filling."""
        payload: Dict[str, Any] = {"query": self.query, "source": self.source}
        if self.candidate_id:
            payload["candidate_id"] = self.candidate_id
        if self.docking_id:
            payload["docking_id"] = self.docking_id
        return payload


def normalize_input(payload: Dict[str, Any]) -> NormalizedInput:
    """Branch on the input shape and produce the one normalized form Tool 1 takes."""
    source = (payload.get("source") or "").strip() or _infer_source(payload)

    if source == SOURCE_CARRYOVER:
        target = (payload.get("target") or "").strip()
        drug = (payload.get("drug") or "").strip()
        disease = (payload.get("disease") or "").strip()
        # The query string is built automatically from the carried fields — no
        # construction step and no NER, the combination is already resolved.
        query = " ".join(part for part in (target, drug, disease) if part)
        if not query:
            raise AmbiguousQueryError(
                "A ScreenSuite carry-over must name at least one of target, drug or disease."
            )
        return NormalizedInput(
            query=query,
            source=SOURCE_CARRYOVER,
            candidate_id=(payload.get("candidate_id") or "").strip() or None,
            docking_id=(payload.get("docking_id") or "").strip() or None,
            target=target or None,
            drug=drug or None,
            disease=disease or None,
        )

    query = (payload.get("query") or "").strip()
    if not query:
        raise AmbiguousQueryError("A query is required for a standalone novelty search.")

    entities = _extract_entities(query)
    if not any(entities.values()) and len(query.split()) < 3:
        raise AmbiguousQueryError(
            "The query is too ambiguous to search. Ask the user to name the drug, "
            "target or disease of interest."
        )
    # The query string is used as typed; the entities only inform title detection
    # and are reported back so the supervisor can show what was understood.
    return NormalizedInput(
        query=query,
        source=SOURCE_USER_DIRECT,
        target=entities.get("target"),
        disease=entities.get("disease"),
    )


def _infer_source(payload: Dict[str, Any]) -> str:
    """A payload carrying resolved fields is a carry-over even if unlabelled."""
    if any(payload.get(k) for k in ("candidate_id", "docking_id")):
        return SOURCE_CARRYOVER
    if payload.get("target") or payload.get("drug") or payload.get("disease"):
        return SOURCE_CARRYOVER if not payload.get("query") else SOURCE_USER_DIRECT
    return SOURCE_USER_DIRECT


def _extract_entities(query: str) -> Dict[str, Optional[str]]:
    """
    Reuse the platform's shared target/disease entity tagger. It degrades to a
    rule-based tagger when the scispaCy models are absent, so this never fails
    the request — recall just drops.
    """
    try:
        from DRP_Main.app.modules.literature.ner_service import NERService

        return NERService().extract_query_entities(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NER unavailable for NovSearch term extraction: %s", exc)
        return {"target": None, "disease": None}


# ══════════════════════════════════════════════════════════════════════════════
#  §6 — the agent
# ══════════════════════════════════════════════════════════════════════════════

Progress = Callable[[str], None]


async def run_novsearch(
    payload: Dict[str, Any],
    num_results: int = 5,
    progress: Optional[Progress] = None,
) -> Dict[str, Any]:
    """
    Run Tool 1 → Tool 2 → Tool 3 and return the object the supervisor consumes.

    `progress` is an optional callback so a job runner can stream step messages.
    """
    emit = progress or (lambda _msg: None)
    normalized = normalize_input(payload)

    # Names the corpus actually searched. The label said USPTO PatentsView long
    # after retrieval moved — first to SerpAPI, now to Europe PMC's SureChEMBL
    # patent set — and a progress line naming the wrong source sends anyone
    # debugging a thin result set to the wrong place.
    emit(f"Searching patents (Europe PMC / SureChEMBL) for '{normalized.query}'...")
    candidates = await query_cache.get(normalized.query, num_results)
    if not candidates:
        raise AmbiguousQueryError(
            f"No patents found for '{normalized.query}'. Try a different search term."
        )

    emit(f"Indexing {len(candidates)} patents (fetch → chunk → embed → store)...")
    indexing = await index_candidates(candidates)

    emit("Synthesising the query-scoped novelty report...")
    titles = {c["patent_id"]: c.get("title", "") for c in candidates}
    report = await synthesis_service.synthesize_report(
        normalized.query, indexing.indexed_patent_ids, titles
    )
    report["total_chunks"] = indexing.total_chunks or report.get("total_chunks", 0)

    return _assemble(normalized, candidates, indexing, report)


def _assemble(
    normalized: NormalizedInput,
    candidates: List[Dict[str, Any]],
    indexing: Any,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """The §6 final object. candidate_id / docking_id appear only for a carry-over."""
    obj: Dict[str, Any] = {
        "module": "novsearch",
        "input_source": normalized.source,
        "query": normalized.query,
    }
    if normalized.candidate_id:
        obj["candidate_id"] = normalized.candidate_id
    if normalized.docking_id:
        obj["docking_id"] = normalized.docking_id

    obj["report"] = report
    obj["patents_table"] = [
        {
            "patent_id": pv.display_patent_id(c["patent_id"]),
            "title": c.get("title", ""),
            "abstract_snippet": c.get("abstract_snippet", ""),
            "assignee": c.get("assignee", ""),
            "filing_date": c.get("filing_date"),
            "publication_date": c.get("publication_date"),
            "relevance_score": c.get("relevance_score"),
            "rank": c.get("rank"),
        }
        for c in candidates
    ]
    obj["indexing"] = indexing.as_dict()
    obj["session_state_update"] = {
        "novsearch_last_query": normalized.query,
        "novsearch_indexed_patents": [
            pv.display_patent_id(p) for p in indexing.indexed_patent_ids
        ],
    }
    return obj


async def answer_followup(
    question: str,
    patent_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """QA in single-patent, multiple-patent or all-patent scope (§5)."""
    if not (question or "").strip():
        raise AmbiguousQueryError("A question is required.")
    return await synthesis_service.answer_question(question, patent_ids, top_k)


def status() -> Dict[str, Any]:
    """What is currently indexed and which backends are wired up."""
    from DRP_Main.app.core.config import settings

    return {
        "vector_store": vector_store.statistics(),
        "query_cache": {"cached_queries": query_cache.keys()},
        "retrieval_backend": "google_patents_serpapi",
        "embedding_backend": f"databricks_serving/{settings.NOVSEARCH_EMBEDDING_ENDPOINT}",
        "reranker_backend": f"databricks_serving/{settings.NOVSEARCH_CROSS_ENCODER_ENDPOINT}",
        "llm_backend": settings.NOVSEARCH_LLM_NAME,
        "llm_fallback": f"groq-{settings.GROQ_MODEL}",
    }
