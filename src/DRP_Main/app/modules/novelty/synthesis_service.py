"""
NovSearch Tool 3 — synthesis and QA (spec §5).

Report synthesis retrieves top-matching chunks per indexed patent, assembles one
per-patent-labelled context, and makes a single call to the patent-domain LLM.
The prompt requires the answer to address the specific query normalized in §2 —
not to summarise the patents generally — and every claim to be tied to a patent ID.

QA runs in three modes: one patent, a researcher-selected subset, or every
currently indexed patent.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.novelty import google_patents_service as pv
from DRP_Main.app.modules.novelty.databricks_clients import llm_client
from DRP_Main.app.modules.novelty.indexing_service import vector_store

logger = get_logger(__name__)

_METADATA_KEYWORDS = (
    "own", "owner", "owns", "assignee", "assigned to", "inventor", "inventors",
    "invented by", "who made", "filed by", "filing date", "publication date",
    "applicant", "who holds",
)

_CLAIM_KEYWORDS = ("claim", "claims", "claiming", "claimed")

_FORMAT_RULES = """
RESPONSE FORMAT RULES:
- Answer ONLY what the question asks. Simple questions in 1-2 sentences.
- For complex questions use 150-300 words.
- Emphasise key terms in CAPS (e.g. FEDRATINIB, JAK2, THROMBOCYTOSIS).
- Bullets only for list-like content; paragraphs for explanation.
- Plain text only: no asterisks, no markdown, no chunk or score references.
- Cite the specific patent ID for every claim you make.
- If the information is absent from the supplied content, say so in one sentence.
"""

_REPORT_SECTIONS = (
    "AGENT ANALYSIS",
    "INDEPENDENT CLAIMS SUMMARY",
    "NOVELTY ASSESSMENT",
    "FREEDOM-TO-OPERATE RISK",
    "KEY FINDINGS",
    "RECOMMENDED NEXT STEPS",
    "GAPS IN CURRENT ANALYSIS",
)

_SPLIT_MARKER = "RECOMMENDED NEXT STEPS"


# ══════════════════════════════════════════════════════════════════════════════
#  Context assembly
# ══════════════════════════════════════════════════════════════════════════════

def _label(chunk: Dict[str, Any]) -> str:
    section = (chunk.get("section_type") or "").upper()
    if chunk.get("is_claim") and chunk.get("claim_number") is not None:
        return f"CLAIM #{chunk['claim_number']} ({section})"
    return section


def _patent_block(patent_id: str, title: str, chunks: List[Dict[str, Any]]) -> str:
    header = f"\n{'=' * 60}\nPATENT: {pv.display_patent_id(patent_id)}\nTITLE: {title}\n{'=' * 60}\n"
    body = "".join(f"[{_label(c)}]\n{c.get('chunk_text', '')}\n\n" for c in chunks)
    return header + body


async def _gather_context(
    question: str,
    patent_ids: List[str],
    titles: Optional[Dict[str, str]] = None,
    top_k: Optional[int] = None,
) -> Tuple[str, List[str], int]:
    """Retrieve and label top chunks per patent. Returns (context, used_ids, count)."""
    titles = titles or {}
    blocks: List[str] = []
    used: List[str] = []
    total = 0
    for pid in patent_ids:
        chunks = await vector_store.search_by_patent(question, pid, top_k=top_k)
        if not chunks:
            continue
        blocks.append(_patent_block(pid, titles.get(pid, pv.display_patent_id(pid)), chunks))
        used.append(pid)
        total += len(chunks)
    return "\n".join(blocks), used, total


# ══════════════════════════════════════════════════════════════════════════════
#  §5 — report synthesis
# ══════════════════════════════════════════════════════════════════════════════

def _report_prompt(query: str, context: str, patent_count: int) -> str:
    return f"""You are a patent attorney analysing USPTO patents for a drug discovery team.

THE SPECIFIC QUESTION YOU MUST ANSWER: "{query}"

Below is content retrieved from {patent_count} USPTO patent(s), labelled by patent ID.

PATENT CONTENT:
{context}

Produce a structured report with these EXACT section headings:

AGENT ANALYSIS:
Three to four sentences answering the specific question above directly — not a general
summary of the patents. Reference specific patent IDs.

INDEPENDENT CLAIMS SUMMARY:
The independent claims across these patents most relevant to the question, each tied to
its patent ID and claim number.

NOVELTY ASSESSMENT:
State High Novelty, Medium Novelty or Low Novelty, with one paragraph of justification
citing specific patents and claims.

FREEDOM-TO-OPERATE RISK:
State High Risk, Medium Risk or Low Risk, with one paragraph of justification.

KEY FINDINGS:
Four to six specific findings drawn from the patent content, each with its patent ID.

RECOMMENDED NEXT STEPS:
Five to seven actionable next steps for the research team.

GAPS IN CURRENT ANALYSIS:
What is missing, uncertain, or needs further investigation.

RULES:
- Answer the specific question. Do not summarise the patents generally.
- Every claim you state must name the patent ID it comes from.
- Plain text only. No asterisks, no markdown.
- Capitalise drug names, protein targets and mechanisms.
- If a section cannot be answered from this content, say why in one sentence.
"""


async def synthesize_report(
    query: str,
    patent_ids: List[str],
    titles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    One call to the patent-domain LLM producing the query-scoped novelty report.

    Returns the §5 output shape: `agent_answer`, `recommendations`, `patents_used`,
    `total_patents`, `total_chunks`, `model_used`.
    """
    if not patent_ids:
        return {
            "agent_answer": "No patents were indexed for analysis.",
            "recommendations": "Could not generate recommendations — no patents indexed.",
            "patents_used": [],
            "total_patents": 0,
            "total_chunks": 0,
            "model_used": None,
        }

    context, used, total_chunks = await _gather_context(query, patent_ids, titles)
    if not context:
        return {
            "agent_answer": "No relevant content could be retrieved from the indexed patents.",
            "recommendations": "Insufficient evidence for recommendations.",
            "patents_used": [],
            "total_patents": 0,
            "total_chunks": 0,
            "model_used": None,
        }

    prompt = _report_prompt(query, context, len(used))
    try:
        text, model_used = await llm_client.generate(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("NovSearch report synthesis failed")
        return {
            "agent_answer": f"Analysis could not be completed: {exc}",
            "recommendations": f"Recommendations unavailable: {exc}",
            "patents_used": [pv.display_patent_id(p) for p in used],
            "total_patents": len(used),
            "total_chunks": total_chunks,
            "model_used": None,
        }

    agent_answer, recommendations = _split_report(text)
    return {
        "agent_answer": agent_answer,
        "recommendations": recommendations,
        "patents_used": [pv.display_patent_id(p) for p in used],
        "total_patents": len(used),
        "total_chunks": total_chunks,
        "model_used": model_used,
    }


def _split_report(text: str) -> Tuple[str, str]:
    """
    `agent_answer` is Analysis → Key Findings; `recommendations` is Next Steps →
    Gaps. The split is on the RECOMMENDED NEXT STEPS heading; if the model omits
    it, the whole body is the answer rather than being silently truncated.
    """
    match = re.search(rf"^\s*{_SPLIT_MARKER}", text, re.MULTILINE | re.IGNORECASE)
    if not match:
        return text.strip(), ""
    return text[: match.start()].strip(), text[match.start() :].strip()


# ══════════════════════════════════════════════════════════════════════════════
#  §5 — QA, three modes
# ══════════════════════════════════════════════════════════════════════════════

async def answer_question(
    question: str,
    patent_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Answer a follow-up question.

    Scope follows what the caller supplies: one patent id → `single_patent`, a
    selected subset → `multiple_patent`, nothing → every indexed patent
    (`multi_patent`). The mode names are the ones the spec's QA output declares.
    """
    indexed = vector_store.indexed_patent_ids()
    if not indexed:
        return {
            "answer": "No patents are indexed yet. Run a novelty search first.",
            "mode": "multi_patent",
            "patent_ids_used": [],
            "chunks_used": 0,
        }

    requested = [pv.normalize_patent_id(p) for p in (patent_ids or []) if p]
    unknown = [p for p in requested if p not in indexed]
    if unknown:
        return {
            "answer": (
                "These patents are not indexed: "
                f"{', '.join(pv.display_patent_id(p) for p in unknown)}. "
                "Run a novelty search that covers them first."
            ),
            "mode": "single_patent" if len(requested) == 1 else "multiple_patent",
            "patent_ids_used": [],
            "chunks_used": 0,
        }

    if len(requested) == 1:
        return await _answer_single(question, requested[0], top_k)
    scope = requested or indexed
    mode = "multiple_patent" if requested else "multi_patent"
    return await _answer_many(question, scope, mode, top_k)


async def _answer_single(
    question: str, patent_id: str, top_k: Optional[int]
) -> Dict[str, Any]:
    lowered = question.lower()
    is_metadata = any(kw in lowered for kw in _METADATA_KEYWORDS)
    is_claims = any(kw in lowered for kw in _CLAIM_KEYWORDS)

    chunks = await vector_store.search_by_patent(question, patent_id, top_k=top_k)
    if is_metadata:
        # Metadata-style questions are answered from metadata chunks directly.
        meta = await vector_store.metadata_chunks(patent_id)
        chunks = meta + [c for c in chunks if c.get("section_type") != "metadata"]
    elif is_claims:
        # Claims questions prioritise claim-tagged chunks.
        chunks = [c for c in chunks if c.get("is_claim")] + [
            c for c in chunks if not c.get("is_claim")
        ]

    if not chunks:
        return {
            "answer": f"No relevant content found in patent {pv.display_patent_id(patent_id)}.",
            "mode": "single_patent",
            "patent_ids_used": [],
            "chunks_used": 0,
        }

    context = "".join(f"[{_label(c)}]\n{c.get('chunk_text', '')}\n\n" for c in chunks)
    display = pv.display_patent_id(patent_id)
    if is_metadata:
        prompt = (
            f"You are a patent analyst. Answer this question about patent {display} using "
            f"only the metadata below. Answer in 1-2 sentences, using the exact names and "
            f"dates given.\n\nMETADATA:\n{context}\nQUESTION: {question}"
        )
    else:
        prompt = (
            f"You are a patent analyst. Patent: {display}\n{_FORMAT_RULES}\n\n"
            f"PATENT CONTENT:\n{context}\nQUESTION: {question}"
        )

    answer, _ = await _safe_generate(prompt)
    return {
        "answer": answer,
        "mode": "single_patent",
        "patent_ids_used": [display],
        "chunks_used": len(chunks),
    }


async def _answer_many(
    question: str, patent_ids: List[str], mode: str, top_k: Optional[int]
) -> Dict[str, Any]:
    context, used, total = await _gather_context(question, patent_ids, top_k=top_k)
    if not context:
        return {
            "answer": "No relevant content found in the selected patents.",
            "mode": mode,
            "patent_ids_used": [],
            "chunks_used": 0,
        }

    prompt = (
        f"You are a patent analyst. Using the content below from {len(used)} USPTO "
        f"patent(s), answer this question in a single unified answer, citing whichever "
        f"specific patents are actually relevant.\n{_FORMAT_RULES}\n\n"
        f"{context}\n\nQUESTION: {question}"
    )
    answer, _ = await _safe_generate(prompt)
    return {
        "answer": answer,
        "mode": mode,
        "patent_ids_used": [pv.display_patent_id(p) for p in used],
        "chunks_used": total,
    }


async def _safe_generate(prompt: str) -> Tuple[str, Optional[str]]:
    try:
        return await llm_client.generate(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("NovSearch QA generation failed")
        return f"Could not answer: {exc}", None
