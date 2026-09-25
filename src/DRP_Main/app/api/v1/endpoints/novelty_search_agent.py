"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NOVELTY SEARCH AI AGENT  —  novelty_search_agent.py                ║
║                                                                              ║
║  Single-flow patent novelty analysis pipeline.                              ║
║  Google Patents only.                                                        ║
║  BAAI/bge-small-en-v1.5 via HuggingFace Inference API.                      ║
║  Qdrant Cloud for vector storage and search.                                 ║
║  ONE Gemini call per run.                                                    ║
║                                                                              ║
║  Pipeline (all steps except the last are Gemini-free):                      ║
║    1. SerpAPI → Google Patents search                                        ║
║    2. BM25 + RRF + cross-encoder rerank                                     ║
║    3. HTML fetch + BeautifulSoup parse                                       ║
║    4. Section-aware chunking                                                 ║
║    5. HuggingFace Inference API embeddings → Qdrant Cloud                   ║
║    6. Vector search → retrieve top chunks per patent                        ║
║    7. ONE Gemini call → novelty assessment + claims summary +               ║
║                          FTO risk + recommendations                         ║
║                                                                              ║
║  Follow-up Q&A:                                                             ║
║    Ask (single patent) → one Gemini call via RAG                            ║
║    Ask All Patents     → one Gemini synthesis call                          ║
║                                                                              ║
║  Cache Management:                                                           ║
║    - Startup: full wipe (Qdrant collection, memory, DB history)             ║
║    - Cap: 20 patents max in Qdrant collection, evict oldest when exceeded   ║
║    - Same query + more results: retrieve cached, fetch only difference      ║
║    - Patent already in Qdrant: skip HTML fetch/parse/chunk/embed entirely   ║
║                                                                              ║
║  Usage Tracking  (GET /agent/usage):                                        ║
║    Gemini  — input tokens, output tokens, cost per call, cumulative         ║
║    SerpAPI — calls made, cost per call, cumulative                          ║
║    Pricing — Gemini 2.5 Flash: $0.075/1M input, $0.30/1M output            ║
║              SerpAPI: $0.015 per search (standard plan)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import csv
import difflib
import html as html_module
import io
import json
import logging
import os
import re
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

import httpx
import numpy as np
from bs4 import BeautifulSoup, GuessedAtParserWarning
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from DRP_Main.app.core.llm import llm_client
from DRP_Main.app.core.config import settings

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# ══════════════════════════════════════════════════════════════════════════════
#  §1  CONFIG & LOGGING
# ══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger("novelty_agent")

load_dotenv()
SERPAPI_API_KEY   = os.getenv("SERPAPI_API_KEY",   "")
HF_TOKEN          = os.getenv("HF_TOKEN",          "")
QDRANT_URL        = os.getenv("QDRANT_URL",        "")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY",    "")
DATABASE_URL      = os.getenv("DATABASE_URL",      "sqlite:///./noveltysearch.db")

MODEL_NAME              = settings.DATABRICKS_LLM_ENDPOINT
HF_EMBEDDING_MODEL      = "BAAI/bge-small-en-v1.5"
QDRANT_COLLECTION_NAME  = "patent_chunks"
QDRANT_VECTOR_SIZE      = 384          # bge-small-en-v1.5 output dimension
MAX_PATENTS_IN_STORE    = 20
GEMINI_RETRY_ATTEMPTS   = 3

# ── Pricing ───────────────────────────────────────────────────────────────────
# Databricks Foundation Model API pay-per-token rate for
# databricks-meta-llama-3-3-70b-instruct (see workspace pricing page for current values).
GEMINI_INPUT_COST_PER_1M  = 0.0
GEMINI_OUTPUT_COST_PER_1M = 0.0
SERPAPI_COST_PER_SEARCH   = 0.015

# ══════════════════════════════════════════════════════════════════════════════
#  §2  USAGE TRACKER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeminiCallRecord:
    timestamp:    str
    call_type:    str
    query:        str
    input_tokens: int
    output_tokens: int
    input_cost:   float
    output_cost:  float
    total_cost:   float
    retries:      int
    success:      bool


@dataclass
class SerpAPICallRecord:
    timestamp:  str
    query:      str
    call_type:  str
    results_returned: int
    cost:       float


@dataclass
class UsageTracker:
    gemini_calls:          List[GeminiCallRecord] = field(default_factory=list)
    gemini_total_input:    int   = 0
    gemini_total_output:   int   = 0
    gemini_total_cost:     float = 0.0
    gemini_total_retries:  int   = 0
    gemini_failed_calls:   int   = 0

    serp_calls:            List[SerpAPICallRecord] = field(default_factory=list)
    serp_total_calls:      int   = 0
    serp_total_cost:       float = 0.0

    session_start:         str   = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def record_gemini(self, call_type, query, input_tokens, output_tokens,
                      retries=0, success=True):
        input_cost  = (input_tokens  / 1_000_000) * GEMINI_INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M
        total_cost  = input_cost + output_cost
        timestamp   = datetime.utcnow().isoformat()
        rec = GeminiCallRecord(
            timestamp=timestamp, call_type=call_type, query=query[:200],
            input_tokens=input_tokens, output_tokens=output_tokens,
            input_cost=round(input_cost,6), output_cost=round(output_cost,6),
            total_cost=round(total_cost,6), retries=retries, success=success,
        )
        self.gemini_calls.append(rec)
        if success:
            self.gemini_total_input  += input_tokens
            self.gemini_total_output += output_tokens
            self.gemini_total_cost   += total_cost
            self.gemini_total_retries += retries
        else:
            self.gemini_failed_calls += 1

        query_preview = (query[:60] + "...") if len(query) > 60 else query
        print(
            f"\n{'='*80}\n"
            f"Gemini Token Usage:\n"
            f"  Input Tokens:  {input_tokens:>10,}\n"
            f"  Output Tokens: {output_tokens:>10,}\n"
            f"  Total Tokens:  {input_tokens + output_tokens:>10,}\n"
            f"  Timestamp:     {timestamp}\n"
            f"  Query:         {query_preview}\n"
            f"  Call Type:     {call_type}\n"
            f"{'='*80}\n"
        )
        logger.info("Gemini [%s] in=%d out=%d cost=$%.6f retries=%d",
                    call_type, input_tokens, output_tokens, total_cost, retries)
        return rec

    def record_serp(self, query, call_type, results_returned,
                    cost=SERPAPI_COST_PER_SEARCH, provider="serpapi"):
        # `cost` is a parameter because Europe PMC is free: billing a metered rate
        # for an unmetered call would make the cost summary fiction. The record
        # shape is unchanged so the usage endpoint keeps its contract.
        rec = SerpAPICallRecord(
            timestamp=datetime.utcnow().isoformat(), query=query[:200],
            call_type=call_type, results_returned=results_returned,
            cost=cost,
        )
        self.serp_calls.append(rec)
        self.serp_total_calls += 1
        self.serp_total_cost  += cost
        logger.info("patent-search [%s/%s] results=%d cost=$%.4f  cumulative_calls=%d",
                    provider, call_type, results_returned, cost,
                    self.serp_total_calls)
        return rec

    def summary(self) -> dict:
        now = datetime.utcnow().isoformat()
        return {
            "session_start": self.session_start,
            "report_generated_at": now,
            "gemini": {
                "model": MODEL_NAME,
                "pricing": {
                    "input_per_1m_tokens_usd":  GEMINI_INPUT_COST_PER_1M,
                    "output_per_1m_tokens_usd": GEMINI_OUTPUT_COST_PER_1M,
                    "note": "Gemini 2.5 Flash pricing.",
                    "billing_link": "https://console.cloud.google.com/billing",
                },
                "totals": {
                    "total_calls":        len(self.gemini_calls),
                    "successful_calls":   len(self.gemini_calls) - self.gemini_failed_calls,
                    "failed_calls":       self.gemini_failed_calls,
                    "total_retries":      self.gemini_total_retries,
                    "total_input_tokens": self.gemini_total_input,
                    "total_output_tokens":self.gemini_total_output,
                    "total_tokens":       self.gemini_total_input + self.gemini_total_output,
                    "total_cost_usd":     round(self.gemini_total_cost, 6),
                    "total_cost_display": f"${self.gemini_total_cost:.6f}",
                },
                "by_call_type": self._gemini_by_type(),
                "recent_calls": [
                    {
                        "timestamp":     r.timestamp,
                        "call_type":     r.call_type,
                        "query_preview": r.query[:80],
                        "input_tokens":  r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "cost_usd":      r.total_cost,
                        "retries":       r.retries,
                        "success":       r.success,
                    }
                    for r in self.gemini_calls[-20:]
                ],
            },
            "serpapi": {
                "pricing": {
                    "cost_per_search_usd": SERPAPI_COST_PER_SEARCH,
                    "note": "SerpAPI standard plan: 100 searches/month free, then $0.015/search.",
                    "billing_link": "https://serpapi.com/manage-api-key",
                },
                "totals": {
                    "total_calls":     self.serp_total_calls,
                    "total_cost_usd":  round(self.serp_total_cost, 4),
                    "total_cost_display": f"${self.serp_total_cost:.4f}",
                },
                "by_call_type": self._serp_by_type(),
                "recent_calls": [
                    {
                        "timestamp":        r.timestamp,
                        "call_type":        r.call_type,
                        "query_preview":    r.query[:80],
                        "results_returned": r.results_returned,
                        "cost_usd":         r.cost,
                    }
                    for r in self.serp_calls[-20:]
                ],
            },
            "combined": {
                "total_cost_usd":     round(self.gemini_total_cost + self.serp_total_cost, 4),
                "total_cost_display": f"${self.gemini_total_cost + self.serp_total_cost:.4f}",
            },
        }

    def _gemini_by_type(self) -> dict:
        breakdown: Dict[str, dict] = {}
        for r in self.gemini_calls:
            if r.call_type not in breakdown:
                breakdown[r.call_type] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
                }
            if r.success:
                breakdown[r.call_type]["calls"]         += 1
                breakdown[r.call_type]["input_tokens"]  += r.input_tokens
                breakdown[r.call_type]["output_tokens"] += r.output_tokens
                breakdown[r.call_type]["cost_usd"]       = round(
                    breakdown[r.call_type]["cost_usd"] + r.total_cost, 6)
        return breakdown

    def _serp_by_type(self) -> dict:
        breakdown: Dict[str, dict] = {}
        for r in self.serp_calls:
            if r.call_type not in breakdown:
                breakdown[r.call_type] = {"calls": 0, "cost_usd": 0.0}
            breakdown[r.call_type]["calls"]    += 1
            breakdown[r.call_type]["cost_usd"]  = round(
                breakdown[r.call_type]["cost_usd"] + r.cost, 4)
        return breakdown


usage_tracker = UsageTracker()

# ══════════════════════════════════════════════════════════════════════════════
#  §3  DATABRICKS LLM CLIENT — pay-per-token Foundation Model API, replaces Gemini
# ══════════════════════════════════════════════════════════════════════════════

logger.info("✓ Databricks LLM client ready  (model=%s)", MODEL_NAME)


async def _gemini_call(prompt: str, call_type: str = "unknown",
                        query: str = "") -> str:
    """Named `_gemini_call` for compatibility with existing call sites in this file —
    routes to Databricks' pay-per-token Foundation Model API, not Gemini."""
    retries = 0
    for attempt in range(1, GEMINI_RETRY_ATTEMPTS + 1):
        try:
            text = await asyncio.to_thread(
                llm_client.databricks,
                messages=[{"role": "user", "content": prompt}],
                endpoint=MODEL_NAME,
            )
            input_tokens  = len(prompt) // 4
            output_tokens = len(text)   // 4
            usage_tracker.record_gemini(call_type=call_type, query=query,
                                        input_tokens=input_tokens,
                                        output_tokens=output_tokens,
                                        retries=retries, success=True)
            return text.replace("*", "")

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                delay_match = re.search(r"retry.*?(\d+)s", err_str, re.IGNORECASE)
                wait = int(delay_match.group(1)) + 2 if delay_match else 60
                logger.warning("Databricks serving 429 attempt %d/%d — waiting %ds",
                               attempt, GEMINI_RETRY_ATTEMPTS, wait)
                retries += 1
                if attempt < GEMINI_RETRY_ATTEMPTS:
                    await asyncio.sleep(wait)
                else:
                    usage_tracker.record_gemini(call_type=call_type, query=query,
                                                input_tokens=0, output_tokens=0,
                                                retries=retries, success=False)
                    raise
            else:
                usage_tracker.record_gemini(call_type=call_type, query=query,
                                            input_tokens=0, output_tokens=0,
                                            retries=retries, success=False)
                raise
    raise RuntimeError("Gemini call failed after all retries")

# ══════════════════════════════════════════════════════════════════════════════
#  §4  DATABASE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class SearchHistory(Base):
    __tablename__ = "novelty_search_history"
    id             = Column(Integer, primary_key=True, index=True)
    query          = Column(String)
    source         = Column(String)
    results        = Column(Text)
    chatbot_answer = Column(Text, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)


class AgentRun(Base):
    __tablename__ = "novelty_agent_runs"
    id           = Column(Integer, primary_key=True, index=True)
    user_query   = Column(String)
    patents_used = Column(Text)
    final_answer = Column(Text)
    created_at   = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _wipe_db_history():
    with engine.connect() as conn:
        try:
            conn.execute(text("DELETE FROM novelty_search_history"))
            conn.execute(text("DELETE FROM novelty_agent_runs"))
            conn.commit()
            logger.info("✓ Novelty agent DB history wiped on startup")
        except Exception as e:
            logger.warning("DB wipe warning: %s", e)

# ══════════════════════════════════════════════════════════════════════════════
#  §5  IN-MEMORY CACHE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_cache: Dict[str, Any] = {
    "latest_search": None,
    "queries":       {},
}

# ══════════════════════════════════════════════════════════════════════════════
#  §6  UTILITY HELPERS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_METADATA_KEYWORDS = [
    "own", "owner", "owns", "assignee", "assigned to", "inventor",
    "inventors", "invented by", "who made", "filed by", "filing date",
    "publication date", "applicant", "who holds",
]

_FORMAT_RULES = """
RESPONSE FORMAT RULES:
- Answer ONLY what the question asks. Simple questions 1-2 sentences.
- For complex questions use 150-300 words.
- Emphasise key terms in CAPS (e.g. METFORMIN, AMPK, TYPE-2 DIABETES)
- Bullets ONLY for list-like content; paragraphs for explanation
- No asterisks, no markdown, no chunk/score/index references
- If information is absent, state it clearly in one sentence
"""


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`(.*?)`",        r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = html_module.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def clean_patent_id(patent_id: str) -> str:
    if not patent_id:
        return ""
    pid = patent_id.strip()
    pid = re.sub(r"^/?patent/",                           "", pid, flags=re.IGNORECASE)
    pid = re.sub(r"^https?://patents.google.com/patent/", "", pid, flags=re.IGNORECASE)
    pid = pid.upper()
    pid = re.sub(r"/[A-Z]{2}$",  "", pid)
    pid = re.sub(r"[\?#].*$",    "", pid)
    pid = re.sub(r"[\s\.]+",     "", pid)
    pid = pid.replace("-", "")
    m = re.match(r"^([A-Z]{2})(RE)?0*(\d+)([A-Z0-9]*)$", pid)
    if m:
        country, reflag, number, suffix = m.groups()
        pid = f"{country}{reflag or ''}{number}{suffix}"
    return pid


def extract_patent_from_serp_result(patent_obj: dict) -> Optional[str]:
    pid = patent_obj.get("patent_id") or patent_obj.get("patentId") or patent_obj.get("id")
    if pid:
        return str(pid)
    for k in ("link", "url", "source", "more_info"):
        val = patent_obj.get(k)
        if val and isinstance(val, str):
            m = re.search(r"/patent/([A-Z0-9REre\-]+)", val)
            if m:
                return m.group(1)
    title = patent_obj.get("title", "")
    if title:
        m = re.search(r"\b([A-Z]{2}(?:RE)?\d+[A-Z0-9]*)\b", title.upper())
        if m:
            return m.group(1)
    return None


def _looks_like_title(query: str) -> bool:
    q = query.strip()
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
        return True
    words = q.split()
    if len(words) < 4:
        return False
    if words[0].lower() in {"what","how","why","where","when","does","is","are",
                             "can","will","should","would","could"}:
        return False
    if any("-" in w for w in words):
        return True
    if any(w[0].isupper() for w in words if len(w) > 2):
        return True
    return False


def _title_similarity(query: str, title: str) -> float:
    return difflib.SequenceMatcher(
        None,
        query.lower().strip().strip('"\''),
        title.lower().strip(),
    ).ratio()

# ══════════════════════════════════════════════════════════════════════════════
#  §7  HUGGING FACE EMBEDDING CLIENT
#      Replaces local SentenceTransformer.
#      All embedding calls go through the HF Inference API.
# ══════════════════════════════════════════════════════════════════════════════

class HFEmbeddingClient:
    """
    Thin async-friendly wrapper around huggingface_hub.InferenceClient
    for the BAAI/bge-small-en-v1.5 feature-extraction endpoint.

    bge-small-en-v1.5 returns 384-dimensional L2-normalised vectors when
    the input is prefixed with "query: " (retrieval queries) or
    "passage: " (documents/chunks).  This mirrors the original BGE usage
    convention kept in the old SentenceTransformer code.
    """

    def __init__(self, model: str = HF_EMBEDDING_MODEL, token: str = HF_TOKEN):
        self._client = InferenceClient(token=token)
        self._model  = model
        logger.info("✓ HFEmbeddingClient ready  (model=%s)", model)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Call the HF Inference API for a list of texts.
        Returns a float32 numpy array of shape (len(texts), QDRANT_VECTOR_SIZE).
        Handles both old API (list of floats per item) and new API (nested list).
        """
        result = self._client.feature_extraction(texts, model=self._model)

        # result may be: List[List[float]] or np.ndarray or List[np.ndarray]
        if isinstance(result, np.ndarray):
            arr = result.astype("float32")
        else:
            arr = np.array(result, dtype="float32")

        # If the model returns [batch, seq_len, hidden] (full token embeddings),
        # take the CLS token (index 0) to get sentence embeddings.
        if arr.ndim == 3:
            arr = arr[:, 0, :]

        # L2-normalise so cosine ≡ dot-product (consistent with old FAISS IndexFlatIP)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        arr   = arr / norms
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single retrieval query. Returns shape (384,)."""
        return self._embed_batch(["query: " + query])[0]

    def embed_passages(self, texts: List[str],
                       batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of passages/chunks. Returns shape (len(texts), 384).
        Batches requests to avoid HF payload limits (~1 MB per request).
        """
        all_vectors: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = ["passage: " + t for t in texts[i : i + batch_size]]
            all_vectors.append(self._embed_batch(batch))
        return np.vstack(all_vectors) if all_vectors else np.empty((0, QDRANT_VECTOR_SIZE), dtype="float32")


# Singleton — shared by QdrantVectorStore and retrieval helpers
hf_embedder = HFEmbeddingClient()

# ══════════════════════════════════════════════════════════════════════════════
#  §8  HTML FETCHER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class PatentHTMLFetcher:
    def __init__(self):
        self._mem_cache: Dict[str, dict] = {}
        logger.info("✓ PatentHTMLFetcher ready")

    def cache_stats(self) -> dict:
        return {"l1_memory_patents": len(self._mem_cache)}

    def clear_cache(self) -> None:
        self._mem_cache.clear()
        logger.info("Patent HTML memory cache cleared")

    async def fetch_patent_html(self, patent_id: str) -> Optional[str]:
        if patent_id in self._mem_cache:
            return self._mem_cache[patent_id].get("html")
        url = f"https://patents.google.com/patent/{patent_id}/en"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                resp = await c.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.text
            logger.warning("HTTP %s fetching %s", resp.status_code, patent_id)
        except Exception as e:
            logger.error("Error fetching %s: %s", patent_id, e)
        return None

    def parse_patent_html(self, html: str, patent_id: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        data: dict = {"patent_id": patent_id,
                      "url": f"https://patents.google.com/patent/{patent_id}/en",
                      "independent_claims": [],
                      "dependent_claims": []}

        full_text = soup.get_text("\n", strip=True)

        # ── Title ────────────────────────────────────────────────────────────
        te = soup.find("span", {"itemprop": "title"}) or soup.find("meta", {"name": "DC.title"})
        if te:
            data["title"] = te.get("content") if te.name == "meta" else te.get_text(strip=True)

        # ── Abstract ─────────────────────────────────────────────────────────
        ab = soup.find("section", {"itemprop": "abstract"}) or soup.find("div", class_="abstract")
        if ab:
            data["abstract"] = re.sub(r"^Abstract\s*", "", ab.get_text(" ", strip=True), flags=re.I)
        else:
            m = re.search(r"(?:^|\n)\s*Abstract\s*[:—-]?\s*(.*?)(?=\n(?:Background|Summary|Introduction|Detailed Description|Claims|The invention|What is claimed)|$)", full_text, re.DOTALL | re.I)
            if m:
                data["abstract"] = m.group(1).strip()

        # ── Background ───────────────────────────────────────────────────────
        bg_sec = (soup.find("section", {"itemprop": "background"})
                  or soup.find("div", class_="background")
                  or soup.find("section", class_="background"))
        if bg_sec:
            data["background"] = bg_sec.get_text(" ", strip=True)
        else:
            m = re.search(r"(?:^|\n)\s*(?:BACKGROUND|Background of the Invention|Background|FIELD OF THE INVENTION|Field of the Invention)\s*[:—-]?\s*(.*?)(?=\n(?:SUMMARY|Summary of the Invention|Summary|DETAILED DESCRIPTION|Detailed Description|Description|CLAIMS?|What is claimed|The invention claimed)|$)", full_text, re.DOTALL | re.I)
            if m:
                data["background"] = m.group(1).strip()

        # ── Summary ──────────────────────────────────────────────────────────
        sum_sec = (soup.find("section", {"itemprop": "summary"})
                   or soup.find("div", class_="summary")
                   or soup.find("section", class_="summary"))
        if sum_sec:
            data["summary"] = sum_sec.get_text(" ", strip=True)
        else:
            m = re.search(r"(?:^|\n)\s*(?:SUMMARY|Summary of the Invention|Summary)\s*[:—-]?\s*(.*?)(?=\n(?:DETAILED DESCRIPTION|Detailed Description|Description|CLAIMS?|What is claimed|The invention claimed)|$)", full_text, re.DOTALL | re.I)
            if m:
                data["summary"] = m.group(1).strip()

        # ── Description ──────────────────────────────────────────────────────
        desc_sec = (soup.find("section", {"itemprop": "description"})
                    or soup.find("div", class_="description"))
        if desc_sec:
            full = desc_sec.get_text("\n", strip=True)
            if not data.get("background"):
                bg_in_desc = re.search(r"(?:BACKGROUND|Background of the Invention)(.*?)(?=SUMMARY|Summary|DETAILED|BRIEF|$)", full, re.DOTALL | re.I)
                if bg_in_desc:
                    data["background"] = bg_in_desc.group(1).strip()
            if not data.get("summary"):
                sum_in_desc = re.search(r"(?:SUMMARY|Summary of the Invention)(.*?)(?=DETAILED|BRIEF|DESCRIPTION|$)", full, re.DOTALL | re.I)
                if sum_in_desc:
                    data["summary"] = sum_in_desc.group(1).strip()
            desc_match = re.search(r"(?:DETAILED DESCRIPTION|Detailed Description|Description)(.*?)(?=CLAIMS?|What is claimed|$)", full, re.DOTALL | re.I)
            if desc_match:
                data["description"] = desc_match.group(1).strip()
            elif not any(k in data for k in ["background","summary","description"]):
                data["description"] = full
        else:
            if not data.get("background"):
                m = re.search(r"(?:^|\n)\s*(?:BACKGROUND|Background of the Invention|Background|FIELD OF THE INVENTION|Field of the Invention)\s*[:—-]?\s*(.*?)(?=\n(?:SUMMARY|Summary of the Invention|Summary|DETAILED DESCRIPTION|Detailed Description|Description|CLAIMS?|What is claimed|The invention claimed)|$)", full_text, re.DOTALL | re.I)
                if m:
                    data["background"] = m.group(1).strip()
            if not data.get("summary"):
                m = re.search(r"(?:^|\n)\s*(?:SUMMARY|Summary of the Invention|Summary)\s*[:—-]?\s*(.*?)(?=\n(?:DETAILED DESCRIPTION|Detailed Description|Description|CLAIMS?|What is claimed|The invention claimed)|$)", full_text, re.DOTALL | re.I)
                if m:
                    data["summary"] = m.group(1).strip()
            m = re.search(r"(?:^|\n)\s*(?:DETAILED DESCRIPTION|Description|DESCRIPTION)\s*(?:OF THE|OF)?\s*(?:INVENTION|THE INVENTION)?\s*[:—-]?\s*(.*?)(?=\n(?:CLAIMS?|What is claimed|The invention claimed)|$)", full_text, re.DOTALL | re.I)
            if m:
                data["description"] = m.group(1).strip()

        # ── Claims ───────────────────────────────────────────────────────────
        indep, dep = [], []
        cl_sec = None
        for tag, attrs in [
            ("section", {"itemprop": "claims"}),
            ("div",     {"class":   "claims"}),
            ("section", {"class":   "claims"}),
            ("div",     {"class":   "patent-claims"}),
            ("div",     {"id":      "claims"}),
        ]:
            cl_sec = soup.find(tag, attrs)
            if cl_sec:
                break

        if cl_sec:
            claim_elems = cl_sec.find_all(
                "div",
                class_=lambda c: c and any(
                    part in {"claim", "claim-dependent"}
                    for part in (c if isinstance(c, list) else [c])
                )
            ) + cl_sec.find_all("claim")

            if claim_elems:
                seen_numbers = set()
                for e in claim_elems:
                    cn = None
                    if e.get("num"):
                        cn = e.get("num")
                    elif e.get("id"):
                        m = re.search(r"CLM-?0*(\d+)", e.get("id"))
                        if m:
                            cn = m.group(1)
                    if not cn:
                        text_match = re.search(r"^(?:\(?\s*)(\d+)[\.)]\s*", e.get_text(" ", strip=True))
                        if text_match:
                            cn = text_match.group(1)
                    if not cn:
                        continue
                    try:
                        ni = int(re.search(r"\d+", cn).group())
                    except Exception:
                        continue
                    if ni in seen_numbers:
                        continue
                    seen_numbers.add(ni)
                    text_nodes = e.find_all(["div","span"], class_=lambda c: c and "claim-text" in c.lower())
                    ctxt = (" ".join(t.get_text(" ", strip=True) for t in text_nodes if t.get_text(strip=True))
                            if text_nodes else e.get_text(" ", strip=True))
                    ctxt = re.sub(r"^\s*\d+[\.)]\s*", "", ctxt).strip()
                    ctxt = re.sub(r"\s+", " ", ctxt)
                    if not ctxt:
                        continue
                    obj  = {"number": ni, "text": ctxt}
                    deps = self._extract_dependencies(ctxt)
                    if deps:
                        obj["depends_on"] = deps
                        dep.append(obj)
                    else:
                        indep.append(obj)

        if not (indep or dep):
            claim_keywords = ["claims", "what is claimed", "the invention claimed"]
            claims_start = -1
            for kw in claim_keywords:
                pos = full_text.lower().find(kw)
                if pos != -1:
                    claims_start = pos
                    break
            claims_text = full_text[claims_start:] if claims_start != -1 else full_text

            patterns = [
                (r"(?:^|\n)\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.\s|\n\s*\d+\)|\n\s*Claim\s+\d+|$)", "."),
                (r"(?:^|\n)\s*(\d+)\)\s+(.*?)(?=\n\s*\d+\)\s|\n\s*\d+\.\s|$)",                ")"),
                (r"(?:^|\n)\s*\((\d+)\)\s+(.*?)(?=\n\s*\(\d+\)\s|$)",                         ")"),
                (r"(?:^|\n)\s*Claim\s+(\d+)\s*[.:\-]?\s*(.*?)(?=\n\s*Claim\s+\d+|$)",         "."),
            ]
            seen: set = set()
            for pattern, _ in patterns:
                for cn, ctxt in re.findall(pattern, claims_text, re.DOTALL):
                    key = (cn.strip(), ctxt.strip())
                    if key not in seen:
                        seen.add(key)
                        try:
                            ni   = int(cn)
                            ctxt = re.sub(r"\s+", " ", ctxt.strip())
                            obj  = {"number": ni, "text": ctxt}
                            deps = self._extract_dependencies(ctxt)
                            if deps:
                                obj["depends_on"] = deps
                                dep.append(obj)
                            else:
                                indep.append(obj)
                        except (ValueError, AttributeError):
                            continue
            indep = sorted(indep, key=lambda x: x["number"])
            dep   = sorted(dep,   key=lambda x: x["number"])

        data["independent_claims"] = indep
        data["dependent_claims"]   = dep

        # ── Metadata ─────────────────────────────────────────────────────────
        ae = (soup.find("dd", {"itemprop": "assigneeCurrent"})
              or soup.find("meta", {"name": "assignee"}))
        if ae:
            data["assignee"] = ae.get("content") if ae.name == "meta" else ae.get_text(strip=True)
        invs = soup.find_all("dd", {"itemprop": "inventor"})
        if invs:
            data["inventors"] = [i.get_text(strip=True) for i in invs]
        fd = soup.find("time", {"itemprop": "filingDate"})
        if fd:
            data["filing_date"] = fd.get("datetime") or fd.get_text(strip=True)
        pd2 = soup.find("time", {"itemprop": "publicationDate"})
        if pd2:
            data["publication_date"] = pd2.get("datetime") or pd2.get_text(strip=True)

        indep_count = len(data.get("independent_claims", []))
        dep_count   = len(data.get("dependent_claims",   []))
        logger.info("✓ Patent %s: %d indep claims, %d dep claims, abstract=%s",
                    patent_id, indep_count, dep_count,
                    "Y" if data.get("abstract") else "N")
        if indep_count == 0 and dep_count == 0:
            logger.warning("⚠️  Patent %s: NO CLAIMS EXTRACTED", patent_id)
        return data

    @staticmethod
    def _extract_dependencies(text: str) -> List[int]:
        deps = []
        for pat in [r"claim (\d+)", r"claims (\d+)-(\d+)",
                    r"claims (\d+) to (\d+)", r"claims (\d+) or (\d+)"]:
            for m in re.findall(pat, text, re.I):
                deps.extend(int(x) for x in (m if isinstance(m, tuple) else [m]) if str(x).isdigit())
        return list(set(deps))

    async def process_patent(self, patent_id: str) -> Optional[dict]:
        html = await self.fetch_patent_html(patent_id)
        if not html:
            return None
        structured = self.parse_patent_html(html, patent_id)
        self._mem_cache[patent_id] = {"html": html, "structured": structured}
        return structured

    async def process_multiple_patents(self, ids: List[str],
                                        max_concurrent: int = 2) -> dict:
        results = {}
        for i in range(0, len(ids), max_concurrent):
            batch = ids[i : i + max_concurrent]
            batch_results = await asyncio.gather(
                *[self.process_patent(pid) for pid in batch],
                return_exceptions=True)
            for pid, res in zip(batch, batch_results):
                results[pid] = None if isinstance(res, Exception) else res
        return results

    def get_cached_data(self, pid: str) -> Optional[dict]:
        return self._mem_cache.get(pid, {}).get("structured")

# ══════════════════════════════════════════════════════════════════════════════
#  §9  CHUNKER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class PatentChunker:
    SECTION_WEIGHTS = {
        "independent_claims": 1.00, "abstract": 0.95, "summary": 0.90,
        "dependent_claims": 0.85,   "metadata": 0.85,
        "description": 0.80,        "background": 0.70,
    }

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk_text(self, text: str, max_chunks: Optional[int] = None) -> List[str]:
        words  = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunks.append(" ".join(words[i : i + self.chunk_size]))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def create_chunks_from_patent(self, data: dict) -> List[dict]:
        pid    = data.get("patent_id", "UNKNOWN")
        chunks: List[dict] = []

        def _add(section_type: str, text: str, idx: int,
                 extra_meta: Optional[dict] = None):
            chunks.append({
                "chunk_id":       f"{pid}_{section_type}_{idx}",
                "patent_id":      pid,
                "section_type":   section_type,
                "text":           text,
                "chunk_index":    idx,
                "metadata":       extra_meta or {"section": section_type, "is_claim": False},
                "section_weight": self.SECTION_WEIGHTS.get(section_type, 0.75),
            })

        if data.get("abstract"):
            _add("abstract", data["abstract"], 0)
        for c in data.get("independent_claims", []):
            _add("independent_claims", c["text"], c["number"],
                 {"is_claim": True, "claim_type": "independent", "claim_number": c["number"]})
        for c in data.get("dependent_claims", []):
            _add("dependent_claims", c["text"], c["number"],
                 {"is_claim": True, "claim_type": "dependent", "claim_number": c["number"],
                  "depends_on": c.get("depends_on", [])})
        for i, t in enumerate(self.chunk_text(data.get("summary",      ""), max_chunks=3)):
            _add("summary",     t, i)
        for i, t in enumerate(self.chunk_text(data.get("description",  ""), max_chunks=10)):
            _add("description", t, i)
        for i, t in enumerate(self.chunk_text(data.get("background",   ""), max_chunks=3)):
            _add("background",  t, i)

        meta_parts = []
        if data.get("assignee"):         meta_parts.append(f"Assignee: {data['assignee']}")
        if data.get("inventors"):        meta_parts.append(f"Inventors: {', '.join(data['inventors'])}")
        if data.get("filing_date"):      meta_parts.append(f"Filing Date: {data['filing_date']}")
        if data.get("publication_date"): meta_parts.append(f"Publication Date: {data['publication_date']}")
        if meta_parts:
            _add("metadata", ". ".join(meta_parts), 0,
                 {"section": "metadata", "is_claim": False})
        return chunks

    def process_multiple_patents(self, patents: dict) -> dict:
        return {pid: self.create_chunks_from_patent(d)
                for pid, d in patents.items() if d}

# ══════════════════════════════════════════════════════════════════════════════
#  §10  RERANKER  (unchanged — CrossEncoder only, no local SentenceTransformer)
# ══════════════════════════════════════════════════════════════════════════════

class PatentReranker:
    def __init__(self):
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[dict] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = []
        for word in text.lower().split():
            tokens.append(word)
            if "-" in word:
                tokens.extend(p for p in word.split("-") if p)
        return tokens

    def prepare_bm25_index(self, documents: List[dict]) -> None:
        for d in documents:
            if "combined_text" not in d:
                d["combined_text"] = f"{d.get('Title','')} {d.get('Snippet','')}".lower()
        self.bm25      = BM25Okapi([self._tokenize(d["combined_text"]) for d in documents])
        self.documents = documents

    def bm25_rank(self, query: str, top_k: int = 20) -> List[dict]:
        if not self.bm25:
            return []
        scores  = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{"document": self.documents[i], "bm25_score": float(scores[i]),
                 "bm25_rank": r+1} for r, i in enumerate(top_idx)]

    def reciprocal_rank_fusion(self, rankings: List[List[dict]], k: int = 60) -> List[dict]:
        rrf: dict = {}
        for ranking in rankings:
            for item in ranking:
                rrf[item["doc_id"]] = rrf.get(item["doc_id"], 0) + 1.0 / (k + item["rank"])
        return [{"doc_id": d, "rrf_score": s, "rrf_rank": i+1}
                for i, (d, s) in enumerate(
                    sorted(rrf.items(), key=lambda x: x[1], reverse=True))]

    def title_pin_score(self, query: str, documents: List[dict],
                        pin_weight: float = 6.0) -> List[dict]:
        q = query.strip('"\'').lower()
        for d in documents:
            title           = d.get("Title", d.get("title","")).lower()
            sim             = difflib.SequenceMatcher(None, q, title).ratio()
            d["_title_pin"] = round(sim * pin_weight, 4)
        return documents

    def cross_encoder_rerank(self, query: str, documents: List[dict],
                              top_k: int = 10) -> List[dict]:
        if not documents:
            return []
        pairs  = [[query, d.get("combined_text",
                                f"{d.get('Title','')} {d.get('Snippet','')}")[:512]]
                  for d in documents]
        scores = self.cross_encoder.predict(pairs)
        ranked = []
        for d, s in zip(documents, scores):
            base = (float(s) + 10) / 2
            pin  = d.get("_title_pin", 0.0)
            ranked.append({**d, "relevance_score": round(base + pin, 4),
                           "_base_score": round(base, 4), "_pin_bonus": round(pin, 4)})
        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        if ranked:
            lo, hi = ranked[-1]["relevance_score"], ranked[0]["relevance_score"]
            rng    = hi - lo or 1
            for d in ranked:
                d["relevance_score"] = round(1 + 9*(d["relevance_score"]-lo)/rng, 3)
        return ranked[:top_k]

    def rerank_pipeline(self, query: str, initial: List[dict], top_k: int = 10,
                        use_bm25: bool = True, use_rrf: bool = True,
                        is_title_query: bool = False) -> List[dict]:
        if not initial:
            return []
        for i, d in enumerate(initial):
            if "PatentID" in d:
                d["patent_id"] = d["PatentID"]
            elif "patent_id" not in d:
                link = d.get("Link", "")
                d["patent_id"] = (link.split("/patent/")[1].split("/")[0]
                                  if "/patent/" in link else f"doc_{i}")
        bm25_ranking = []
        if use_bm25:
            self.prepare_bm25_index(initial)
            bm25_ranking = [{"doc_id": r["document"]["patent_id"], "rank": r["bm25_rank"]}
                            for r in self.bm25_rank(query, len(initial))]
        api_ranking = [{"doc_id": d["patent_id"], "rank": i+1}
                       for i, d in enumerate(initial)]
        if use_rrf and use_bm25:
            rrf_order  = [r["doc_id"] for r in
                          self.reciprocal_rank_fusion([api_ranking, bm25_ranking])]
            doc_map    = {d["patent_id"]: d for d in initial}
            candidates = [doc_map[did] for did in rrf_order if did in doc_map][:30]
        else:
            candidates = initial[:30]
        if is_title_query:
            candidates = self.title_pin_score(query, candidates)
        return self.cross_encoder_rerank(query, candidates, top_k=top_k)

# ══════════════════════════════════════════════════════════════════════════════
#  §11  QDRANT VECTOR STORE
#       Drop-in replacement for the old PatentVectorStore.
#       Public interface is identical so all callers remain unchanged.
# ══════════════════════════════════════════════════════════════════════════════

class QdrantVectorStore:
    """
    Cloud-native vector store backed by Qdrant.

    Key differences from the old FAISS-based PatentVectorStore:
      • No local model — embeddings are generated by HFEmbeddingClient (HF API).
      • No local FAISS index — vectors live in Qdrant Cloud.
      • Patent-level deduplication: is_patent_indexed() checks Qdrant before
        any HTML fetch / chunk / embed work is done.
      • Eviction is handled by deleting points whose payload.patent_id matches
        the oldest tracked patent.
      • Section weights are applied at query time (same logic as before).
    """

    COLLECTION = QDRANT_COLLECTION_NAME
    VECTOR_SIZE = QDRANT_VECTOR_SIZE

    def __init__(self):
        self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                                    timeout=30)
        self._patent_order: OrderedDict[str, bool] = OrderedDict()
        # in-memory chunk metadata mirror (patent_id → List[dict])
        # populated on add_chunks; consulted for get_chunks_by_patent /
        # get_metadata_chunks so we avoid round-trips to Qdrant for metadata.
        self._chunk_meta: Dict[str, List[dict]] = {}
        self._ensure_collection()
        self._load_existing_patent_ids()
        logger.info("✓ QdrantVectorStore ready  (collection=%s, patents=%d)",
                    self.COLLECTION, len(self._patent_order))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self.COLLECTION not in existing:
            self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=qdrant_models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            self._client.create_payload_index(
                collection_name=self.COLLECTION,
                field_name="patent_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Created Qdrant collection '%s' with patent_id index", self.COLLECTION)
        else:
            logger.info("Reusing Qdrant collection '%s'", self.COLLECTION)

    def _load_existing_patent_ids(self) -> None:
        """
        Scroll through the collection and rebuild _patent_order from stored
        payloads.  Called once on construction so warm restarts are aware of
        existing data.
        """
        try:
            offset = None
            seen: set = set()
            while True:
                results, offset = self._client.scroll(
                    collection_name=self.COLLECTION,
                    scroll_filter=None,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results:
                    pid = point.payload.get("patent_id")
                    if pid and pid not in seen:
                        seen.add(pid)
                        self._patent_order[pid] = True
                if offset is None:
                    break
        except Exception as e:
            logger.warning("Could not load existing patent IDs from Qdrant: %s", e)

    def _enforce_cap(self) -> None:
        """Evict the oldest patent(s) until we are at or below MAX_PATENTS_IN_STORE."""
        while len(self._patent_order) > MAX_PATENTS_IN_STORE:
            oldest_pid, _ = next(iter(self._patent_order.items()))
            logger.info("Evicting oldest patent from Qdrant store: %s", oldest_pid)
            self._delete_patent(oldest_pid)

    def _delete_patent(self, pid: str) -> None:
        """Delete all points for a patent from Qdrant and clean up local state."""
        try:
            self._client.delete(
                collection_name=self.COLLECTION,
                points_selector=qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(
                        key="patent_id",
                        match=qdrant_models.MatchValue(value=pid),
                    )]
                ),
            )
        except Exception as e:
            logger.warning("Qdrant delete failed for patent %s: %s", pid, e)
        self._patent_order.pop(pid, None)
        self._chunk_meta.pop(pid, None)
        logger.info("Deleted patent %s from Qdrant", pid)

    # ── public interface (mirrors old PatentVectorStore) ──────────────────────

    def get_indexed_patent_ids(self) -> set:
        return set(self._patent_order.keys())

    def is_patent_indexed(self, pid: str) -> bool:
        return pid in self._patent_order

    def add_chunks(self, chunks: List[dict], batch_size: int = 32) -> dict:
        """
        Embed chunks via HF API and upsert into Qdrant.
        Chunks for a patent already in the store are skipped at the caller level
        (run_pipeline checks is_patent_indexed before calling add_chunks).
        """
        if not chunks:
            return {"added": 0, "skipped": 0}

        texts    = [c["text"] for c in chunks]
        vectors  = hf_embedder.embed_passages(texts, batch_size=batch_size)

        points: List[qdrant_models.PointStruct] = []
        for chunk, vec in zip(chunks, vectors):
            pid = chunk.get("patent_id", "")
            payload = {
                "patent_id":      pid,
                "section_type":   chunk.get("section_type", ""),
                "chunk_text":     chunk.get("text", ""),
                "chunk_id":       chunk.get("chunk_id", ""),
                "chunk_index":    chunk.get("chunk_index", 0),
                "section_weight": chunk.get("section_weight", 0.75),
                # claim-specific
                "is_claim":       chunk.get("metadata", {}).get("is_claim", False),
                "claim_number":   chunk.get("metadata", {}).get("claim_number", None),
                "claim_type":     chunk.get("metadata", {}).get("claim_type",   None),
                "depends_on":     chunk.get("metadata", {}).get("depends_on",   []),
            }
            points.append(qdrant_models.PointStruct(
                id      = str(uuid.uuid4()),
                vector  = vec.tolist(),
                payload = payload,
            ))
            # mirror into local metadata cache
            if pid:
                self._chunk_meta.setdefault(pid, []).append({
                    **chunk,
                    "payload": payload,
                })

        # upsert in batches
        for i in range(0, len(points), batch_size):
            self._client.upsert(collection_name=self.COLLECTION,
                                points=points[i : i + batch_size])

        # track patent order
        for chunk in chunks:
            pid = chunk.get("patent_id", "")
            if pid and pid not in self._patent_order:
                self._patent_order[pid] = True

        self._enforce_cap()
        logger.info("Upserted %d chunks into Qdrant", len(points))
        return {"added": len(points), "skipped": 0}

    def search_by_patent(self, query: str, patent_id: str, top_k: int = 10,
                          apply_section_weights: bool = True) -> List[dict]:
        """
        Embed the query and search Qdrant, filtering to a specific patent_id.
        Returns a list of chunk-like dicts with similarity_score / weighted_score,
        sorted by weighted_score descending.
        """
        if not self.is_patent_indexed(patent_id):
            return []

        query_vec = hf_embedder.embed_query(query).tolist()
        response = self._client.query_points(
            collection_name=self.COLLECTION,
            query=query_vec,
            query_filter=qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(
                    key="patent_id",
                    match=qdrant_models.MatchValue(value=patent_id),
                )]
            ),
            limit=top_k,
            with_payload=True,
        )
        hits = response.points

        results = []
        for hit in hits:
            p = hit.payload
            section_weight = p.get("section_weight", 0.75)
            sim_score      = float(hit.score)
            results.append({
                "patent_id":        p.get("patent_id",      patent_id),
                "section_type":     p.get("section_type",   ""),
                "text":             p.get("chunk_text",      ""),
                "chunk_id":         p.get("chunk_id",        ""),
                "chunk_index":      p.get("chunk_index",     0),
                "section_weight":   section_weight,
                "similarity_score": sim_score,
                "weighted_score":   (sim_score * section_weight
                                     if apply_section_weights else sim_score),
                "metadata": {
                    "is_claim":     p.get("is_claim",    False),
                    "claim_number": p.get("claim_number"),
                    "claim_type":   p.get("claim_type"),
                    "depends_on":   p.get("depends_on",  []),
                },
            })

        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results

    def get_chunks_by_patent(self, pid: str) -> List[dict]:
        """
        Return all stored chunks for a patent from the local mirror.
        Falls back to a Qdrant scroll if the mirror is empty (e.g. after restart).
        """
        if pid in self._chunk_meta:
            return self._chunk_meta[pid]
        # fallback: scroll Qdrant
        chunks = []
        try:
            offset = None
            while True:
                results, offset = self._client.scroll(
                    collection_name=self.COLLECTION,
                    scroll_filter=qdrant_models.Filter(
                        must=[qdrant_models.FieldCondition(
                            key="patent_id",
                            match=qdrant_models.MatchValue(value=pid),
                        )]
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results:
                    p = point.payload
                    chunks.append({
                        "patent_id":      p.get("patent_id",    pid),
                        "section_type":   p.get("section_type", ""),
                        "text":           p.get("chunk_text",   ""),
                        "chunk_id":       p.get("chunk_id",     ""),
                        "chunk_index":    p.get("chunk_index",  0),
                        "section_weight": p.get("section_weight", 0.75),
                        "metadata": {
                            "is_claim":     p.get("is_claim",    False),
                            "claim_number": p.get("claim_number"),
                            "claim_type":   p.get("claim_type"),
                            "depends_on":   p.get("depends_on",  []),
                        },
                    })
                if offset is None:
                    break
        except Exception as e:
            logger.warning("Qdrant scroll failed for %s: %s", pid, e)
        self._chunk_meta[pid] = chunks
        return chunks

    def get_metadata_chunks(self, pid: str) -> List[dict]:
        return [c for c in self.get_chunks_by_patent(pid)
                if c.get("section_type") == "metadata"]

    def get_statistics(self) -> dict:
        try:
            info        = self._client.get_collection(self.COLLECTION)
            total_pts   = info.points_count
        except Exception:
            total_pts = 0
        return {
            "total_chunks":  total_pts,
            "total_patents": len(self._patent_order),
            "patent_ids":    list(self._patent_order.keys()),
            "cap":           MAX_PATENTS_IN_STORE,
            "backend":       "qdrant_cloud",
            "collection":    self.COLLECTION,
        }

    def clear(self) -> None:
        """
        Wipe the Qdrant collection entirely (delete + recreate) and reset
        all local state.  Called at startup and can be triggered manually.
        """
        try:
            self._client.delete_collection(self.COLLECTION)
            logger.info("Qdrant collection '%s' deleted", self.COLLECTION)
        except Exception as e:
            logger.warning("Could not delete collection: %s", e)
        self._patent_order.clear()
        self._chunk_meta.clear()
        self._ensure_collection()
        logger.info("✓ QdrantVectorStore cleared and recreated")

# ══════════════════════════════════════════════════════════════════════════════
#  §12  SINGLETON INSTANCES
# ══════════════════════════════════════════════════════════════════════════════

html_fetcher    = PatentHTMLFetcher()
chunker         = PatentChunker(chunk_size=500, overlap=100)
vector_store    = QdrantVectorStore()
patent_reranker = PatentReranker()
logger.info("✓ All services initialised")

# ══════════════════════════════════════════════════════════════════════════════
#  §13  STARTUP WIPE
#       Clears Qdrant collection, HTML memory cache, in-memory query cache,
#       and DB history.  No local files to delete (no FAISS, no disk HTML cache).
# ══════════════════════════════════════════════════════════════════════════════

def startup_wipe() -> None:
    logger.info("🧹 Novelty agent startup wipe…")
    html_fetcher.clear_cache()
    vector_store.clear()                       # drops + recreates Qdrant collection
    _cache.update({"latest_search": None, "queries": {}})
    _wipe_db_history()
    logger.info("✓ Startup wipe complete")


startup_wipe()

# ══════════════════════════════════════════════════════════════════════════════
#  §14  GOOGLE PATENTS SEARCH  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

#: Europe PMC's patent corpus is the SureChEMBL patent set — the same EMBL-EBI
#: chemistry-patent index, reachable over a documented REST endpoint with no key
#: and no per-search charge. SerpAPI billed $0.015 a search against a 100/month
#: free tier and simply stopped answering once that ran out ("Your account has run
#: out of searches"), which took NovSearch down with it.
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

#: Europe PMC's page size cap for a single search request.
_EPMC_MAX_PAGE_SIZE = 1000


#: Words that carry no discriminating power in a patent query. Europe PMC ANDs
#: every term, so one generic word ("combination", "inhibitor") can take a real
#: result set to zero: `jak2 AND imatinib` finds 2 patents, and adding
#: `AND combination` finds none. These are dropped before the narrowest attempt.
_EPMC_GENERIC_TERMS = frozenset({
    "a", "an", "and", "the", "of", "for", "in", "on", "with", "to", "or", "by",
    "analysis", "assessment", "combination", "combinations", "compound",
    "compounds", "drug", "drugs", "novel", "novelty", "perform", "prior", "art",
    "study", "therapy", "treatment", "use", "uses", "using", "via",
})


def _epmc_queries(query: str, is_title: bool = False) -> List[str]:
    """
    Progressively broader Europe PMC queries for one caller query, narrowest first.

    Europe PMC has no Google-style operators: `intitle:` becomes `TITLE:`, the
    corpus is narrowed with `SRC:"PAT"`, and — the part that matters — a bare
    space is a *phrase* match, not an AND. `jak2 imatinib combination` therefore
    matched nothing at all, which is why every free-text novelty query came back
    empty once it stopped going through Google.

    So terms are ANDed explicitly, and because ANDing everything is brittle the
    caller gets a ladder: all terms, then content terms only, then the two most
    specific, then the single most specific. `_patent_search` walks it and stops
    at the first query that returns anything.
    """
    q = re.sub(r"^\s*intitle:", "", query.strip(), flags=re.IGNORECASE)
    q = q.replace('"', " ").strip()
    if not q:
        return []

    if is_title:
        # A title search is a phrase lookup by intent; relaxing it would return
        # a different patent rather than the same one, so it stays exact.
        return [f'(TITLE:"{q}") AND (SRC:"PAT")']

    terms = [t for t in re.split(r"[^\w\-]+", q) if t]
    if not terms:
        return []
    content = [t for t in terms if t.lower() not in _EPMC_GENERIC_TERMS] or terms

    ladders: List[List[str]] = [terms, content, content[:2], content[:1]]
    out, seen = [], set()
    for group in ladders:
        if not group:
            continue
        built = f'({" AND ".join(group)}) AND (SRC:"PAT")'
        if built not in seen:
            seen.add(built)
            out.append(built)
    return out


async def _patent_search(query: str, num: int, call_type: str = "standard",
                         is_title: bool = False) -> List[dict]:
    """
    Search the SureChEMBL patent corpus via Europe PMC.

    Returns records in the shape the rest of this module already expects from a
    search result — `patent_id`, `title`, `snippet`, `link` — so path extraction,
    deduplication and reranking downstream are untouched.
    """
    queries = _epmc_queries(query, is_title)
    if not queries:
        return []
    hits: List[dict] = []
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            for epmc_q in queries:
                resp = await c.get(EUROPEPMC_SEARCH_URL, params={
                    "query":      epmc_q,
                    "format":     "json",
                    "resultType": "core",
                    "pageSize":   min(max(int(num), 1), _EPMC_MAX_PAGE_SIZE),
                })
                resp.raise_for_status()
                hits = (resp.json().get("resultList") or {}).get("result") or []
                if hits:
                    break
                logger.info("Europe PMC: no patents for %s — broadening", epmc_q)
    except Exception:
        logger.exception("Europe PMC patent search failed for: %s", query)
        usage_tracker.record_serp(query=query, call_type=call_type,
                                  results_returned=0, cost=0.0,
                                  provider="europepmc")
        return []

    results = []
    for h in hits:
        pid = (h.get("id") or "").strip()
        if not pid:
            continue
        results.append({
            "patent_id": pid,
            "title":     h.get("title") or "No title",
            "snippet":   (h.get("abstractText") or "")[:500],
            # Google Patents renders the same publication number, so existing
            # links in saved results keep resolving.
            "link":      f"https://patents.google.com/patent/{pid}/en",
            "pubYear":   h.get("pubYear"),
        })

    usage_tracker.record_serp(query=query, call_type=call_type,
                              results_returned=len(results), cost=0.0,
                              provider="europepmc")
    return results


async def _serp_search(query: str, num: int,
                        call_type: str = "standard") -> List[dict]:
    """
    Legacy SerpAPI path, kept for a deployment that still sets SERPAPI_API_KEY.

    `_patent_search` is the default; nothing calls this unless a key is present.
    """
    if not SERPAPI_API_KEY:
        return await _patent_search(query, num, call_type)
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.get("https://serpapi.com/search",
                               params={"engine": "google_patents", "q": query,
                                       "api_key": SERPAPI_API_KEY, "num": num})
        data    = resp.json()
        results = []
        if "error" in data:
            logger.warning("SerpAPI error: %s", data["error"])
        else:
            results = data.get("organic_results", [])
        usage_tracker.record_serp(query=query, call_type=call_type,
                                  results_returned=len(results))
        return results
    except Exception:
        logger.exception("SerpAPI call failed for: %s", query)
        usage_tracker.record_serp(query=query, call_type=call_type,
                                  results_returned=0)
        return []


async def search_patents_only(query: str, num_results: int = 5,
                               rerank: bool = True) -> List[dict]:
    is_title = _looks_like_title(query)
    if is_title:
        fetch_n    = max(20, num_results * 3)
        raw        = query.strip('"\'')
        # Two passes, not three: SerpAPI needed a bare, an `intitle:` and a quoted
        # variant because Google scored them differently. Europe PMC has one
        # title field, so the quoted pass returned exactly the `intitle:` set —
        # a third of the calls for no additional patents.
        r1, r2 = await asyncio.gather(
            _patent_search(raw, fetch_n, call_type="title_standard"),
            _patent_search(raw, fetch_n, call_type="title_intitle", is_title=True),
        )
        all_organic = r1 + r2
    else:
        fetch_n     = max(20, num_results * 2) if rerank else num_results
        all_organic = await _patent_search(query, fetch_n, call_type="standard")

    seen:    set  = set()
    results: list = []
    for p in all_organic:
        title   = p.get("title", "No title")
        raw_pid = extract_patent_from_serp_result(p)
        cleaned = clean_patent_id(str(raw_pid)) if raw_pid else ""
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        entry = {
            "Title":    title,
            "Snippet":  p.get("snippet", "")[:500],
            "Link":     f"https://patents.google.com/patent/{cleaned}/en",
            "PatentID": cleaned,
        }
        if is_title:
            entry["_title_sim"] = _title_similarity(query, title)
        results.append(entry)

    if not results:
        return []
    if rerank:
        try:
            return patent_reranker.rerank_pipeline(
                query, results, num_results,
                use_bm25=True, use_rrf=True, is_title_query=is_title)
        except Exception as e:
            logger.warning("Reranking failed: %s", e)
    return results[:num_results]

# ══════════════════════════════════════════════════════════════════════════════
#  §15  QUERY CACHE HELPER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

async def _get_patents_with_cache(query: str, num_results: int) -> List[dict]:
    cached_entry = _cache["queries"].get(query)
    if cached_entry:
        cached_results = cached_entry.get("results", [])
        if len(cached_results) >= num_results:
            logger.info("Full cache hit for query '%s'", query)
            return cached_results[:num_results]
        logger.info("Partial cache hit: have %d, need %d",
                    len(cached_results), num_results)
        cached_ids  = {p["PatentID"] for p in cached_results}
        extra       = await search_patents_only(query, num_results * 2, rerank=True)
        new_results = [p for p in extra if p.get("PatentID") not in cached_ids]
        combined    = (cached_results + new_results)[:num_results]
        _cache["queries"][query] = {"results": combined}
        return combined

    results = await search_patents_only(query, num_results, rerank=True)
    _cache["queries"][query] = {"results": results}
    _cache["latest_search"]  = {"query": query, "results": results,
                                 "timestamp": datetime.utcnow().isoformat()}
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  §16  PIPELINE RUNNER
#       Core change: is_patent_indexed() check now queries Qdrant rather than
#       an in-process OrderedDict, so the skip is truly persistent across
#       requests within the same session.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    patents:        List[dict]
    indexed_ids:    List[str]
    skipped_ids:    List[str]
    total_chunks:   int
    pipeline_steps: List[str]


async def run_pipeline(patent_list: List[dict]) -> PipelineResult:
    """
    Fetch HTML → parse → chunk → embed → upsert to Qdrant.
    Patents already in Qdrant are skipped entirely (no HTML fetch, no embed).
    """
    steps:   List[str] = []
    all_ids  = [p.get("PatentID") for p in patent_list if p.get("PatentID")]
    already  = vector_store.get_indexed_patent_ids()
    new_ids  = [pid for pid in all_ids if pid not in already]
    skipped  = [pid for pid in all_ids if pid in already]

    if skipped:
        steps.append(
            f"Skipped {len(skipped)} already-indexed (reusing Qdrant vectors): {skipped}"
        )

    if new_ids:
        steps.append(f"Fetching HTML for {len(new_ids)} new patents…")
        structured = await html_fetcher.process_multiple_patents(new_ids, max_concurrent=2)
        successful = {k: v for k, v in structured.items() if v}
        steps.append(f"Parsed: {list(successful.keys())}")
        if successful:
            steps.append("Chunking, embedding (HF API), and upserting to Qdrant…")
            all_chunks  = chunker.process_multiple_patents(successful)
            flat_chunks = [c for cs in all_chunks.values() for c in cs]
            vector_store.add_chunks(flat_chunks, batch_size=32)
            steps.append(f"Upserted {len(flat_chunks)} chunks to Qdrant")

    stats = vector_store.get_statistics()
    return PipelineResult(
        patents        = patent_list,
        indexed_ids    = list(vector_store.get_indexed_patent_ids()),
        skipped_ids    = skipped,
        total_chunks   = stats["total_chunks"],
        pipeline_steps = steps,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  §17  SINGLE GEMINI SYNTHESIS  (unchanged — only vector_store calls differ)
# ══════════════════════════════════════════════════════════════════════════════

async def _run_agent_synthesis(user_query: str,
                                patent_list: List[dict]) -> Tuple[str, str]:
    indexed_ids = list(vector_store.get_indexed_patent_ids())
    if not indexed_ids:
        return ("No patents were successfully indexed for analysis.",
                "Could not generate recommendations — no patents indexed.")

    context_parts = []
    for pid in indexed_ids[:MAX_PATENTS_IN_STORE]:
        chunks = vector_store.search_by_patent(user_query, pid, top_k=8)
        if not chunks:
            continue
        patent_meta = next((p for p in patent_list if p.get("PatentID") == pid), {})
        title   = patent_meta.get("Title", pid)
        section = f"\n{'='*60}\nPATENT: {pid}\nTITLE: {title}\n{'='*60}\n"
        for c in chunks:
            lbl = c["section_type"].upper()
            if c.get("metadata", {}).get("is_claim"):
                lbl = f"CLAIM #{c['metadata']['claim_number']} ({lbl})"
            section += f"[{lbl}]\n{c['text']}\n\n"
        context_parts.append(section)

    if not context_parts:
        return ("No relevant content could be retrieved from the indexed patents.",
                "Insufficient evidence for recommendations.")

    full_context = "\n".join(context_parts)

    prompt = f"""You are an expert patent analyst specialising in drug discovery, protein targets, and therapeutic areas.

A researcher has asked: "{user_query}"

Below is the content retrieved from {len(context_parts)} patent(s) from Google Patents.
Analyse this content and produce a comprehensive research report.

PATENT CONTENT:
{full_context}

Produce a structured report with these EXACT sections:

AGENT ANALYSIS:
A clear 3-4 sentence answer directly addressing whether the query is novel or already patented,
referencing specific patent IDs.

INDEPENDENT CLAIMS SUMMARY:
List the key independent claims found across the analysed patents most relevant to the query.

NOVELTY ASSESSMENT:
State clearly — High Novelty / Medium Novelty / Low Novelty — with a one paragraph justification
citing specific patents and claims.

FREEDOM-TO-OPERATE RISK:
State — High Risk / Medium Risk / Low Risk — with a one paragraph justification.

KEY FINDINGS:
Bullet list of 4-6 specific findings from the patent content.

RECOMMENDED NEXT STEPS:
Numbered list of 5-7 actionable research steps the researcher should take next.

GAPS IN CURRENT ANALYSIS:
What is missing, uncertain, or requires further investigation.

Rules:
- Use plain text. No asterisks. No markdown.
- CAPITALISE key drug names, protein targets, and mechanisms.
- Reference specific patent IDs when making claims.
- If a section cannot be answered from the content, state clearly why."""

    try:
        full_response = await _gemini_call(
            prompt, call_type="agent_run", query=user_query)
        split_marker = "RECOMMENDED NEXT STEPS"
        if split_marker in full_response:
            idx             = full_response.index(split_marker)
            agent_answer    = full_response[:idx].strip()
            recommendations = full_response[idx:].strip()
        else:
            agent_answer    = full_response
            recommendations = ""
        return agent_answer, recommendations
    except Exception as e:
        logger.exception("Agent synthesis failed")
        return (f"Analysis could not be completed: {e}",
                f"Recommendations unavailable: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  §18  FOLLOW-UP RAG  (unchanged — vector_store interface identical)
# ══════════════════════════════════════════════════════════════════════════════

async def _rag_single_patent(query: str, patent_id: str, top_k: int = 10) -> dict:
    chunks = vector_store.search_by_patent(query, patent_id, top_k)
    q_low  = query.lower()
    is_meta = any(kw in q_low for kw in _METADATA_KEYWORDS)
    if is_meta:
        meta = vector_store.get_metadata_chunks(patent_id)
        if meta:
            chunks = meta + [c for c in chunks if c.get("section_type") != "metadata"]
    if not chunks:
        return {"answer": f"No relevant content found in patent {patent_id}.",
                "chunks_used": 0}

    ctx_parts = []
    for c in chunks:
        lbl = c["section_type"].upper()
        if c.get("metadata", {}).get("is_claim"):
            lbl = f"CLAIM #{c['metadata']['claim_number']} ({lbl})"
        ctx_parts.append(f"[{lbl}]\n{c['text']}\n")
    ctx = "\n".join(ctx_parts)

    if is_meta:
        prompt = (f"You are a patent analyst. Answer the question about patent {patent_id} "
                  f"using only the metadata.\nSTRICT RULES:\n- Answer in 1-2 sentences only\n"
                  f"- Use exact names from metadata\n\nMETADATA:\n{ctx}\n\nQUESTION: {query}")
    else:
        is_claims = any(w in q_low for w in ["claim","claims","claiming"])
        claim_chunks = [c for c in chunks if c.get("metadata", {}).get("is_claim")]
        if is_claims:
            cl_ctx = "\n".join(
                f"Claim #{c['metadata']['claim_number']} ({c['metadata'].get('claim_type','')}):\n{c['text'][:400]}"
                for c in claim_chunks[:5])
            prompt = (f"You are a patent claims analyst. Patent: {patent_id}\n{_FORMAT_RULES}\n\n"
                      f"CLAIMS:\n{cl_ctx}\nCONTEXT:\n{ctx}\nQUESTION: {query}")
        else:
            prompt = (f"You are a patent analyst. Patent: {patent_id}\n{_FORMAT_RULES}\n\n"
                      f"Use the top relevant retrieved chunks below to answer the question.\n\n"
                      f"PATENT CONTENT:\n{ctx}\nQUESTION: {query}")

    try:
        answer = await _gemini_call(prompt, call_type="ask", query=query)
        return {"answer": answer, "chunks_used": len(chunks)}
    except Exception as e:
        return {"answer": f"Could not answer: {e}", "chunks_used": 0}


async def _rag_all_patents(query: str, top_k: int = 10) -> dict:
    all_pids = list(vector_store.get_indexed_patent_ids())
    if not all_pids:
        return {"answer": "No patents indexed.", "patents_analyzed": []}

    context_parts = []
    valid_pids    = []
    total_chunks  = 0
    for pid in all_pids[:MAX_PATENTS_IN_STORE]:
        chunks = vector_store.search_by_patent(query, pid, top_k)
        if not chunks:
            continue
        section = f"\nPATENT {pid}:\n"
        for c in chunks:
            section += f"[{c['section_type'].upper()}] {c['text'][:400]}\n"
        context_parts.append(section)
        valid_pids.append(pid)
        total_chunks += len(chunks)

    if not context_parts:
        return {"answer": "No relevant content found.", "patents_analyzed": []}

    prompt = (f"You are a patent analyst. Synthesise the following patent content "
              f"to answer the question: '{query}'\n\n"
              f"{''.join(context_parts)}\n\n{_FORMAT_RULES}\n\n"
              f"Provide a unified answer referencing specific patent IDs where relevant.")
    try:
        answer = await _gemini_call(prompt, call_type="ask_all", query=query)
        return {"answer": answer, "patents_analyzed": valid_pids,
                "chunks_used": total_chunks}
    except Exception as e:
        return {"answer": f"Could not synthesise: {e}", "patents_analyzed": []}

# ══════════════════════════════════════════════════════════════════════════════
#  §19  ROUTER  (unchanged — added "backend" field to /agent/status)
# ══════════════════════════════════════════════════════════════════════════════

router = APIRouter()


@router.post("/agent/run")
async def run_agent(user_query: str, num_results: int = 5,
                    db: Session = Depends(get_db)):
    """
    PRIMARY ENDPOINT — Full novelty search pipeline + one Gemini synthesis.
    Steps:
      1. Search Google Patents + rerank           [no Gemini]
      2. Skip already-indexed patents (Qdrant)    [no Gemini, no HF API]
      3. Fetch HTML + parse + chunk + embed       [HF API for new patents only]
      4. Upsert vectors to Qdrant                 [cloud storage]
      5. ONE Gemini call → full analysis report   [tracked in /agent/usage]
    """
    if not user_query.strip():
        raise HTTPException(400, "user_query cannot be empty")
    if num_results < 1 or num_results > 20:
        raise HTTPException(400, "num_results must be between 1 and 20")

    patent_list = await _get_patents_with_cache(user_query, num_results)
    if not patent_list:
        raise HTTPException(404, "No patents found. Try a different search term.")

    pipeline_result = await run_pipeline(patent_list)
    agent_answer, recommendations = await _run_agent_synthesis(user_query, patent_list)

    try:
        db.add(AgentRun(user_query=user_query,
                        patents_used=json.dumps(pipeline_result.indexed_ids),
                        final_answer=agent_answer))
        db.add(SearchHistory(query=user_query, source="Agent Run",
                             results=json.dumps([p.get("PatentID") for p in patent_list]),
                             chatbot_answer=agent_answer))
        db.commit()
    except Exception:
        logger.exception("DB write failed")

    return {
        "agent_answer":    agent_answer,
        "recommendations": recommendations,
        "patents_used":    patent_list,
        "pipeline_steps":  pipeline_result.pipeline_steps,
        "total_patents":   len(pipeline_result.indexed_ids),
        "total_chunks":    pipeline_result.total_chunks,
    }


@router.get("/agent/ask")
async def agent_ask(query: str, patent_id: str, top_k: int = 10,
                    db: Session = Depends(get_db)):
    """Follow-up question about a specific patent. One Gemini call (tracked)."""
    if not query.strip() or not patent_id:
        raise HTTPException(400, "query and patent_id are required")
    pid = clean_patent_id(patent_id)
    if not vector_store.is_patent_indexed(pid):
        raise HTTPException(400, f"Patent {pid} not indexed. Run /agent/run first.")
    result = await _rag_single_patent(query, pid, top_k)
    try:
        db.add(SearchHistory(query=query, source=f"Ask - {pid}",
                             results=f"chunks={result['chunks_used']}",
                             chatbot_answer=result["answer"]))
        db.commit()
    except Exception:
        pass
    return {"answer": result["answer"], "patent_id": pid,
            "chunks_used": result.get("chunks_used", 0), "mode": "single_patent"}


@router.get("/agent/ask_all")
async def agent_ask_all(query: str, top_k: int = 10,
                        db: Session = Depends(get_db)):
    """Follow-up question across ALL indexed patents. One Gemini call (tracked)."""
    if not query.strip():
        raise HTTPException(400, "query cannot be empty")
    if not vector_store.get_indexed_patent_ids():
        raise HTTPException(400, "No patents indexed. Run /agent/run first.")
    result = await _rag_all_patents(query, top_k)
    try:
        db.add(SearchHistory(query=query, source="Ask All",
                             results=json.dumps(result.get("patents_analyzed", [])),
                             chatbot_answer=result["answer"]))
        db.commit()
    except Exception:
        pass
    return {"answer": result["answer"],
            "patents_analyzed": result.get("patents_analyzed", []),
            "chunks_used": result.get("chunks_used", 0),
            "mode": "multi_patent"}


@router.get("/agent/status")
def agent_status():
    stats = vector_store.get_statistics()
    html_ = html_fetcher.cache_stats()
    return {
        "vector_store": stats,
        "html_cache":   html_,
        "query_cache":  {"cached_queries": list(_cache["queries"].keys()),
                         "count": len(_cache["queries"])},
        "patent_cap":   MAX_PATENTS_IN_STORE,
        "embedding_backend": f"huggingface_inference_api/{HF_EMBEDDING_MODEL}",
        "vector_backend":    f"qdrant_cloud/{QDRANT_COLLECTION_NAME}",
    }


@router.get("/agent/health")
def agent_health():
    uptime = (datetime.utcnow() -
              datetime.fromisoformat(usage_tracker.session_start)).total_seconds()
    return {
        "status":          "healthy",
        "timestamp":       datetime.utcnow().isoformat(),
        "uptime_seconds":  uptime,
    }


@router.get("/agent/usage")
def agent_usage():
    """Full usage summary: Gemini tokens + SerpAPI calls + combined cost."""
    return usage_tracker.summary()


@router.get("/agent/history")
def agent_history(db: Session = Depends(get_db)):
    rows = db.query(AgentRun).order_by(AgentRun.created_at.desc()).all()
    return [{"id": r.id, "user_query": r.user_query,
             "patents_used": json.loads(r.patents_used or "[]"),
             "final_answer": r.final_answer,
             "created_at":   r.created_at.isoformat()} for r in rows]


@router.get("/agent/download_history")
def download_history(db: Session = Depends(get_db)):
    rows   = db.query(SearchHistory).order_by(SearchHistory.created_at.desc()).all()
    output = io.StringIO()
    w      = csv.writer(output)
    w.writerow(["ID","Query","Source","Answer","Time"])
    for h in rows:
        w.writerow([h.id, h.query, h.source, h.chatbot_answer or "",
                    h.created_at.isoformat()])
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="novelty_history.csv"'},
    )