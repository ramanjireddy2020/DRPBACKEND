"""
NovSearch — hosted-model clients (spec §3, §4, §5).

Every model NovSearch uses is self-hosted on Databricks Model Serving:

  * Embeddings          → 1024-dim (Tool 2) — GTE-large-en-v1.5 on this workspace
  * Cross-encoder rerank → (query, candidate) relevance scores (Tool 1)
  * SaulLM (legal/patent-domain) → synthesis and QA (Tool 3), with Groq
    (`core/llm.py`, already used elsewhere on the platform) as the fallback
    when the serving endpoint is unavailable or rate-limited.

Embeddings and cross-encoder go through the same
`/serving-endpoints/{name}/invocations` REST call, so there is one auth path
and one timeout policy for both.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class ModelServingError(RuntimeError):
    """A Databricks Model Serving endpoint could not be reached or answered."""


# ══════════════════════════════════════════════════════════════════════════════
#  Shared serving transport
# ══════════════════════════════════════════════════════════════════════════════

_workspace_client = None
_workspace_client_init_failed = False


def _get_workspace_client():
    """
    Lazily build a WorkspaceClient the same way `uc_store.py` does: prefer a named
    `.databrickscfg` profile (local dev on workspaces where PAT creation is
    disabled by policy), then explicit host+token, then the SDK's default chain —
    which inside a deployed Databricks App resolves automatically to the app's
    own service-principal credentials, needing neither a profile nor a token.
    """
    global _workspace_client, _workspace_client_init_failed
    if _workspace_client is not None or _workspace_client_init_failed:
        return _workspace_client
    try:
        from databricks.sdk import WorkspaceClient

        if settings.DATABRICKS_CONFIG_PROFILE:
            _workspace_client = WorkspaceClient(profile=settings.DATABRICKS_CONFIG_PROFILE)
        elif settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN:
            _workspace_client = WorkspaceClient(host=settings.DATABRICKS_HOST, token=settings.DATABRICKS_TOKEN)
        else:
            _workspace_client = WorkspaceClient()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("databricks_clients: WorkspaceClient unavailable: %s", exc)
        _workspace_client_init_failed = True
    return _workspace_client


def _serving_url(endpoint: str) -> str:
    client = _get_workspace_client()
    host = (settings.DATABRICKS_HOST or (client.config.host if client else "") or "").rstrip("/")
    if not host:
        raise ModelServingError(
            "No Databricks host resolved — NovSearch needs a workspace for "
            "Model Serving (embeddings, cross-encoder, SaulLM)."
        )
    return f"{host}/serving-endpoints/{endpoint}/invocations"


def _headers() -> Dict[str, str]:
    client = _get_workspace_client()
    if client is None:
        raise ModelServingError(
            "No Databricks auth available (no profile, no DATABRICKS_HOST/TOKEN, "
            "and the SDK's default credential chain found nothing)."
        )
    auth_headers = dict(client.config.authenticate())
    auth_headers["Content-Type"] = "application/json"
    return auth_headers


async def _invoke(endpoint: str, payload: Dict[str, Any]) -> Any:
    """POST to a serving endpoint and return the decoded JSON body."""
    url = _serving_url(endpoint)
    try:
        async with httpx.AsyncClient(timeout=settings.NOVSEARCH_REQUEST_TIMEOUT) as client:
            resp = await client.post(url, headers=_headers(), json=payload)
    except httpx.HTTPError as exc:
        raise ModelServingError(f"Serving endpoint '{endpoint}' unreachable: {exc}") from exc
    if resp.status_code >= 400:
        raise ModelServingError(
            f"Serving endpoint '{endpoint}' returned {resp.status_code}: {resp.text[:400]}"
        )
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
#  §4 step 4 — BGE-large embeddings
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingClient:
    """
    BGE-large-en-v1.5 on Databricks Model Serving.

    1024-dimensional, L2-normalised, and prefixed per BGE's retrieval convention:
    `"query: "` for search queries, `"passage: "` for stored chunks. The store is
    cosine, so normalising here makes cosine equivalent to a dot product.
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        self.endpoint = endpoint or settings.NOVSEARCH_EMBEDDING_ENDPOINT
        self.dim = settings.NOVSEARCH_EMBEDDING_DIM

    async def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype="float32")
        # `llm/v1/embeddings` (OpenAI-compatible) endpoints — e.g. GTE-large —
        # take "input", not "inputs".
        body = await _invoke(self.endpoint, {"input": texts})
        arr = _coerce_embeddings(body, self.dim)
        if arr.shape[0] != len(texts):
            raise ModelServingError(
                f"Embedding endpoint returned {arr.shape[0]} vectors for {len(texts)} inputs."
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    async def embed_query(self, query: str) -> List[float]:
        vectors = await self._embed([f"query: {query}"])
        return vectors[0].tolist()

    async def embed_passages(self, texts: List[str]) -> List[List[float]]:
        batch = max(1, settings.NOVSEARCH_EMBED_BATCH)
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            chunk = [f"passage: {t}" for t in texts[i : i + batch]]
            out.extend(v.tolist() for v in await self._embed(chunk))
        return out


def _coerce_embeddings(body: Any, dim: int) -> np.ndarray:
    """
    Normalise the several shapes a serving endpoint may answer with:
    `{"predictions": [...]}, {"data": [{"embedding": [...]}]}, {"outputs": [...]}`
    or a bare list. Token-level output `[batch, seq, hidden]` is reduced to the
    CLS token, which is what BGE's sentence embedding is.
    """
    payload = body
    if isinstance(body, dict):
        for key in ("predictions", "outputs", "embeddings", "data", "result"):
            if key in body:
                payload = body[key]
                break
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = [item.get("embedding", item.get("vector")) for item in payload]

    arr = np.array(payload, dtype="float32")
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ModelServingError(
            f"Embedding endpoint returned shape {arr.shape}; expected (n, {dim})."
        )
    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  §3 step 6 — MiniLM cross-encoder
# ══════════════════════════════════════════════════════════════════════════════

class CrossEncoderClient:
    """MiniLM cross-encoder on Databricks Model Serving — one score per pair."""

    def __init__(self, endpoint: Optional[str] = None) -> None:
        self.endpoint = endpoint or settings.NOVSEARCH_CROSS_ENCODER_ENDPOINT

    async def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        body = await _invoke(
            self.endpoint,
            {"inputs": [{"text": q, "text_pair": d} for q, d in pairs]},
        )
        payload = body
        if isinstance(body, dict):
            for key in ("predictions", "outputs", "scores", "data", "result"):
                if key in body:
                    payload = body[key]
                    break
        scores: List[float] = []
        for item in payload:
            if isinstance(item, dict):
                item = item.get("score", item.get("logit", item.get("prediction", 0.0)))
            if isinstance(item, (list, tuple)):
                item = item[0]
            scores.append(float(item))
        if len(scores) != len(pairs):
            raise ModelServingError(
                f"Cross-encoder returned {len(scores)} scores for {len(pairs)} pairs."
            )
        return scores


# ══════════════════════════════════════════════════════════════════════════════
#  §5 — patent-domain LLM (SaulLM) with Bedrock Llama 3.3 70B fallback
# ══════════════════════════════════════════════════════════════════════════════

class PatentLLMClient:
    """
    One synthesis call per report or QA turn.

    Primary: SaulLM (or whatever's configured) on Databricks Model Serving.
    Fallback: Groq — already a first-class provider on this platform (`core/llm.py`,
    used by LitMineX/CurateX), so a cold or rate-limited serving endpoint degrades
    to a call this codebase already knows how to make, no separate cloud creds.
    `generate` returns the text *and* the model identifier that produced it, so
    `model_used` in the report reflects what actually answered rather than what
    was configured.
    """

    def __init__(self) -> None:
        self.endpoint = settings.NOVSEARCH_LLM_ENDPOINT
        self.primary_name = settings.NOVSEARCH_LLM_NAME
        self.fallback_name = f"groq-{settings.GROQ_MODEL}"

    async def generate(self, prompt: str) -> Tuple[str, str]:
        try:
            text = await self._saullm(prompt)
            return _strip_markdown(text), self.primary_name
        except Exception as exc:  # noqa: BLE001 — any failure is a fallback trigger
            logger.warning("Model Serving LLM call failed (%s); falling back to Groq", exc)
        text = await self._groq(prompt)
        return _strip_markdown(text), self.fallback_name

    async def _saullm(self, prompt: str) -> str:
        body = await _invoke(
            self.endpoint,
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": settings.NOVSEARCH_LLM_MAX_TOKENS,
                "temperature": 0.0,
            },
        )
        return _extract_completion(body)

    async def _groq(self, prompt: str) -> str:
        import asyncio

        from DRP_Main.app.core.llm import llm_client  # lazy: matches the module's lazy-import convention

        def _call() -> str:
            return llm_client.groq(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.NOVSEARCH_LLM_MAX_TOKENS,
                temperature=0.0,
            )

        return await asyncio.to_thread(_call)


def _extract_completion(body: Any) -> str:
    """Pull the text out of an OpenAI-style, MLflow-style or bare-string response."""
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        choices = body.get("choices")
        if choices:
            first = choices[0]
            message = first.get("message") or {}
            return message.get("content") or first.get("text") or ""
        for key in ("predictions", "outputs", "generation", "candidates", "result"):
            if key in body:
                value = body[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, list) and value:
                    item = value[0]
                    if isinstance(item, str):
                        return item
                    if isinstance(item, dict):
                        return item.get("text") or item.get("generated_text") or ""
    raise ModelServingError("Could not read a completion out of the LLM response.")


def _strip_markdown(text: str) -> str:
    """The report format is plain text — asterisks would leak into the wire object."""
    return (text or "").replace("*", "").strip()


# Singletons — stateless HTTP wrappers, safe to share across requests.
embedding_client = EmbeddingClient()
cross_encoder_client = CrossEncoderClient()
llm_client = PatentLLMClient()
