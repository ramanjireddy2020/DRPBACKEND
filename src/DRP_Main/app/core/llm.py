"""LLM client wrapper for Groq, Gemini, and Databricks Foundation Model API providers."""
import re
from typing import Any, Dict, List, Optional, Union

import httpx
from groq import Groq
from google import genai

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    """
    Foundation-model chat endpoints (unlike Groq's JSON mode) routinely wrap
    JSON answers in a markdown code fence even when told to return raw JSON —
    strip it so callers can json.loads() the content directly.
    """
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self):
        self._groq_client: Optional[Groq] = None
        self._gemini_client: Optional[genai.Client] = None

        if settings.GROQ_API_KEY:
            self._groq_client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("Groq client initialized")

        if settings.GOOGLE_API_KEY:
            self._gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            logger.info("Gemini client initialized")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Ask whichever provider this deployment actually has, in preference order.

        Databricks Model Serving first — it is the platform's own, and keeps the
        request inside the workspace. Then Groq, then Gemini.

        The fallback exists because `databricks()` alone is gated on *both*
        DATABRICKS_HOST and DATABRICKS_TOKEN, and the deployed app sets only the
        host. Callers that asked for Databricks directly got an exception, caught
        it, and quietly degraded: article chat returned the abstract's opening
        sentences to every question, and every LLM-written line on the platform
        fell back to templated text — a 200 response that looked like a broken
        feature rather than a missing credential.

        Raises only when no provider is configured at all, so a caller can still
        tell "nothing to ask" from "the answer was unhelpful".
        """
        errors: List[str] = []

        if settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN:
            try:
                return self.databricks(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 — try the next provider
                errors.append(f"databricks: {exc}")

        if self._groq_client:
            try:
                # Groq rejects the serving-endpoint-only arguments.
                allowed = {k: v for k, v in kwargs.items()
                           if k in ("temperature", "max_tokens", "top_p", "stop")}
                return self.groq(messages, **allowed)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"groq: {exc}")

        if self._gemini_client:
            try:
                # Gemini takes one prompt, so the turns are flattened. System
                # content leads, which is how it is weighted anyway.
                prompt = "\n\n".join(
                    str(m.get("content", "")) for m in messages if m.get("content")
                )
                return self.gemini(prompt)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"gemini: {exc}")

        raise RuntimeError(
            "No LLM provider available — "
            + ("; ".join(errors) if errors
               else "set DATABRICKS_TOKEN, GROQ_API_KEY or GOOGLE_API_KEY")
        )

    def groq(self, messages: List[Any], model: Optional[str] = None, **kwargs) -> str:
        """Query Groq LLM."""
        if not self._groq_client:
            raise RuntimeError("Groq client not configured")

        model_name = model or settings.GROQ_MODEL or "llama-3.3-70b-versatile"
        try:
            response = self._groq_client.chat.completions.create(
                messages=messages,
                model=model_name,
                **kwargs
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            logger.error("Groq call failed: %s", e)
            raise

    def gemini(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Query Gemini LLM."""
        if not self._gemini_client:
            raise RuntimeError("Gemini client not configured")

        model_name = model or settings.GEMINI_MODEL or "gemini-2.5-flash"
        try:
            response = self._gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                **kwargs
            )
            return response.text if response.text else ""
        except Exception as e:
            logger.error("Gemini call failed: %s", e)
            raise

    def _databricks_host_and_headers(self) -> tuple[str, Dict[str, str]]:
        """
        Resolve the workspace host + auth headers for calling a serving endpoint.
        Prefers static DATABRICKS_HOST/DATABRICKS_TOKEN (local dev / PAT profile);
        on Databricks Apps neither is injected under those names, so falls back to
        the SDK's default auth chain — the same auto-auth pattern the orchestration
        router already relies on for the app's own service-principal credentials.
        """
        if settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN:
            return settings.DATABRICKS_HOST.rstrip("/"), {
                "Authorization": f"Bearer {settings.DATABRICKS_TOKEN}"
            }
        from databricks.sdk import WorkspaceClient

        cfg = WorkspaceClient().config
        return cfg.host.rstrip("/"), cfg.authenticate()

    def databricks(self, messages: List[Dict[str, str]], endpoint: Optional[str] = None,
                    return_raw: bool = False, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Query a Databricks pay-per-token Foundation Model API serving endpoint.

        Uses the OpenAI-compatible chat-completions shape every `llm/v1/chat` serving
        endpoint accepts — no idle cost, no GPU provisioning, unlike a custom endpoint.
        `return_raw=True` returns the full decoded response body (choices/usage/
        finish_reason) for callers that need more than the completion text.
        """
        host, headers = self._databricks_host_and_headers()
        model_name = endpoint or settings.DATABRICKS_LLM_ENDPOINT
        url = f"{host}/serving-endpoints/{model_name}/invocations"
        try:
            resp = httpx.post(
                url,
                headers={**headers, "Content-Type": "application/json"},
                json={"messages": messages, **kwargs},
                timeout=60.0,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"].get("content") or ""
            stripped = _strip_markdown_fence(content)
            if stripped != content:
                body["choices"][0]["message"]["content"] = stripped
            if return_raw:
                return body
            return stripped
        except Exception as e:
            logger.error("Databricks serving call failed: %s", e)
            raise


# Global singleton
llm_client = LLMClient()