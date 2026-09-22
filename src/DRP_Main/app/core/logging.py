"""
Centralised logging setup for the entire application.
Call setup_logging() once at startup from main.py.
"""
import logging
import os
from pathlib import Path


def _ensure_log_dir() -> str:
    # Honor LOG_DIR (e.g. /tmp on read-only hosts like Databricks Apps),
    # otherwise default to the package-local logs/ directory.
    log_dir = Path(os.environ.get("LOG_DIR") or (Path(__file__).resolve().parent.parent / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with file + stream handlers.
    Must be called before any module-level loggers are created.

    On read-only filesystems (e.g. Databricks Apps) file logging is
    unavailable; we degrade gracefully to stream-only logging so the
    application can still boot.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        log_file = os.path.join(_ensure_log_dir(), "api.log")
        handlers.insert(0, logging.FileHandler(log_file))
    except OSError:
        # Read-only FS / no writable log dir — continue with stream logging only.
        logging.getLogger(__name__).warning("file logging unavailable — using stream logging only")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )

    # Pipe uvicorn logs through the same handlers
    for uvicorn_logger in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_log = logging.getLogger(uvicorn_logger)
        uv_log.handlers = logging.getLogger().handlers


def get_logger(name: str | None = None) -> logging.Logger:
    """Convenience wrapper — returns a named logger."""
    return logging.getLogger(name or __name__)


def init_langfuse():
    """
    Initialise and return a Langfuse client using settings.
    Import lazily to avoid import errors when Langfuse is not installed.
    """
    try:
        from langfuse import Langfuse
        from DRP_Main.app.core.config import settings

        if not settings.LANGFUSE_SECRET_KEY:
            return None

        return Langfuse(
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except ImportError:
        logging.getLogger(__name__).warning("langfuse package not installed — tracing disabled")
        return None
