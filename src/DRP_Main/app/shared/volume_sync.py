"""
Mirror a Unity Catalog volume onto local disk.

Databricks Apps do **not** expose UC volumes as a POSIX filesystem — unlike
clusters and jobs, the container has no FUSE layer, so `/Volumes/...` exists as
a stub and every read fails with `PermissionError: [Errno 13] ... '/Volumes'`,
even with READ_VOLUME granted and the volume attached as an app resource. The
only way in is the Files API.

TxKG also *writes* its caches (`cache_optimized3.pkl`, `annotation_edges.pkl`,
`rwrh_transition.pkl`) next to the dataset, so a read-only volume could never
have served as `BIOKG_DATA_DIR` regardless. Mirroring to a writable local dir
solves both problems at once.

`ensure_local_dataset()` is idempotent: a file already present with a matching
size is left alone, so only a cold container pays the download.
"""
from __future__ import annotations

import os
import shutil
import threading

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_synced: dict[str, str] = {}

#: Where a mirrored volume lands. /tmp is the only writable path on Apps.
LOCAL_ROOT = os.environ.get("VOLUME_CACHE_DIR", "/tmp/volume_cache")


def is_volume_path(path: str) -> bool:
    return bool(path) and path.replace("\\", "/").startswith("/Volumes/")


def _client():
    from databricks.sdk import WorkspaceClient

    # On Apps the SDK authenticates as the app's service principal from the
    # injected environment — no explicit credentials needed.
    return WorkspaceClient()


def sync_volume_dir(volume_path: str, local_dir: str) -> str:
    """
    Download every file in `volume_path` into `local_dir`, skipping files that
    already exist at the same size. Returns `local_dir`.
    """
    w = _client()
    os.makedirs(local_dir, exist_ok=True)

    entries = list(w.files.list_directory_contents(volume_path))
    logger.info("volume %s: %d entries to mirror -> %s", volume_path, len(entries), local_dir)

    for e in entries:
        name = os.path.basename(e.path.rstrip("/"))
        if getattr(e, "is_directory", False):
            sync_volume_dir(e.path, os.path.join(local_dir, name))
            continue

        dest = os.path.join(local_dir, name)
        remote_size = getattr(e, "file_size", None)
        if os.path.exists(dest) and remote_size is not None and os.path.getsize(dest) == remote_size:
            continue

        logger.info("downloading %s (%s bytes)", e.path, remote_size)
        resp = w.files.download(e.path)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(resp.contents, fh)
        os.replace(tmp, dest)  # never leave a half-written file at the real name

    return local_dir


def ensure_local_dataset(env_var: str, subdir: str) -> str:
    """
    Resolve `env_var` to a locally readable directory.

    If it does not point at a UC volume, it is returned untouched — local and
    cluster deployments keep working unchanged. If it does, the volume is
    mirrored once per process and `env_var` is rewritten to the local path, so
    modules that read it at import time (txkg_test.py does) see the local copy.
    """
    path = os.environ.get(env_var, "")
    if not is_volume_path(path):
        return path

    with _lock:
        if path in _synced:
            return _synced[path]

        local_dir = os.path.join(LOCAL_ROOT, subdir)
        sync_volume_dir(path, local_dir)
        os.environ[env_var] = local_dir  # before the consuming module is imported
        _synced[path] = local_dir
        logger.info("%s remapped %s -> %s", env_var, path, local_dir)
        return local_dir
