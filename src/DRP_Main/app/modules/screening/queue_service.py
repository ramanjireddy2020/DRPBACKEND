"""
Class-based queue for protein processing tasks.
Replaces app/services/queue_service.py — fixes global mutable state with a proper class.
"""
import os
import queue
import threading
import uuid
from typing import Optional

from fastapi.encoders import jsonable_encoder

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.screening.pipeline_service import (
    get_protein_status_by_list,
    is_protein_processed,
    protein_preparation,
)
from DRP_Main.app.modules.screening.schemas import DrugBase, ProteinBase
from DRP_Main.app.shared.file_utils import protein_output_directory, write_json_file

logger = get_logger(__name__)


class QueueService:
    """
    Thread-safe queue that processes protein docking tasks sequentially.

    One worker thread at a time, tracked by `_worker_running` rather than by
    `Thread.is_alive()`. That distinction matters: a worker that has decided to
    exit is still "alive" for a short while as it unwinds, so an `is_alive()`
    check could see a dying worker, decline to start a replacement, and leave
    the task sitting in the queue forever. The exit decision and the enqueue
    decision are therefore made under the same lock.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._is_paused: bool = False
        self._processing_thread: Optional[threading.Thread] = None
        self._worker_running: bool = False
        self._current_task: Optional[ProteinBase] = None
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_to_queue(self, protein_record: ProteinBase, drug_records: list[DrugBase]) -> dict:
        """Enqueue a protein processing task and ensure the worker is running."""
        task_id = str(uuid.uuid4())
        protein_record = protein_record.model_copy(update={"task_id": task_id})
        self._queue.put((protein_record, drug_records, task_id))
        self._ensure_worker_running()
        logger.info("Task queued: %s (id=%s)", protein_record.protein_name, task_id)
        return {"status": "queued", "task_id": task_id}

    def stop_processing(self) -> dict:
        with self._lock:
            self._is_paused = True
        return {"status": "stopped"}

    def resume_processing(self) -> dict:
        with self._lock:
            self._is_paused = False
        self._ensure_worker_running()
        return {"status": "resumed"}

    def get_queue_status(self) -> dict:
        with self._lock:
            current = self._current_task
        return {
            "queue_size": self._queue.qsize(),
            "currently_processing": jsonable_encoder(current) if current else None,
        }

    def get_all_tasks(self) -> dict:
        queue_list = list(self._queue.queue)
        with self._lock:
            current = self._current_task
        return {
            "queue_size": self._queue.qsize(),
            "current_task": jsonable_encoder(current) if current else None,
            "queue": jsonable_encoder(queue_list),
        }

    def get_current_processing_state(self, protein_records: list[ProteinBase]) -> dict:
        with self._lock:
            current = self._current_task
        queue_count = self._queue.qsize()

        if queue_count == 0 and current is None:
            protein_data = get_protein_status_by_list(
                [p.protein_name for p in protein_records]
            )
            return {"queue_size": 0, "proteins": protein_data}

        protein_names_in_queue = {task[0].protein_name for task in self._queue.queue}
        current_name = current.protein_name if current else None

        for protein in protein_records:
            if protein.protein_name == current_name:
                protein.status = "Processing"
            elif protein.protein_name in protein_names_in_queue:
                protein.status = "Queued"
            else:
                done, data = is_protein_processed(protein)
                if done and data is not None:
                    protein.status = data.status
                    updated = data.model_dump()
                    for field, value in updated.items():
                        setattr(protein, field, value)

        return {
            "queue_size": queue_count,
            "proteins": [jsonable_encoder(p) for p in protein_records],
        }

    # ── Internal worker ────────────────────────────────────────────────────────

    def _ensure_worker_running(self) -> None:
        """Start a worker unless one is already running. Never starts a second."""
        with self._lock:
            if self._worker_running or self._is_paused:
                return
            self._worker_running = True
            self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._processing_thread.start()

    def _process_queue(self) -> None:
        while True:
            # Deciding to stop and clearing the running flag happen together,
            # under the lock `add_to_queue` takes — so a task enqueued during
            # the handover either finds this worker still flagged as running
            # (and is picked up below) or starts a new one.
            with self._lock:
                if self._is_paused or self._queue.empty():
                    self._worker_running = False
                    return
                protein_record, drug_records, task_id = self._queue.get()
                self._current_task = protein_record

            try:
                protein_preparation(protein_record, drug_records, task_id)
            except Exception as exc:  # noqa: BLE001 — one task must not kill the worker
                logger.error("Error processing task %s: %s", task_id, exc)
                self._record_failure(protein_record, exc)
            finally:
                with self._lock:
                    self._current_task = None
                self._queue.task_done()

    @staticmethod
    def _record_failure(protein_record: ProteinBase, exc: Exception) -> None:
        """
        Persist a failed task's status to disk.

        Without this a crashed task leaves no trace: the in-memory queue entry
        is gone, no status JSON was ever written, and `/list-processed-proteins`
        and the WebSocket both report the protein as simply absent rather than
        failed — the task appears to have silently vanished.
        """
        try:
            failed = protein_record.model_copy(update={"status": f"Failed: {exc}"})
            output_dir = protein_output_directory(protein_record.protein_name)
            os.makedirs(output_dir, exist_ok=True)
            write_json_file(
                os.path.join(output_dir, f"{protein_record.protein_name}.json"),
                failed.model_dump(),
            )
        except Exception as write_error:  # noqa: BLE001 — reporting must not raise
            logger.warning(
                "Could not persist failure for %s: %s",
                protein_record.protein_name,
                write_error,
            )


# Module-level singleton — replaces all the old module-level globals
queue_service = QueueService()
