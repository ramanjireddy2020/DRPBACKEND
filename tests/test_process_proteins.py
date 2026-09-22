"""
ScreenSuite legacy surface — Excel upload, `/process-protein/`, the queue, and
the status WebSocket.

This replaces a hand-run smoke script that pointed at one developer's VM
(`vishnu-virtual-machine:8000`), read a workbook from their home directory, did
its HTTP POST at *import* time (so pytest failed during collection, taking the
whole suite with it), and ended in a `while True` WebSocket loop that could
never terminate. None of it could pass anywhere.

These tests drive the same endpoints in-process through `TestClient`: no server,
no network, no external binaries. Docking itself is stubbed — what is under test
is the queue's state machine and the HTTP/WebSocket contracts around it.

Run:
    cd src && python -m pytest ../tests/test_process_proteins.py --noconftest -q
"""
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from DRP_Main.app.modules.screening import (  # noqa: E402
    download_service,
    pipeline_service,
    queue_service as queue_module,
)
from DRP_Main.app.modules.screening.enums import ProteinProcessStatus  # noqa: E402
from DRP_Main.app.modules.screening.queue_service import QueueService  # noqa: E402
from DRP_Main.app.modules.screening.router import router  # noqa: E402
from DRP_Main.app.modules.screening.schemas import DrugBase, ProteinBase  # noqa: E402
from DRP_Main.app.shared.file_utils import protein_output_directory  # noqa: E402

PROTEIN = "AcrB"
DRUG = "Chlorthalidone"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    """
    Redirect all output to a temp directory.

    Without this the legacy path writes into the repo's own `app/data`, so a
    test run would leave protein status files behind and later runs would read
    them back as cached results.
    """
    monkeypatch.setattr(
        "DRP_Main.app.core.config.settings.SCREENING_OUTPUT_ROOT", str(tmp_path), raising=False
    )
    return tmp_path


@pytest.fixture()
def fresh_queue(monkeypatch):
    """
    A private QueueService per test.

    The module-level singleton is shared process-wide and carries a worker
    thread, so tests that used it would leak state into each other.
    """
    service = QueueService()
    monkeypatch.setattr(queue_module, "queue_service", service)
    monkeypatch.setattr("DRP_Main.app.modules.screening.router.queue_service", service)
    return service


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/screening")
    return TestClient(app)


@pytest.fixture()
def stub_docking(monkeypatch):
    """
    Replace the docking pipeline with a recorder.

    Real docking needs `vina` and `obabel`, which are not installed here — and
    these tests are about the queue and the HTTP contract, not about docking.
    """
    calls = []

    def fake_preparation(protein_record, drug_records, task_id):
        calls.append(
            {
                "protein": protein_record.protein_name,
                "drugs": [d.drug_name for d in drug_records],
                "task_id": task_id,
            }
        )
        return protein_record.model_copy(
            update={
                "status": ProteinProcessStatus.success,
                "top_n_affinity_records": [
                    {
                        "Mode": 1,
                        "Affinity_kcal_per_mol": -9.1,
                        "protein_ligand": f"{protein_record.protein_name}_{DRUG}",
                        "protein": protein_record.protein_name,
                        "ligand": DRUG,
                        "out_pdbqt_file": "result.pdbqt",
                    }
                ],
            }
        )

    monkeypatch.setattr(pipeline_service, "protein_preparation", fake_preparation)
    monkeypatch.setattr(queue_module, "protein_preparation", fake_preparation)
    monkeypatch.setattr(
        "DRP_Main.app.modules.screening.router.pipeline_service.protein_preparation",
        fake_preparation,
    )
    return calls


def _workbook_bytes() -> bytes:
    """
    Build the Proteins/Drugs workbook in memory.

    The original read this from `/home/vishnu/Documents/...`; the shape of the
    sheets is the actual contract, so it belongs in the test.
    """
    from openpyxl import Workbook

    workbook = Workbook()
    proteins = workbook.active
    proteins.title = "Proteins"
    proteins.append(["Protein Name", "PDB ID", "AlphaFold ID"])
    proteins.append([PROTEIN, "6ABJ", None])

    drugs = workbook.create_sheet("Drugs")
    drugs.append(["Drug Name", "PubChem ID"])
    # PubChem ids arrive from pandas as floats — `clean_id` must cope.
    drugs.append([DRUG, 2732.0])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _sample_records(tmp_path):
    """A protein and drug whose structure files exist on disk."""
    pdb = tmp_path / f"{PROTEIN}_6ABJ.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")
    sdf = tmp_path / f"{DRUG}_2732.sdf"
    sdf.write_text("mock sdf\n")
    return (
        {"protein_name": PROTEIN, "pdb_file_path": str(pdb)},
        [{"drug_name": DRUG, "sdf_file_path": str(sdf)}],
    )


# ── Excel upload ──────────────────────────────────────────────────────────────

def test_excel_upload_returns_downloaded_records(client, monkeypatch, isolated_output):
    """The workbook drives retrieval and comes back as proteins/drugs records."""
    fetched = []

    def fake_pdb(pdb_id, protein_file, protein_name):
        fetched.append(("pdb", pdb_id, protein_name))
        Path(protein_file).write_text("ATOM\n")
        return protein_file

    def fake_pubchem(cid, drug_file, drug_name):
        fetched.append(("pubchem", cid, drug_name))
        Path(drug_file).write_text("sdf\n")
        return drug_file

    monkeypatch.setattr(download_service, "download_pdb", fake_pdb)
    monkeypatch.setattr(download_service, "download_pubchem", fake_pubchem)

    response = client.post(
        "/api/v1/screening/download-protein-drugs/",
        files={
            "file": (
                "sample.xlsx",
                _workbook_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    # The lowercase keys the frontend reads.
    assert set(body) == {"proteins", "drugs"}
    assert body["proteins"][0]["protein_name"] == PROTEIN
    assert body["drugs"][0]["drug_name"] == DRUG
    # A float PubChem id from pandas must be normalised, not passed as "2732.0".
    assert ("pubchem", "2732", DRUG) in fetched
    assert ("pdb", "6ABJ", PROTEIN) in fetched


def test_excel_upload_rejects_other_file_types(client):
    response = client.post(
        "/api/v1/screening/download-protein-drugs/",
        files={"file": ("notes.txt", b"not a workbook", "text/plain")},
    )
    assert response.status_code == 400
    assert "Excel" in response.json()["detail"]


def test_excel_upload_survives_a_failed_row(client, monkeypatch, isolated_output):
    """One unfetchable structure must not fail the whole upload."""
    monkeypatch.setattr(
        download_service,
        "download_pdb",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("404")),
    )
    monkeypatch.setattr(
        download_service,
        "download_pubchem",
        lambda cid, path, name: (Path(path).write_text("sdf\n"), path)[1],
    )

    response = client.post(
        "/api/v1/screening/download-protein-drugs/",
        files={"file": ("s.xlsx", _workbook_bytes(), "application/vnd.ms-excel")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["proteins"] == []          # the failed protein is not offered
    assert len(body["drugs"]) == 1         # the good drug still is


# ── Synchronous /process-protein/ ─────────────────────────────────────────────

def test_process_protein_returns_ranked_records(client, stub_docking, isolated_output):
    protein, drugs = _sample_records(isolated_output)

    response = client.post(
        "/api/v1/screening/process-protein/",
        json={"protein_record": protein, "drug_records": drugs},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["protein_name"] == PROTEIN
    assert body["status"] == ProteinProcessStatus.success
    assert body["top_n_affinity_records"][0]["Affinity_kcal_per_mol"] == -9.1
    assert body["abort"] is False
    # The original asserted only the status code; the elapsed time is part of
    # the response contract too.
    assert body["time_taken"].endswith(("sec", "min"))
    assert stub_docking[0]["drugs"] == [DRUG]


def test_process_protein_rejects_malformed_body(client):
    response = client.post(
        "/api/v1/screening/process-protein/", json={"protein_record": {"protein_name": PROTEIN}}
    )
    assert response.status_code == 422


# ── Queue ─────────────────────────────────────────────────────────────────────

def _wait_until(predicate, timeout=5.0):
    """Poll until `predicate()` or the timeout — the worker is a real thread."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_enqueue_returns_task_id_and_runs_the_task(client, fresh_queue, stub_docking, isolated_output):
    protein, drugs = _sample_records(isolated_output)

    response = client.post(
        "/api/v1/screening/process-protein-v2/",
        json={"protein_record": protein, "drug_records": drugs},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["task_id"]

    assert _wait_until(lambda: len(stub_docking) == 1), "worker never ran the task"
    # The task_id handed to the caller is the one the pipeline receives.
    assert stub_docking[0]["task_id"] == body["task_id"]
    assert _wait_until(lambda: fresh_queue.get_queue_status()["queue_size"] == 0)


def test_queue_status_and_all_tasks_shapes(client, fresh_queue, isolated_output):
    """Both report an idle queue without a current task."""
    status = client.get("/api/v1/screening/queue-status")
    assert status.status_code == 200
    assert status.json() == {"queue_size": 0, "currently_processing": None}

    tasks = client.get("/api/v1/screening/all-tasks")
    assert tasks.status_code == 200
    body = tasks.json()
    assert body["queue_size"] == 0
    assert body["current_task"] is None
    assert body["queue"] == []


def test_paused_queue_holds_work_until_resumed(client, fresh_queue, stub_docking, isolated_output):
    protein, drugs = _sample_records(isolated_output)

    assert client.delete("/api/v1/screening/stop-processing").json() == {"status": "stopped"}

    client.post(
        "/api/v1/screening/process-protein-v2/",
        json={"protein_record": protein, "drug_records": drugs},
    )
    # Nothing may run while paused, and the task must not be lost.
    time.sleep(0.2)
    assert stub_docking == []
    assert fresh_queue.get_queue_status()["queue_size"] == 1

    assert client.post("/api/v1/screening/resume-processing").json() == {"status": "resumed"}
    assert _wait_until(lambda: len(stub_docking) == 1), "resume did not restart the worker"


def test_tasks_are_processed_in_order(fresh_queue, stub_docking, isolated_output):
    protein, drugs = _sample_records(isolated_output)
    names = ["First", "Second", "Third"]

    fresh_queue.stop_processing()          # enqueue everything before any runs
    for name in names:
        fresh_queue.add_to_queue(
            ProteinBase(protein_name=name, pdb_file_path=protein["pdb_file_path"]),
            [DrugBase(**drugs[0])],
        )
    fresh_queue.resume_processing()

    assert _wait_until(lambda: len(stub_docking) == 3), "not every task ran"
    assert [call["protein"] for call in stub_docking] == names


def test_replacement_worker_starts_even_if_the_old_thread_is_alive(
    fresh_queue, monkeypatch, isolated_output
):
    """
    Regression, stated deterministically.

    A worker that has handed over is still `is_alive()` for a short while as it
    unwinds. Gating on `is_alive()` therefore declined to start a replacement
    and the task sat in the queue forever. The gate is now `_worker_running`,
    which the outgoing worker clears under the same lock `add_to_queue` takes.

    Here the previous thread is deliberately still running while the service is
    flagged idle — exactly the state the old check got wrong.
    """
    processed = []
    monkeypatch.setattr(
        queue_module,
        "protein_preparation",
        lambda protein_record, drug_records, task_id: processed.append(protein_record.protein_name),
    )

    still_unwinding = threading.Thread(target=lambda: time.sleep(2), daemon=True)
    still_unwinding.start()
    try:
        fresh_queue._processing_thread = still_unwinding
        fresh_queue._worker_running = False          # it has already handed over

        fresh_queue.add_to_queue(ProteinBase(protein_name="B", pdb_file_path="b.pdb"), [])

        assert _wait_until(lambda: processed == ["B"]), (
            f"task stranded behind a live-but-finished worker: processed={processed}, "
            f"queued={fresh_queue.get_queue_status()['queue_size']}"
        )
    finally:
        still_unwinding.join(timeout=3)


def test_task_queued_during_worker_handover_is_not_lost(fresh_queue, monkeypatch, isolated_output):
    """The same invariant under real thread timing, as a stress check."""
    processed = []
    handover = threading.Event()

    def fake_preparation(protein_record, drug_records, task_id):
        processed.append(protein_record.protein_name)
        if len(processed) == 1:
            # Let the main thread enqueue while this worker is finishing up.
            handover.set()
            time.sleep(0.1)
        return protein_record

    monkeypatch.setattr(queue_module, "protein_preparation", fake_preparation)

    fresh_queue.add_to_queue(ProteinBase(protein_name="A", pdb_file_path="a.pdb"), [])
    assert handover.wait(5), "first task never started"
    fresh_queue.add_to_queue(ProteinBase(protein_name="B", pdb_file_path="b.pdb"), [])

    assert _wait_until(lambda: processed == ["A", "B"]), f"task lost: processed={processed}"


def test_only_one_worker_thread_runs(fresh_queue, monkeypatch, isolated_output):
    """Concurrent enqueues must not spawn parallel workers — docking is serial."""
    concurrent = []
    active = 0
    guard = threading.Lock()

    def fake_preparation(protein_record, drug_records, task_id):
        nonlocal active
        with guard:
            active += 1
            concurrent.append(active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return protein_record

    monkeypatch.setattr(queue_module, "protein_preparation", fake_preparation)

    for index in range(6):
        fresh_queue.add_to_queue(
            ProteinBase(protein_name=f"P{index}", pdb_file_path="p.pdb"), []
        )

    assert _wait_until(lambda: len(concurrent) == 6, timeout=10)
    assert max(concurrent) == 1, f"workers ran in parallel: {concurrent}"


def test_failed_task_is_persisted_not_silently_dropped(fresh_queue, monkeypatch, isolated_output):
    """
    Regression: a crashed task used to be logged and forgotten — no status file,
    no queue entry, so the protein simply looked absent.
    """
    def boom(protein_record, drug_records, task_id):
        raise RuntimeError("Open Babel not found")

    monkeypatch.setattr(queue_module, "protein_preparation", boom)

    fresh_queue.add_to_queue(ProteinBase(protein_name=PROTEIN, pdb_file_path="x.pdb"), [])

    status_file = Path(protein_output_directory(PROTEIN)) / f"{PROTEIN}.json"
    assert _wait_until(status_file.exists), "failure was never recorded"

    payload = json.loads(status_file.read_text())
    assert payload["status"].startswith("Failed:")
    assert "Open Babel not found" in payload["status"]

    # And it shows up on the status endpoint rather than being missing.
    assert any(
        entry.get("protein_name") == PROTEIN and entry.get("status", "").startswith("Failed")
        for entry in pipeline_service.get_all_protein_status()
    )


def test_worker_survives_a_failed_task(fresh_queue, monkeypatch, isolated_output):
    """One bad task must not take the worker down and stall the queue."""
    seen = []

    def sometimes_fails(protein_record, drug_records, task_id):
        seen.append(protein_record.protein_name)
        if protein_record.protein_name == "Bad":
            raise RuntimeError("nope")
        return protein_record

    monkeypatch.setattr(queue_module, "protein_preparation", sometimes_fails)

    fresh_queue.stop_processing()
    for name in ("Bad", "Good"):
        fresh_queue.add_to_queue(ProteinBase(protein_name=name, pdb_file_path="x.pdb"), [])
    fresh_queue.resume_processing()

    assert _wait_until(lambda: seen == ["Bad", "Good"]), f"queue stalled: {seen}"


# ── Status WebSocket ──────────────────────────────────────────────────────────

def test_websocket_reports_state_and_closes_when_idle(client, fresh_queue, isolated_output):
    """
    The original looped forever with no exit. The contract is: an empty queue
    gets one final state and the socket closes.
    """
    with client.websocket_connect("/api/v1/screening/ws-process-protein/") as websocket:
        websocket.send_json([{"protein_name": PROTEIN, "pdb_file_path": "x.pdb"}])
        state = websocket.receive_json()

    assert state["queue_size"] == 0
    assert isinstance(state["proteins"], list)


def test_websocket_reports_a_queued_protein(client, fresh_queue, monkeypatch, isolated_output):
    """A protein waiting in the queue is reported as Queued, not as absent."""
    release = threading.Event()

    def blocking_preparation(protein_record, drug_records, task_id):
        release.wait(5)
        return protein_record

    monkeypatch.setattr(queue_module, "protein_preparation", blocking_preparation)

    fresh_queue.add_to_queue(ProteinBase(protein_name="Running", pdb_file_path="a.pdb"), [])
    fresh_queue.add_to_queue(ProteinBase(protein_name=PROTEIN, pdb_file_path="b.pdb"), [])

    try:
        with client.websocket_connect("/api/v1/screening/ws-process-protein/") as websocket:
            websocket.send_json([{"protein_name": PROTEIN, "pdb_file_path": "b.pdb"}])
            state = websocket.receive_json()
            assert state["queue_size"] >= 1
            assert state["proteins"][0]["status"] == "Queued"
    finally:
        release.set()


def test_websocket_closes_on_malformed_payload(client, fresh_queue):
    """A bad first frame must close the socket, not raise inside the handler."""
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/screening/ws-process-protein/") as websocket:
            websocket.send_json({"not": "a list of records"})
            websocket.receive_json()


# ── Status, bundles and cleanup ───────────────────────────────────────────────

def test_list_processed_proteins_reads_from_disk(client, isolated_output):
    assert client.get("/api/v1/screening/list-processed-proteins/").json() == []

    output_dir = Path(protein_output_directory(PROTEIN))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{PROTEIN}.json").write_text(
        json.dumps({"protein_name": PROTEIN, "status": ProteinProcessStatus.success})
    )

    body = client.get("/api/v1/screening/list-processed-proteins/").json()
    assert body[0]["protein_name"] == PROTEIN


def test_unreadable_status_file_is_skipped(client, isolated_output):
    """A truncated JSON file must not break the listing for every protein."""
    for name, content in ((PROTEIN, json.dumps({"protein_name": PROTEIN})), ("Broken", "{oops")):
        directory = Path(protein_output_directory(name))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.json").write_text(content)

    body = client.get("/api/v1/screening/list-processed-proteins/").json()
    assert [entry["protein_name"] for entry in body] == [PROTEIN]


def test_bundle_download_404s_for_unknown_protein(client, isolated_output):
    response = client.get("/api/v1/screening/download-protein-bundle/NoSuchProtein")
    assert response.status_code == 404


def test_bundle_zips_protein_output(client, isolated_output):
    output_dir = Path(protein_output_directory(PROTEIN))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "affinity_results_top_n.csv").write_text("ligand,Affinity_kcal_per_mol\nD,-9.1\n")

    response = client.get(f"/api/v1/screening/download-protein-bundle/{PROTEIN}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    import zipfile

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "affinity_results_top_n.csv" in archive.namelist()


def test_clear_protein_data_empties_output(client, isolated_output):
    output_dir = Path(protein_output_directory(PROTEIN))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{PROTEIN}.json").write_text("{}")

    response = client.delete("/api/v1/screening/clear-protein-data")
    assert response.status_code == 200
    assert not output_dir.exists()
    # The directory root is recreated so the next run has somewhere to write.
    assert os.path.isdir(Path(protein_output_directory(PROTEIN)).parent)
