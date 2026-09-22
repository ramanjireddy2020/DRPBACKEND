"""
ScreenSuite router.

Two surfaces, both mounted under `/screening`:

* **Spec surface** — `/resolve`, `/confirm-resolution`, `/screen`, `/runs/*`.
  Names and identifiers in, one resolved list, one batch submission, one
  combined result with per-entity failures and file references.
* **Legacy surface** — `/download-protein-drugs/`, `/process-protein/`, the
  queue routes and the WebSocket. Kept because `drp/runners.py` and the existing
  frontend call them.

`/screen` runs the batch inline and can take minutes. The asynchronous path is
`/v1/agents/screensuite/screen`, which returns `202 {jobId}` and polls — this
route exists for direct/scripted use and is bounded by
`SCREENING_MAX_COMBINATIONS`.
"""
import asyncio
import io
import time
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.screening import (
    docking_service,
    download_service,
    interaction_service,
    pipeline_service,
    resolution_service,
)
from DRP_Main.app.modules.screening.enums import ResolutionStage
from DRP_Main.app.modules.screening.queue_service import queue_service
from DRP_Main.app.modules.screening.schemas import (
    ConfirmResolutionRequest,
    DownloadProteinDrugResponse,
    DrugBase,
    InteractionRequest,
    ProcessProteinResponse,
    ProteinBase,
    ResolutionRequest,
    ResolutionResult,
    ScreenBatchRequest,
    ScreenBatchResult,
)
from DRP_Main.app.shared.file_utils import read_excel_file

logger = get_logger(__name__)

router = APIRouter()


# ── Environment ───────────────────────────────────────────────────────────────

@router.get("/health")
def screening_health():
    """
    Which parts of the docking stack this deployment can actually run.

    Vina and Open Babel are external executables and neither is pip-installable,
    so a deployment can import this module cleanly and still be unable to dock.
    """
    tools = docking_service.tool_availability()
    conversion = docking_service.conversion_available()
    docking = docking_service.docking_available()
    blocked = [
        name for name, state in tools.items() if state.get("reason") == "blocked_by_policy"
    ]

    if docking and conversion:
        blocker = None
    elif not docking:
        # Either route satisfies Vina; report the binary's reason, since the
        # bindings' only failure mode is "not installed".
        blocker = tools["vina"].get("error") or "AutoDock Vina is not available"
    else:
        blocker = "no PDBQT conversion route available"

    return {
        "module": "ScreenSuite",
        "dockable": bool(docking and conversion),
        "docking_route": (
            "executable" if tools["vina"].get("available")
            else "python-bindings" if tools["vina_in_process"].get("available")
            else None
        ),
        "structure_preparation": conversion,
        "interaction_profiling": interaction_service.prolif_available(),
        "blocked_by_policy": blocked,
        "blocker": blocker,
        "tools": tools,
    }


# ── Resolution step ───────────────────────────────────────────────────────────

@router.post("/resolve", response_model=ResolutionResult)
def resolve_entities(request: ResolutionRequest):
    """
    Resolve protein/drug names to identifiers, each tagged with its source.

    An entity that already carries an identifier is passed through unlooked-up.
    A name matching several structures comes back under
    `awaiting_structure_confirmation` with a shortlist — never auto-selected,
    because docking the wrong structure costs real compute.
    """
    if not request.proteins and not request.drugs:
        raise HTTPException(status_code=422, detail="Provide at least one protein or drug.")
    return resolution_service.resolve(request.proteins, request.drugs, allow_partial=True)


@router.post("/confirm-resolution", response_model=ResolutionResult)
def confirm_resolution(request: ConfirmResolutionRequest):
    """
    Apply shortlist choices from a confirmation checkpoint and re-resolve.

    Once confirmed, a choice is treated exactly like any other resolved entity.
    """
    return resolution_service.confirm(
        [selection.model_dump() for selection in request.selections],
        request.proteins,
        request.drugs,
    )


# ── Batch screening ───────────────────────────────────────────────────────────

@router.post("/screen", response_model=ScreenBatchResult)
def screen(request: ScreenBatchRequest):
    """
    Screen every protein against every drug in one submission.

    Each protein is ranked independently. Anything that fails — a download, a
    receptor, a ligand, an interaction profile — is reported in `failures` with
    its stage, alongside whatever succeeded.
    """
    try:
        return pipeline_service.screen_batch(
            request.proteins,
            request.drugs,
            run_id=request.run_id,
            box_mode=request.box_mode,
            num_modes=request.num_modes,
            exhaustiveness=request.exhaustiveness,
            skip_interactions=request.skip_interactions,
        )
    except pipeline_service.ScreeningError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/resolve-and-screen", response_model=dict)
def resolve_and_screen(request: ResolutionRequest):
    """
    The agent's own path: resolve, then screen only if nothing is ambiguous.

    An ambiguous batch returns the resolution result untouched — the
    confirmation checkpoint comes before any compute is spent.
    """
    resolution = resolution_service.resolve(request.proteins, request.drugs, allow_partial=True)
    if resolution.stage == ResolutionStage.awaiting_confirmation:
        return {"stage": resolution.stage, "resolution": resolution.model_dump()}
    if not resolution.proteins or not resolution.drugs:
        return {
            "stage": "nothing_to_screen",
            "resolution": resolution.model_dump(),
        }
    try:
        result = pipeline_service.screen_batch(resolution.proteins, resolution.drugs)
    except pipeline_service.ScreeningError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "stage": "complete",
        "resolution": resolution.model_dump(),
        "result": result.model_dump(),
    }


@router.get("/runs")
def list_runs():
    """Every screening run with a manifest on disk, newest first."""
    return pipeline_service.list_runs()


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """Read back a finished run's full result."""
    try:
        return pipeline_service.get_run_manifest(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/runs/{run_id}/download")
def download_run(run_id: str):
    """Download a whole run directory as a ZIP bundle."""
    try:
        archive = pipeline_service.archive_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(archive, media_type="application/zip", filename=f"{run_id}.zip")


# ── Interaction profiling ─────────────────────────────────────────────────────

@router.post("/generate-ligplot-report/")
def generate_ligplot_report(request: InteractionRequest):
    """Profile the interactions of one docked pose."""
    report = interaction_service.get_interaction_report(
        request.receptor_pdb,
        request.vina_output_pdbqt,
        mode=request.mode,
        ligand=request.ligand,
    )
    if report.status == "unavailable":
        raise HTTPException(status_code=503, detail=report.error)
    return report


# ── Legacy surface ────────────────────────────────────────────────────────────

@router.post("/download-protein-drugs/", response_model=DownloadProteinDrugResponse)
async def download_protein_drugs_to_server(file: UploadFile = File(...)):
    """Download structures listed in an uploaded Excel workbook."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed.")
    contents = await file.read()
    excel_file = pd.ExcelFile(io.BytesIO(contents))
    protein_records = read_excel_file(excel_file, "Proteins")
    drug_records = read_excel_file(excel_file, "Drugs")
    return download_service.download_protein_drug_structures(protein_records, drug_records)


@router.post("/process-protein/", response_model=ProcessProteinResponse)
def process_protein(protein_record: ProteinBase, drug_records: List[DrugBase]):
    """Screen one protein against its drug records, synchronously."""
    start_time = time.time()
    protein_data = pipeline_service.protein_preparation(protein_record, drug_records, "0")
    elapsed = time.time() - start_time
    time_taken = f"{elapsed / 60:.2f} min" if elapsed > 60 else f"{elapsed:.2f} sec"
    protein_data = protein_data.model_copy(update={"time_taken": time_taken})
    return ProcessProteinResponse(
        protein_name=protein_data.protein_name,
        pdb_file_path=protein_data.pdb_file_path,
        top_n_affinity_records=protein_data.top_n_affinity_records or [],
        status=protein_data.status or "",
        time_taken=protein_data.time_taken or "",
        abort=False,
    )


@router.post("/process-protein-v2/")
async def process_protein_queued(protein_record: ProteinBase, drug_records: List[DrugBase]):
    """Queue a one-protein screening task."""
    return queue_service.add_to_queue(protein_record, drug_records)


@router.get("/queue-status")
async def check_queue_status():
    return queue_service.get_queue_status()


@router.delete("/stop-processing")
async def stop_processing():
    return queue_service.stop_processing()


@router.post("/resume-processing")
async def resume_processing():
    return queue_service.resume_processing()


@router.get("/all-tasks")
async def get_all_tasks():
    return queue_service.get_all_tasks()


@router.get("/list-processed-proteins/")
def list_processed_proteins():
    return pipeline_service.get_all_protein_status()


@router.websocket("/ws-process-protein/")
async def websocket_process_protein(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        protein_records = [ProteinBase(**record) for record in data]
    except Exception as exc:  # noqa: BLE001
        logger.info("WebSocket receive error: %s", exc)
        await websocket.close()
        return

    prev_queue_size = -1
    while True:
        state = queue_service.get_current_processing_state(protein_records)
        current_size = state["queue_size"]
        if current_size == 0:
            await websocket.send_json(state)
            await websocket.close()
            break
        if prev_queue_size != current_size:
            await websocket.send_json(state)
            prev_queue_size = current_size
        await asyncio.sleep(5)


@router.delete("/clear-protein-data")
async def clear_protein_data():
    return pipeline_service.clear_data_protein()


@router.post("/create-protein-bundle/{protein_name}")
async def create_protein_bundle(protein_name: str):
    return pipeline_service.archive_protein(protein_name)


@router.get("/download-protein-bundle/{protein_name}")
async def download_protein_bundle(protein_name: str):
    """Create and download a ZIP archive for one protein's output."""
    try:
        zip_file = pipeline_service.archive_protein(protein_name)
        return FileResponse(zip_file, media_type="application/zip", filename=f"{protein_name}.zip")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error creating archive: {exc}")
