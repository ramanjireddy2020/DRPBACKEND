"""
ScreenSuite pipeline — retrieval → preparation → docking → ranking → profiling.

Two entry points, deliberately:

* `screen_batch` — the spec's contract. Every protein × every drug in one
  submission, each protein ranked independently, per-entity failures reported
  rather than logged, and every generated file referenced in the result.
* `protein_preparation` — the original one-protein-per-call path, kept because
  `drp/runners.py`, `queue_service` and the legacy `/screening` routes are built
  on it. It now delegates its per-protein work to the same internals.

Ranking is never pooled across proteins: each receptor has its own docking box,
so affinities from different receptors are not comparable and a pooled top-N
would silently favour whichever protein happened to have the tightest box.
"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.screening import (
    docking_service,
    download_service,
    interaction_service,
    preparation_service,
)
from DRP_Main.app.modules.screening.enums import (
    DataDirectories,
    DataFiles,
    ProteinProcessStatus,
    processed_output_directory,
    protein_keys,
    run_directory,
    run_protein_directory,
    safe_name,
)
from DRP_Main.app.modules.screening.schemas import (
    DrugBase,
    EntityFailure,
    InteractionReport,
    ProteinBase,
    ProteinScreenResult,
    ResolvedDrug,
    ResolvedProtein,
    ScreenBatchResult,
)
from DRP_Main.app.shared.file_utils import (
    protein_output_directory,
    read_csv_file,
    read_json_file,
    write_json_file,
)

logger = get_logger(__name__)

# In-process status dict — keyed by task_id (supplemental; the JSON file on disk
# remains the primary state, since a worker thread does not survive a restart).
protein_process_status: dict = {}


class ScreeningError(RuntimeError):
    """A batch was refused before any compute was spent."""


def _format_elapsed(seconds: float) -> str:
    return f"{seconds / 60:.2f} min" if seconds > 60 else f"{seconds:.2f} sec"


# ── Docking one protein against many ligands ──────────────────────────────────

def _dock_protein(
    protein_name: str,
    raw_pdb_path: str,
    ligand_files: Dict[str, str],
    output_dir: str,
    *,
    box_mode: Optional[str] = None,
    num_modes: Optional[int] = None,
    exhaustiveness: Optional[int] = None,
    status_key: Optional[str] = None,
) -> Tuple[List[dict], List[EntityFailure], Dict[str, Any]]:
    """
    Prepare one receptor and dock every ligand against it.

    Returns `(affinity_records, failures, artifacts)`. A ligand that fails
    preparation or docking is recorded as a failure and the rest of the batch
    continues — one bad SDF must not lose the whole run's compute.
    """
    def _status(message: str) -> None:
        if status_key and status_key in protein_process_status:
            protein_process_status[status_key][protein_keys.status] = message

    os.makedirs(output_dir, exist_ok=True)
    safe_protein = safe_name(protein_name)
    failures: List[EntityFailure] = []
    artifacts: Dict[str, Any] = {}

    # Step 1 — receptor preparation.
    _status(ProteinProcessStatus.predocking_optimization)
    receptor_pdb = os.path.join(output_dir, f"hydro_{safe_protein}.pdb")
    preparation = preparation_service.clean_pdb(raw_pdb_path, receptor_pdb)
    artifacts["receptor_pdb"] = receptor_pdb
    artifacts["preparation"] = preparation

    # Step 2 — receptor PDBQT. The torsion tree is stripped so Vina treats the
    # receptor as rigid.
    receptor_pdbqt = os.path.join(output_dir, f"{safe_protein}.pdbqt")
    docking_service.convert_to_pdbqt(receptor_pdb, receptor_pdbqt, "pdb")
    clean_receptor_pdbqt = os.path.join(output_dir, f"clean_{safe_protein}.pdbqt")
    docking_service.remove_lines(receptor_pdbqt, clean_receptor_pdbqt)
    artifacts["receptor_pdbqt"] = clean_receptor_pdbqt

    # Step 3 — docking box, derived from the *raw* structure: preparation strips
    # the co-crystal ligand that marks the real binding site.
    box = docking_service.compute_docking_box(raw_pdb_path, box_mode)
    artifacts["box"] = box
    logger.info("Docking box for %s: %s", protein_name, box.get("box_source"))

    # Step 4 — dock each ligand.
    affinity_records: List[dict] = []
    for drug_name, sdf_path in ligand_files.items():
        safe_drug = safe_name(drug_name)
        try:
            _status(f"Minimizing {drug_name}")
            minimized = docking_service.minimize_mol(
                sdf_path, os.path.join(output_dir, f"minimized_{safe_drug}.sdf")
            )

            _status(f"Converting {drug_name} to PDBQT")
            ligand_pdbqt = docking_service.convert_to_pdbqt(
                minimized, os.path.join(output_dir, f"minimized_{safe_drug}.pdbqt"), "sdf"
            )

            _status(f"Generating Vina config for {drug_name}")
            pose_file = os.path.join(output_dir, f"result_{safe_protein}_{safe_drug}.pdbqt")
            config_file = os.path.join(output_dir, f"config_{safe_protein}_{safe_drug}.txt")
            docking_service.generate_vina_config_file(
                protein_name,
                clean_receptor_pdbqt,
                drug_name,
                ligand_pdbqt,
                config_file,
                pose_file,
                box=box,
                num_modes=num_modes,
                exhaustiveness=exhaustiveness,
            )

            _status(f"Docking {protein_name} and {drug_name}")
            log_file = os.path.join(output_dir, f"log_{safe_protein}_{safe_drug}.log")
            records = docking_service.run_vina(config_file, receptor_pdb, vina_log_file=log_file)
            if not records:
                raise RuntimeError(f"Vina produced no affinity table; see {log_file}")
            for record in records:
                record["ligand"] = drug_name
                record["protein"] = protein_name
            affinity_records.extend(records)
            logger.info("Docked %s vs %s: %d poses", protein_name, drug_name, len(records))
        except Exception as exc:  # noqa: BLE001 — one ligand must not stop the rest
            logger.warning("Docking failed for %s vs %s: %s", protein_name, drug_name, exc)
            failures.append(
                EntityFailure(name=drug_name, kind="drug", stage="docking", error=str(exc))
            )

    return affinity_records, failures, artifacts


def _profile_top_poses(
    protein_name: str,
    receptor_pdb: str,
    top_records: List[dict],
    output_dir: str,
) -> Tuple[List[InteractionReport], List[EntityFailure]]:
    """Profile only the selected best pose of each top-ranked ligand."""
    reports: List[InteractionReport] = []
    failures: List[EntityFailure] = []
    for record in top_records:
        ligand = record.get("ligand")
        mode = int(record.get("Mode", 1))
        try:
            report = interaction_service.get_interaction_report(
                receptor_pdb,
                record["out_pdbqt_file"],
                mode=mode,
                ligand=ligand,
                protein=protein_name,
                output_dir=output_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Interaction profiling failed for %s", ligand)
            report = InteractionReport(
                status="failed", protein=protein_name, ligand=ligand, mode=mode, error=str(exc)
            )
        reports.append(report)
        if report.status != "success":
            failures.append(
                EntityFailure(
                    name=str(ligand),
                    kind="drug",
                    stage="interaction",
                    error=report.error or f"interaction profiling returned {report.status}",
                )
            )
    return reports, failures


def _collect_files(output_dir: str) -> Dict[str, List[str]]:
    """Every file a protein's run produced, grouped by kind."""
    groups: Dict[str, List[str]] = {
        "poses": [], "affinity_tables": [], "interaction_reports": [],
        "receptor": [], "configs": [], "logs": [],
    }
    for path in sorted(Path(output_dir).glob("*")):
        if path.is_dir():
            continue
        name, suffix = path.name, path.suffix.lower()
        if name.startswith("result_") and suffix == ".pdbqt":
            groups["poses"].append(str(path))
        elif suffix == ".csv":
            groups["affinity_tables"].append(str(path))
        elif name.startswith("interaction_") and suffix == ".json":
            groups["interaction_reports"].append(str(path))
        elif suffix in (".pdb", ".pdbqt"):
            groups["receptor"].append(str(path))
        elif suffix == ".txt":
            groups["configs"].append(str(path))
        elif suffix == ".log":
            groups["logs"].append(str(path))
    return {key: value for key, value in groups.items() if value}


# ── Batch entry point ─────────────────────────────────────────────────────────

@observe(name="screensuite_batch")
def screen_batch(
    proteins: List[ResolvedProtein],
    drugs: List[ResolvedDrug],
    *,
    run_id: Optional[str] = None,
    box_mode: Optional[str] = None,
    num_modes: Optional[int] = None,
    exhaustiveness: Optional[int] = None,
    skip_interactions: bool = False,
) -> ScreenBatchResult:
    """
    Screen a whole resolved batch.

    Input is trusted as already resolved and confirmed — a record without a
    usable identifier is a contract violation upstream, not something recovered
    from here. Partial results are the defined behaviour: proteins that succeed
    are returned with their rankings, and everything that failed appears in
    `failures` with the stage it failed at.
    """
    if not proteins:
        raise ScreeningError("no proteins to screen")
    if not drugs:
        raise ScreeningError("no drugs to screen")

    combinations = len(proteins) * len(drugs)
    limit = int(settings.SCREENING_MAX_COMBINATIONS or 200)
    if combinations > limit:
        raise ScreeningError(
            f"{len(proteins)} protein(s) × {len(drugs)} drug(s) = {combinations} docking runs, "
            f"over the configured limit of {limit}. Narrow the batch or raise "
            f"SCREENING_MAX_COMBINATIONS."
        )

    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    started = time.time()
    langfuse_context.update_current_observation(
        input={"run_id": run_id, "proteins": len(proteins), "drugs": len(drugs)}
    )
    logger.info("ScreenSuite run %s: %d protein(s) × %d drug(s)", run_id, len(proteins), len(drugs))

    # Stage 1 — retrieval, as a hard gate.
    protein_files, drug_files, failures = download_service.fetch_resolved_structures(proteins, drugs)

    results: List[ProteinScreenResult] = []
    for protein in proteins:
        raw_pdb = protein_files.get(protein.name)
        if not raw_pdb:
            # Already recorded as a download failure by the gate above.
            results.append(
                ProteinScreenResult(
                    protein_name=protein.name,
                    identifier=protein.identifier,
                    source=protein.source.value,
                    status="Failed: structure download",
                )
            )
            continue
        if not drug_files:
            results.append(
                ProteinScreenResult(
                    protein_name=protein.name,
                    identifier=protein.identifier,
                    source=protein.source.value,
                    status="Skipped: no ligand structures available",
                )
            )
            continue

        protein_started = time.time()
        output_dir = run_protein_directory(run_id, protein.name)
        try:
            records, dock_failures, artifacts = _dock_protein(
                protein.name,
                raw_pdb,
                drug_files,
                output_dir,
                box_mode=box_mode,
                num_modes=num_modes,
                exhaustiveness=exhaustiveness,
            )
        except Exception as exc:  # noqa: BLE001 — a dead receptor must not kill the batch
            logger.exception("Receptor preparation failed for %s", protein.name)
            failures.append(
                EntityFailure(
                    name=protein.name, kind="protein", stage="preparation", error=str(exc)
                )
            )
            results.append(
                ProteinScreenResult(
                    protein_name=protein.name,
                    identifier=protein.identifier,
                    source=protein.source.value,
                    status=f"Failed: {exc}",
                    files=_collect_files(output_dir),
                )
            )
            continue

        failures.extend(dock_failures)
        if not records:
            results.append(
                ProteinScreenResult(
                    protein_name=protein.name,
                    identifier=protein.identifier,
                    source=protein.source.value,
                    status="Failed: every ligand failed to dock",
                    receptor_pdb=artifacts.get("receptor_pdb"),
                    files=_collect_files(output_dir),
                    time_taken=_format_elapsed(time.time() - protein_started),
                )
            )
            continue

        # Stage 2 — rank this protein alone.
        all_csv = os.path.join(output_dir, DataFiles.protein_affinity_csv_file)
        top_csv = os.path.join(output_dir, DataFiles.protein_affinity_csv_file_top_n)
        pd.DataFrame(records).to_csv(all_csv, index=False)
        top_records = docking_service.get_top_affinity(records, top_csv)

        # Stage 3 — profile the selected pose of each top-ranked ligand.
        reports: List[InteractionReport] = []
        if not skip_interactions:
            reports, interaction_failures = _profile_top_poses(
                protein.name, artifacts["receptor_pdb"], top_records, output_dir
            )
            failures.extend(interaction_failures)

        results.append(
            ProteinScreenResult(
                protein_name=protein.name,
                identifier=protein.identifier,
                source=protein.source.value,
                status=ProteinProcessStatus.success,
                receptor_pdb=artifacts.get("receptor_pdb"),
                affinity_records=records,
                top_affinity_records=top_records,
                interaction_reports=reports,
                files={
                    **_collect_files(output_dir),
                    "box": artifacts.get("box"),
                    "preparation": artifacts.get("preparation"),
                    "interaction_summary": interaction_service.summarize_interactions(reports),
                },
                time_taken=_format_elapsed(time.time() - protein_started),
            )
        )

    succeeded = [r for r in results if r.status == ProteinProcessStatus.success]
    if succeeded and failures:
        status = "partial"
    elif succeeded:
        status = "success"
    else:
        status = "failed"

    result = ScreenBatchResult(
        run_id=run_id,
        status=status,
        proteins=results,
        failures=failures,
        combinations=combinations,
        files={"run_directory": run_directory(run_id)},
        time_taken=_format_elapsed(time.time() - started),
        recommendation=_batch_recommendation(succeeded, failures),
    )

    manifest_path = os.path.join(run_directory(run_id), DataFiles.run_manifest)
    write_json_file(manifest_path, json.loads(result.model_dump_json()))
    result.files["manifest"] = manifest_path

    langfuse_context.update_current_observation(
        output={"status": status, "proteins_succeeded": len(succeeded), "failures": len(failures)}
    )
    return result


def _batch_recommendation(
    succeeded: List[ProteinScreenResult], failures: List[EntityFailure]
) -> str:
    """One plain-language line naming the strongest hit and what fell out."""
    if not succeeded:
        return (
            "No protein completed screening. "
            + (f"{len(failures)} entity failure(s) — see `failures`." if failures else "")
        ).strip()

    lines = []
    for result in succeeded:
        best = min(
            result.top_affinity_records,
            key=lambda record: record.get("Affinity_kcal_per_mol", 0.0),
            default=None,
        )
        if best:
            lines.append(
                f"{result.protein_name}: best hit {best.get('ligand')} at "
                f"{best.get('Affinity_kcal_per_mol')} kcal/mol"
            )
    message = "; ".join(lines)
    if failures:
        message += f". {len(failures)} entity failure(s) reported separately."
    return message


def get_run_manifest(run_id: str) -> dict:
    """Read back a finished run's manifest."""
    path = os.path.join(run_directory(run_id), DataFiles.run_manifest)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no manifest for run {run_id}")
    return read_json_file(path)


def list_runs() -> List[dict]:
    """Every run with a manifest on disk, newest first."""
    from DRP_Main.app.modules.screening.enums import Directorynames, output_root

    runs_root = Path(output_root()) / Directorynames.runs_directory
    if not runs_root.exists():
        return []
    entries = []
    for directory in runs_root.iterdir():
        manifest = directory / DataFiles.run_manifest
        if not manifest.is_file():
            continue
        try:
            payload = read_json_file(str(manifest))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            {
                "run_id": payload.get("run_id", directory.name),
                "status": payload.get("status"),
                "combinations": payload.get("combinations"),
                "proteins": [p.get("protein_name") for p in payload.get("proteins", [])],
                "modified": manifest.stat().st_mtime,
            }
        )
    return sorted(entries, key=lambda entry: entry["modified"], reverse=True)


def archive_run(run_id: str) -> str:
    """ZIP a whole run directory for download."""
    directory = run_directory(run_id)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"no such run: {run_id}")
    archive_base = os.path.join(DataDirectories.processed_proteins_archieves, safe_name(run_id))
    os.makedirs(DataDirectories.processed_proteins_archieves, exist_ok=True)
    shutil.make_archive(archive_base, "zip", directory)
    return f"{archive_base}.zip"


# ── Legacy per-protein entry point ────────────────────────────────────────────

def is_protein_processed(protein_record: ProteinBase):
    """
    Whether a protein has already been fully processed.

    Returns `(bool, ProteinBase | None)`. Keyed on protein name, which is why
    batch runs write under a run id instead — two different structures sharing a
    name would otherwise hit each other's cache.
    """
    protein_name = protein_record.protein_name
    protein_op_dir = protein_output_directory(protein_name)
    protein_status_file_json = os.path.join(protein_op_dir, f"{protein_name}.json")

    if not os.path.exists(protein_op_dir) or not os.path.exists(protein_status_file_json):
        return False, None

    protein_status = read_json_file(protein_status_file_json)
    if protein_status.get(protein_keys.status) == ProteinProcessStatus.success:
        top_n_csv_path = os.path.join(protein_op_dir, DataFiles.protein_affinity_csv_file_top_n)
        if os.path.exists(top_n_csv_path):
            top_n_percent_affinity = read_csv_file(top_n_csv_path)
            protein_record = protein_record.model_copy(
                update={
                    protein_keys.top_n_affinity_rcords: top_n_percent_affinity,
                    protein_keys.status: ProteinProcessStatus.success,
                }
            )
            logger.info("Protein already processed: %s", protein_name)
            return True, protein_record

    return False, None


@observe(name="protein_preparation")
def protein_preparation(
    protein_record: ProteinBase, all_drug_records: list[DrugBase], protein_tuid: str
) -> ProteinBase:
    """
    Screen one protein against its drug records, from local file paths.

    The original contract: inputs are already-downloaded files, not identifiers.
    `screen_batch` is the path for resolved identifiers.
    """
    langfuse_context.update_current_observation(
        input={
            "protein_name": protein_record.protein_name,
            "drug_count": len(all_drug_records),
        }
    )

    already_done, cached = is_protein_processed(protein_record)
    if already_done and cached is not None:
        langfuse_context.update_current_observation(output={"cached": True})
        return cached

    protein_name = protein_record.protein_name
    protein_op_dir = protein_output_directory(protein_name)
    os.makedirs(protein_op_dir, exist_ok=True)
    protein_status_file_json = os.path.join(protein_op_dir, f"{protein_name}.json")

    protein_process_status[protein_tuid] = {
        "item": protein_record,
        protein_keys.status: ProteinProcessStatus.predocking_optimization,
    }

    ligand_files = {drug.drug_name: drug.sdf_file_path for drug in all_drug_records}
    try:
        records, failures, artifacts = _dock_protein(
            protein_name,
            protein_record.pdb_file_path,
            ligand_files,
            protein_op_dir,
            status_key=protein_tuid,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preparation failed for %s", protein_name)
        protein_process_status[protein_tuid][protein_keys.status] = f"Failed: {exc}"
        failed = protein_record.model_copy(update={protein_keys.status: f"Failed: {exc}"})
        write_json_file(protein_status_file_json, failed.model_dump())
        return failed

    if not records:
        detail = failures[0].error if failures else "no affinity records produced"
        status = f"Failed: {detail}"
        protein_process_status[protein_tuid][protein_keys.status] = status
        failed = protein_record.model_copy(update={protein_keys.status: status})
        write_json_file(protein_status_file_json, failed.model_dump())
        return failed

    csv_file_path = os.path.join(protein_op_dir, DataFiles.protein_affinity_csv_file)
    top_n_csv_path = os.path.join(protein_op_dir, DataFiles.protein_affinity_csv_file_top_n)
    pd.DataFrame(records).to_csv(csv_file_path, index=False)
    top_n_percent_affinity = docking_service.get_top_affinity(records, top_n_csv_path)

    # Analyse only the selected best pose of each top-ranked ligand.
    _profile_top_poses(
        protein_name, artifacts["receptor_pdb"], top_n_percent_affinity, protein_op_dir
    )

    protein_record = protein_record.model_copy(
        update={
            protein_keys.top_n_affinity_rcords: top_n_percent_affinity,
            protein_keys.status: ProteinProcessStatus.success,
        }
    )
    protein_process_status[protein_tuid][protein_keys.status] = ProteinProcessStatus.success
    write_json_file(protein_status_file_json, protein_record.model_dump())
    logger.info("Protein processing complete: %s", protein_name)
    langfuse_context.update_current_observation(
        output={
            "status": ProteinProcessStatus.success,
            "top_n": len(top_n_percent_affinity),
            "failures": len(failures),
        }
    )
    return protein_record


def get_protein_status(protein_name: str) -> dict:
    protein_op_dir = protein_output_directory(protein_name)
    protein_json = os.path.join(protein_op_dir, f"{protein_name}.json")
    return read_json_file(protein_json)


def get_all_protein_status() -> list:
    proteins_dir = processed_output_directory()
    if not os.path.isdir(proteins_dir):
        return []
    protein_status_list = []
    for item in Path(proteins_dir).iterdir():
        if item.is_dir():
            try:
                protein_status_list.append(get_protein_status(item.name))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
    return protein_status_list


def get_protein_status_by_list(protein_list: list[str]) -> list:
    protein_status_list = []
    for protein_name in protein_list:
        try:
            protein_status_list.append(get_protein_status(protein_name))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return protein_status_list


def clear_data_protein() -> str:
    """Clear processed protein output and the downloaded structure caches."""
    logger.info("Clearing processed protein data")
    protein_dir, drug_dir = download_service.structure_dirs()
    for dir_path in (processed_output_directory(), protein_dir, drug_dir):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            logger.info("Cleared: %s", dir_path)
    return "Protein Data is cleared"


def archive_protein(
    protein_name: str,
    base_dir: str = None,
    output_dir: str = None,
) -> str:
    """ZIP one protein's output directory."""
    base_dir = base_dir or processed_output_directory()
    output_dir = output_dir or DataDirectories.processed_proteins_archieves
    protein_path = os.path.join(base_dir, protein_name)
    if not os.path.isdir(protein_path):
        raise FileNotFoundError(f"Error: {protein_name} directory not found in {base_dir}.")
    os.makedirs(output_dir, exist_ok=True)
    archive_name = os.path.join(output_dir, safe_name(protein_name))
    shutil.make_archive(archive_name, "zip", protein_path)
    logger.info("Archived %s to %s.zip", protein_name, archive_name)
    return f"{archive_name}.zip"


def get_top_5_percent(protein_name: str, base_dir: str = None) -> list:
    base_dir = base_dir or processed_output_directory()
    protein_path = os.path.join(base_dir, protein_name)
    if not os.path.isdir(protein_path):
        raise FileNotFoundError(f"Error: {protein_name} directory not found in {base_dir}.")
    return read_csv_file(os.path.join(protein_path, DataFiles.protein_affinity_csv_file_top_n))
