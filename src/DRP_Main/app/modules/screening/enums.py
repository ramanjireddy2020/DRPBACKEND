"""
Directory/file constants and status enums for the ScreenSuite (docking) pipeline.

Output location is resolved through `output_root()` rather than hardcoded, so a
Databricks deployment can point `SCREENING_OUTPUT_ROOT` at a Unity Catalog
volume — local cluster storage is ephemeral and anything written there does not
survive the job session.
"""
import os
import re
from pathlib import Path

from DRP_Main.app.core.config import settings


class Directorynames:
    data_dir_name = "data"
    drug_structures_directory = "drug_structures"
    protein_structures_directory = "protein_structures"
    processed_protein_structures_directory = "processed_protein_structures"
    processed_protein_op_directory = "processed_proteins_output"
    runs_directory = "screening_runs"
    minimized_molecule_dir = "minimized_molecule"
    vina_configs_dir = "vina_configs"
    protein_pdbqt = "protein_pdbqt"
    ligand_pdbqt_dir = "ligand_pdbqt"
    tmp_dir = "tmp"
    vina_output_dir = "vina_output"
    vina_logs_dir = "vina_logs"
    vina_csv_dir = "vina_csv"
    vina_split_dir = "vina_split"
    literature_search_dir = "literature_search"
    protein_archieves_dir = "protein_archieves"


class DataDirectories:
    app_dir = Path(__file__).resolve().parent.parent.parent  # app/
    data_dir = app_dir / "data"
    sample_ip_files = app_dir / "sample_ip_files"

    minimized_molecule_dir = str(data_dir / Directorynames.minimized_molecule_dir)
    vina_configs_dir = str(data_dir / Directorynames.vina_configs_dir)
    protein_pdbqt = str(data_dir / Directorynames.protein_pdbqt)
    ligand_pdbqt_dir = str(data_dir / Directorynames.ligand_pdbqt_dir)
    tmp_dir = str(data_dir / Directorynames.tmp_dir)
    vina_output_dir = str(data_dir / Directorynames.vina_output_dir)
    vina_logs_dir = str(data_dir / Directorynames.vina_logs_dir)
    vina_csv_dir = str(data_dir / Directorynames.vina_csv_dir)
    vina_split_dir = str(data_dir / Directorynames.vina_split_dir)
    literature_search_dir = str(data_dir / Directorynames.literature_search_dir)

    protein_structures = str(data_dir / Directorynames.protein_structures_directory)
    processed_protein_structures = str(data_dir / Directorynames.processed_protein_structures_directory)
    drug_structures = str(data_dir / Directorynames.drug_structures_directory)
    processed_proteins_output = str(data_dir / Directorynames.processed_protein_op_directory)
    processed_proteins_archieves = str(data_dir / Directorynames.protein_archieves_dir)


class DataFiles:
    protein_affinity_csv_file = "affinity_results.csv"
    protein_affinity_csv_file_top_n = "affinity_results_top_n.csv"
    run_manifest = "run_manifest.json"


class SampleInputDirectories:
    app_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = app_dir / "data"
    sample_ip_files = app_dir / "sample_ip_files"


class InputFiles:
    protein_drug_download_ip = str(
        SampleInputDirectories.sample_ip_files / "literature_protein_and_drug_data.xlsx"
    )
    relevance_score_max_counts = str(
        SampleInputDirectories.sample_ip_files / "relevance_score_max_counts.json"
    )
    literature_search_protein_history = str(
        SampleInputDirectories.sample_ip_files / "literature_search_protein_history.json"
    )
    greek_symbols = str(
        SampleInputDirectories.sample_ip_files / "greek_symbols.json"
    )


class ProteinProcessStatus:
    downloading = "Downloading"
    predocking_optimization = "Pre Docking Optimization"
    docking = "Docking"
    getting_lead_molecule = "Getting Lead Molecule"
    interaction_data_analysis = "Interaction data analysis"
    failed_to_convert_to_pdbqt = "Failed To Convert to pdbqt"
    failed = "Failed"
    success = "Success"


class ResolutionStage:
    """Mirrors CurateX's `awaiting_target_confirmation` checkpoint contract."""
    awaiting_confirmation = "awaiting_structure_confirmation"
    resolved = "resolved"


class protein_keys:
    pdb_file_path = "pdb_file_path"
    sdf_file_path = "sdf_file_path"
    protein_name = "protein_name"
    drug_name = "drug_name"
    proteins = "proteins"
    drugs = "drugs"
    status = "status"
    top_n_affinity_rcords = "top_n_affinity_records"
    time_taken = "time_taken"


# ── Path helpers ──────────────────────────────────────────────────────────────

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """
    Make `name` safe to use as a path segment.

    Output paths are name-derived, so a name carrying a separator or a stray
    character would otherwise escape its own run directory.
    """
    cleaned = _SAFE_NAME.sub("_", str(name).strip())
    return cleaned.strip("._") or "unnamed"


def output_root() -> Path:
    """The root every run writes under. `SCREENING_OUTPUT_ROOT` overrides."""
    configured = (settings.SCREENING_OUTPUT_ROOT or "").strip()
    return Path(configured) if configured else DataDirectories.data_dir


def processed_output_directory() -> str:
    """
    Root holding per-protein output from the legacy path.

    Resolved through `output_root()` rather than read off `DataDirectories`, so
    it follows `SCREENING_OUTPUT_ROOT` like everything else — the static class
    attribute is fixed at import and cannot be redirected.
    """
    path = output_root() / Directorynames.processed_protein_op_directory
    os.makedirs(path, exist_ok=True)
    return str(path)


def run_directory(run_id: str) -> str:
    """
    Per-run output directory.

    Runs are scoped by id rather than by protein name alone: two records sharing
    a name would otherwise overwrite each other's files *and* collide on the
    `<protein_name>.json` resume key, making one run return the other's results
    as a cache hit.
    """
    path = output_root() / Directorynames.runs_directory / safe_name(run_id)
    os.makedirs(path, exist_ok=True)
    return str(path)


def run_protein_directory(run_id: str, protein_name: str) -> str:
    """Per-protein output directory inside a run."""
    path = Path(run_directory(run_id)) / safe_name(protein_name)
    os.makedirs(path, exist_ok=True)
    return str(path)
