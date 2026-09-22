"""
Structure retrieval — fetch protein and ligand structures from public databases.

Fetching is a **hard gate**: a download that fails excludes that entity from the
run and is reported back as an `EntityFailure`, rather than being logged and then
treated as available by the next stage. A zero-byte or non-structural response
counts as a failure too — RCSB and PubChem both answer 200 with an HTML error
page for some bad identifiers.
"""
import os
import shutil
from typing import Dict, List, Tuple

import requests
from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.http_client import get_json
from DRP_Main.app.modules.screening.enums import DataDirectories, output_root, protein_keys, safe_name
from DRP_Main.app.modules.screening.schemas import (
    DrugSource,
    EntityFailure,
    ProteinSource,
    ResolvedDrug,
    ResolvedProtein,
)

logger = get_logger(__name__)

#: Smallest plausible structure file. Anything under this is an error page.
_MIN_STRUCTURE_BYTES = 200


def _timeout() -> int:
    return int(settings.SCREENING_RESOLVE_TIMEOUT or 20)


def structure_dirs() -> Tuple[str, str]:
    """(protein_dir, drug_dir) under the configured output root."""
    root = output_root()
    protein_dir = str(root / "protein_structures")
    drug_dir = str(root / "drug_structures")
    os.makedirs(protein_dir, exist_ok=True)
    os.makedirs(drug_dir, exist_ok=True)
    return protein_dir, drug_dir


def create_protein_output_directory(protein_name: str) -> str:
    protein_output_dir = os.path.join(
        DataDirectories.processed_proteins_output, safe_name(protein_name)
    )
    os.makedirs(protein_output_dir, exist_ok=True)
    return protein_output_dir


def _fetch_to_file(url: str, destination: str, *, expect: str) -> None:
    """
    GET `url` into `destination`, or raise with a readable reason.

    Writes through a `.part` file so a failed download never leaves a truncated
    structure behind at the real name for the next stage to pick up.
    """
    response = requests.get(url, timeout=_timeout())
    if response.status_code != 200:
        raise RuntimeError(f"{url} returned HTTP {response.status_code}")

    content = response.content
    if len(content) < _MIN_STRUCTURE_BYTES:
        raise RuntimeError(f"{url} returned {len(content)} bytes — not a structure file")
    head = content[:2048].lstrip()[:200].decode("utf-8", errors="replace").lower()
    if head.startswith(("<!doctype", "<html")):
        raise RuntimeError(f"{url} returned an HTML page, not {expect}")

    temporary = destination + ".part"
    with open(temporary, "wb") as handle:
        handle.write(content)
    os.replace(temporary, destination)


# ── Single-structure fetchers ─────────────────────────────────────────────────

def download_pdb(pdb_id: str, protein_file: str, protein_name: str) -> str:
    _fetch_to_file(
        f"https://files.rcsb.org/download/{pdb_id}.pdb", protein_file, expect="a PDB file"
    )
    logger.info("Downloaded PDB %s for %s -> %s", pdb_id, protein_name, protein_file)
    return protein_file


def download_alphafold(af_id: str, protein_file: str, protein_name: str) -> str:
    """
    Fetch an AlphaFold model.

    The URL is taken from AlphaFold's own prediction API rather than assembled
    from a guessed model version: the version in the filename moves (v4 → v5 →
    v6) and withdrawn versions 404, so a hardcoded version silently stops
    working for every accession. Guessed versions remain as a fallback for when
    the API itself is unreachable.
    """
    errors = []
    payload = get_json(f"https://alphafold.ebi.ac.uk/api/prediction/{af_id}", timeout=_timeout())
    if isinstance(payload, list) and payload:
        url = payload[0].get("pdbUrl")
        if url:
            try:
                _fetch_to_file(url, protein_file, expect="a PDB file")
                logger.info("Downloaded AlphaFold %s for %s", af_id, protein_name)
                return protein_file
            except RuntimeError as exc:
                errors.append(str(exc))

    for version in ("v6", "v5", "v4"):
        url = f"https://alphafold.ebi.ac.uk/files/{af_id}-model_{version}.pdb"
        try:
            _fetch_to_file(url, protein_file, expect="a PDB file")
            logger.info("Downloaded AlphaFold %s (%s) for %s", af_id, version, protein_name)
            return protein_file
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(f"no AlphaFold model available for {af_id}: {errors[0]}")


def download_pubchem(pubchem_id: str, drug_file: str, drug_name: str) -> str:
    """
    Fetch a PubChem 3D conformer SDF.

    Not every compound has a computed 3D conformer; those fall back to the 2D
    record, which RDKit then embeds during minimisation.
    """
    base = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{pubchem_id}/record/SDF"
    last_error = "no attempt made"
    for record_type in ("3d", "2d"):
        url = f"{base}?record_type={record_type}&response_type=save"
        try:
            _fetch_to_file(url, drug_file, expect="an SDF file")
            logger.info("Downloaded PubChem CID %s (%s) for %s", pubchem_id, record_type, drug_name)
            return drug_file
        except RuntimeError as exc:
            last_error = str(exc)
    raise RuntimeError(f"no PubChem record for CID {pubchem_id}: {last_error}")


def download_zinc(zinc_id: str, drug_file: str, drug_name: str) -> str:
    """Fetch a ZINC substance SDF. Only used when a ZINC id was supplied directly."""
    _fetch_to_file(
        f"https://zinc.docking.org/substances/{zinc_id}.sdf", drug_file, expect="an SDF file"
    )
    logger.info("Downloaded ZINC %s for %s", zinc_id, drug_name)
    return drug_file


def clean_id(id_str: str) -> str:
    """Normalise an identifier that arrived via pandas (which floats ints)."""
    id_str = str(id_str).strip()
    if id_str.endswith(".0"):
        id_str = id_str[:-2]
    return id_str


def get_structure_name(id: str, protein_name: str = None, drug_name: str = None) -> str:
    if protein_name and drug_name:
        raise ValueError("Provide either protein_name or drug_name, not both")
    if protein_name:
        return f"{safe_name(protein_name)}_{clean_id(id)}.pdb"
    if drug_name:
        return f"{safe_name(drug_name)}_{clean_id(id)}.sdf"
    raise ValueError("Either protein_name or drug_name must be provided")


# ── Resolved-list retrieval (the pipeline's entry point) ──────────────────────

@observe(name="structure_retrieval")
def fetch_resolved_structures(
    proteins: List[ResolvedProtein], drugs: List[ResolvedDrug]
) -> Tuple[Dict[str, str], Dict[str, str], List[EntityFailure]]:
    """
    Fetch every resolved structure.

    Returns `(protein_files, drug_files, failures)` keyed by entity name. A name
    absent from its dict failed and appears in `failures` — the caller must not
    treat it as available.
    """
    langfuse_context.update_current_observation(
        input={"protein_count": len(proteins), "drug_count": len(drugs)}
    )
    protein_dir, drug_dir = structure_dirs()
    protein_files: Dict[str, str] = {}
    drug_files: Dict[str, str] = {}
    failures: List[EntityFailure] = []

    for protein in proteins:
        file_name = get_structure_name(protein.identifier, protein_name=protein.name)
        file_path = os.path.join(protein_dir, file_name)
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) >= _MIN_STRUCTURE_BYTES:
                logger.info("Reusing cached structure for %s: %s", protein.name, file_path)
            elif protein.source == ProteinSource.pdb:
                download_pdb(protein.identifier, file_path, protein.name)
            else:
                download_alphafold(protein.identifier, file_path, protein.name)
            protein_files[protein.name] = file_path
        except Exception as exc:  # noqa: BLE001 — one bad structure must not stop the batch
            logger.warning("Structure download failed for %s: %s", protein.name, exc)
            failures.append(
                EntityFailure(name=protein.name, kind="protein", stage="download", error=str(exc))
            )

    for drug in drugs:
        file_name = get_structure_name(drug.identifier, drug_name=drug.name)
        file_path = os.path.join(drug_dir, file_name)
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) >= _MIN_STRUCTURE_BYTES:
                logger.info("Reusing cached structure for %s: %s", drug.name, file_path)
            elif drug.source == DrugSource.zinc:
                download_zinc(drug.identifier, file_path, drug.name)
            else:
                download_pubchem(drug.identifier, file_path, drug.name)
            drug_files[drug.name] = file_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Structure download failed for %s: %s", drug.name, exc)
            failures.append(
                EntityFailure(name=drug.name, kind="drug", stage="download", error=str(exc))
            )

    langfuse_context.update_current_observation(
        output={
            "proteins_downloaded": len(protein_files),
            "drugs_downloaded": len(drug_files),
            "failures": len(failures),
        }
    )
    return protein_files, drug_files, failures


# ── Spreadsheet retrieval (legacy /download-protein-drugs/ route) ─────────────

@observe(name="structure_download")
def download_protein_drug_structures(protein_records: list, drug_records: list) -> dict:
    """
    Fetch structures from spreadsheet-shaped records.

    Kept for the Excel upload route. New callers should use
    `fetch_resolved_structures`, which reports failures instead of logging them.
    """
    langfuse_context.update_current_observation(
        input={"protein_count": len(protein_records), "drug_count": len(drug_records)}
    )
    protein_dir, drug_dir = structure_dirs()
    successful_downloads = {protein_keys.proteins: [], protein_keys.drugs: []}

    for row in protein_records:
        protein_name = row.get("Protein Name") or row.get("protein_name")
        if not protein_name:
            continue
        for column, downloader in (("PDB ID", download_pdb), ("AlphaFold ID", download_alphafold)):
            raw_id = row.get(column)
            if not raw_id:
                continue
            structure_id = clean_id(raw_id)
            file_path = os.path.join(
                protein_dir, get_structure_name(structure_id, protein_name=protein_name)
            )
            try:
                if not os.path.exists(file_path):
                    downloader(structure_id, file_path, protein_name)
                successful_downloads[protein_keys.proteins].append(
                    {
                        protein_keys.protein_name: protein_name,
                        protein_keys.pdb_file_path: file_path,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Download failed for %s (%s): %s", protein_name, structure_id, exc)

    for row in drug_records:
        drug_name = row.get("Drug Name") or row.get("drug_name")
        if not drug_name:
            continue
        pubchem_id = clean_id(row.get("PubChem ID", ""))
        if not pubchem_id:
            continue
        file_path = os.path.join(drug_dir, get_structure_name(pubchem_id, drug_name=drug_name))
        try:
            if not os.path.exists(file_path):
                download_pubchem(pubchem_id, file_path, drug_name)
            successful_downloads[protein_keys.drugs].append(
                {protein_keys.drug_name: drug_name, protein_keys.sdf_file_path: file_path}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download failed for %s (CID %s): %s", drug_name, pubchem_id, exc)

    langfuse_context.update_current_observation(
        output={
            "proteins_downloaded": len(successful_downloads[protein_keys.proteins]),
            "drugs_downloaded": len(successful_downloads[protein_keys.drugs]),
        }
    )
    return successful_downloads


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_structure_dir(dir_path: str) -> None:
    """Remove a structure directory and everything in it."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def cleanup_protein_structure_dir() -> None:
    cleanup_structure_dir(structure_dirs()[0])


def cleanup_drug_structure_dir() -> None:
    cleanup_structure_dir(structure_dirs()[1])
