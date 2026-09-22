"""
Receptor preparation — strip non-protein components, fill gaps, add hydrogens.

PDBFixer (with OpenMM) replaces the PyMOL path this module used to take: it is
pip-installable, needs no GUI process, and does the residue/atom repair PyMOL
never did. Both are imported lazily — `pdbfixer`/`openmm` are optional, and a
module-level import would make the whole screening router disappear from the API
on any environment without them (`api/v1/router.py` skips a router it cannot
import).

Without PDBFixer the fallback cleaner still produces a dockable receptor, but it
cannot add hydrogens or rebuild missing atoms. That degradation is logged and
reported in the returned metadata rather than passed off as a full preparation.
"""
import os
from typing import Any, Dict

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)

#: Records that never belong in a docking receptor.
_DROP_RECORDS = ("HETATM", "ANISOU", "CONECT", "MASTER", "HYDBND", "SIGATM", "SIGUIJ")
#: Solvent/ion residues that survive as ATOM records in some depositions.
_SOLVENT_RESIDUES = {"HOH", "DOD", "WAT", "SOL", "TIP3", "NA", "CL", "K", "MG", "CA", "ZN", "SO4", "PO4"}


def _pdbfixer_available() -> bool:
    """Whether the full preparation path can run in this environment."""
    try:
        import openmm.app  # noqa: F401
        import pdbfixer  # noqa: F401
    except Exception:  # noqa: BLE001 — ImportError, or a broken native build
        return False
    return True


def _basic_clean_pdb(input_pdb: str, output_pdb: str) -> Dict[str, Any]:
    """
    Text-level fallback cleaner: keep protein ATOM records, drop everything else.

    Also collapses alternate locations to the first conformer — Vina treats every
    altloc as a real atom and would otherwise dock into an overlapping receptor.
    """
    kept = 0
    with open(input_pdb, "r", errors="replace") as source, open(output_pdb, "w") as destination:
        for line in source:
            record = line[:6].strip()
            if record in ("MODEL", "ENDMDL"):
                # Only the first model of an NMR ensemble is used.
                if record == "ENDMDL":
                    break
                continue
            if record not in ("ATOM", "TER"):
                continue
            if record == "ATOM":
                if line[17:20].strip().upper() in _SOLVENT_RESIDUES:
                    continue
                altloc = line[16]
                if altloc not in (" ", "A"):
                    continue
                kept += 1
            destination.write(line)
        destination.write("END\n")

    if kept == 0:
        raise RuntimeError(f"no protein ATOM records survived cleaning of {input_pdb}")
    logger.warning(
        "PDBFixer unavailable — %s cleaned without hydrogens or missing-atom repair (%d atoms). "
        "Install pdbfixer+openmm for full preparation.",
        os.path.basename(output_pdb),
        kept,
    )
    return {"method": "basic", "hydrogens_added": False, "atoms": kept}


@observe(name="pdb_preparation")
def clean_pdb(input_pdb: str, output_pdb: str, timeout: int = 60, ph: float = 7.0) -> Dict[str, Any]:
    """
    Prepare a receptor for docking.

    Removes heterogens and waters, replaces nonstandard residues, fills missing
    atoms on residues that are present, and adds hydrogens at `ph`. Missing
    *loops* are deliberately not built — modelling absent residues invents
    geometry the structure never contained, which a docking box would then treat
    as real surface.

    Returns preparation metadata; the caller records it so a downgraded run is
    visible in results rather than silent.
    """
    langfuse_context.update_current_observation(
        input={"input_pdb": input_pdb, "output_pdb": output_pdb}
    )
    if not os.path.isfile(input_pdb):
        raise FileNotFoundError(f"receptor not found: {input_pdb}")
    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)), exist_ok=True)

    if not _pdbfixer_available():
        metadata = _basic_clean_pdb(input_pdb, output_pdb)
        langfuse_context.update_current_observation(output=metadata)
        return metadata

    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    logger.info("Preparing receptor with PDBFixer: %s", input_pdb)
    fixer = PDBFixer(filename=input_pdb)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # don't build missing loops
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(output_pdb, "w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)

    metadata = {
        "method": "pdbfixer",
        "hydrogens_added": True,
        "atoms": fixer.topology.getNumAtoms(),
        "ph": ph,
    }
    logger.info("Prepared receptor %s (%d atoms)", output_pdb, metadata["atoms"])
    langfuse_context.update_current_observation(output=metadata)
    return metadata
