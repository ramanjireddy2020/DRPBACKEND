"""
Interaction profiling — ProLIF fingerprints for a docked pose.

Replaces the PLIP + PyMOL path. ProLIF is pip-installable and reads the Vina
output directly through Meeko, so no PDB-complex assembly step is needed.

The output is organised **by interaction type**, each holding its actual rows.
ProLIF's native `to_dataframe()` gives a MultiIndex frame whose columns are
`(ligand, residue, interaction)` tuples; returning those column labels (or a
`to_json()` dump of the frame) puts the shape of a dataframe on the wire instead
of the findings, which is what §6 of the spec calls out. `_reshape_fingerprint`
is the part that had to be validated against a real docked pose.

Every heavy import is inside a function: `prolif`, `MDAnalysis` and `meeko` are
optional, and a module-level import would drop the entire screening router from
the API on environments without them.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.screening.docking_service import extract_vina_pose
from DRP_Main.app.modules.screening.schemas import InteractionEntry, InteractionReport

logger = get_logger(__name__)

#: Canonical interaction buckets, so a report always has the same keys even
#: when a pose makes no interaction of a given type.
INTERACTION_TYPES = [
    "hydrophobic",
    "hbond_donor",
    "hbond_acceptor",
    "halogen_bond",
    "pi_stacking",
    "pi_cation",
    "cation_pi",
    "anionic",
    "cationic",
    "metal_donor",
    "metal_acceptor",
    "water_bridge",
    "vdw_contact",
    "xbond",
]

#: ProLIF spells some interactions differently across versions.
_TYPE_ALIASES = {
    "hbdonor": "hbond_donor",
    "hbacceptor": "hbond_acceptor",
    "hydrophobic": "hydrophobic",
    "pistacking": "pi_stacking",
    "facetoface": "pi_stacking",
    "edgetoface": "pi_stacking",
    "pication": "pi_cation",
    "cationpi": "cation_pi",
    "xbdonor": "halogen_bond",
    "xbacceptor": "halogen_bond",
    "xbond": "halogen_bond",
    "halogenbond": "halogen_bond",
    "metaldonor": "metal_donor",
    "metalacceptor": "metal_acceptor",
    "waterbridge": "water_bridge",
    "vdwcontact": "vdw_contact",
}


def _normalize_type(raw: str) -> str:
    key = str(raw).replace("_", "").replace("-", "").lower()
    return _TYPE_ALIASES.get(key, str(raw).lower())


def prolif_available() -> bool:
    try:
        import MDAnalysis  # noqa: F401
        import meeko  # noqa: F401
        import prolif  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _read_pose_molecules(pose_pdbqt: str) -> List[Any]:
    """
    Load a pose PDBQT as RDKit molecules, by whichever route works.

    Meeko's `RDKitMolCreate` is tried first: it rebuilds correct bond orders
    from the `REMARK SMILES` header Meeko writes, which is the best input ProLIF
    can get. But **only Meeko writes that header** — a ligand prepared with Open
    Babel (the route the platform mandates) produces a Vina pose with no SMILES
    remark, and Meeko then returns nothing. Relying on it alone silently broke
    interaction profiling for every obabel-prepared ligand.

    The fallback converts the pose to an SDF in-process and lets RDKit perceive
    it. Bond orders are inferred from geometry rather than known exactly, so
    Meeko stays the preferred route.
    """
    from DRP_Main.app.modules.screening.docking_service import convert_with_pybel

    try:
        from meeko import PDBQTMolecule, RDKitMolCreate

        pdbqt_molecule = PDBQTMolecule.from_file(pose_pdbqt, skip_typing=True)
        molecules = [m for m in RDKitMolCreate.from_pdbqt_mol(pdbqt_molecule) if m is not None]
        if molecules:
            return molecules
        logger.info(
            "Meeko parsed no molecule from %s (no REMARK SMILES header — ligand was not "
            "Meeko-prepared); falling back to Open Babel perception",
            Path(pose_pdbqt).name,
        )
    except Exception as exc:  # noqa: BLE001 — missing meeko, or an unparseable header
        logger.info("Meeko could not read %s (%s); falling back", Path(pose_pdbqt).name, exc)

    from rdkit import Chem

    sdf_path = str(Path(pose_pdbqt).with_suffix(".pose.sdf"))
    # A pose must not be altered: no added hydrogens, no recomputed charges.
    convert_with_pybel(
        pose_pdbqt, sdf_path, "pdbqt", "sdf", add_hydrogens=False, charges=False
    )

    molecules = [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m is not None]
    if molecules:
        return molecules

    # Geometry-perceived bond orders can leave a valence RDKit rejects. ProLIF
    # needs ring/aromaticity perception but not valence correctness, so retry
    # with the valence check skipped rather than losing the pose entirely.
    logger.warning(
        "Strict sanitization rejected %s; retrying with valence checking relaxed",
        Path(sdf_path).name,
    )
    relaxed = []
    for molecule in Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False):
        if molecule is None:
            continue
        try:
            Chem.SanitizeMol(
                molecule,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception as exc:  # noqa: BLE001 — an unusable pose, not a crash
            logger.warning("Could not sanitize a pose from %s: %s", Path(sdf_path).name, exc)
            continue
        relaxed.append(molecule)
    return relaxed


def _reshape_fingerprint(frame) -> Dict[str, List[InteractionEntry]]:
    """
    Turn a ProLIF fingerprint frame into rows grouped by interaction type.

    Columns are `(ligand, residue, interaction)`; the index is one row per pose.
    Only `True` cells are interactions — a False cell means that interaction was
    tested for and not found, and emitting it would inflate every report.
    """
    grouped: Dict[str, List[InteractionEntry]] = {}
    for column in frame.columns:
        parts = list(column) if isinstance(column, tuple) else [column]
        # Take residue/interaction from the tail, so both 2- and 3-level
        # column layouts (with or without the ligand level) are handled.
        interaction = _normalize_type(parts[-1])
        residue = str(parts[-2]) if len(parts) >= 2 else "unknown"

        series = frame[column]
        for position, value in enumerate(series.tolist()):
            if not bool(value):
                continue
            grouped.setdefault(interaction, []).append(
                InteractionEntry(
                    residue=residue,
                    interaction_type=interaction,
                    ligand_pose=position + 1,
                    count=int(value) if not isinstance(value, bool) else 1,
                )
            )

    for entries in grouped.values():
        entries.sort(key=lambda entry: (entry.ligand_pose, entry.residue))
    return grouped


@observe(name="interaction_analysis")
def get_interaction_report(
    receptor_pdb: str,
    vina_output_pdbqt: str,
    mode: int = 1,
    ligand: Optional[str] = None,
    protein: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> InteractionReport:
    """
    Profile one docked pose against the prepared receptor.

    Returns a report with `status="unavailable"` when ProLIF's stack is not
    installed, and `status="failed"` with the reason on an analysis error — the
    caller records either outcome as a per-entity failure rather than losing the
    docking result that produced the pose.
    """
    langfuse_context.update_current_observation(
        input={"receptor": receptor_pdb, "pose_file": vina_output_pdbqt, "mode": mode}
    )
    ligand_label = ligand or Path(vina_output_pdbqt).stem
    destination = Path(output_dir) if output_dir else Path(receptor_pdb).parent
    report_path = destination / f"interaction_{ligand_label}_mode_{mode}.json"

    if not prolif_available():
        return InteractionReport(
            status="unavailable",
            protein=protein,
            ligand=ligand_label,
            mode=mode,
            error="prolif, MDAnalysis and meeko are required for interaction profiling",
        )

    import MDAnalysis as mda
    import prolif as plf
    from MDAnalysis.converters.RDKitInferring import MDAnalysisInferrer

    pose_path = Path(vina_output_pdbqt).with_name(
        f"{Path(vina_output_pdbqt).stem}_mode_{mode}.pdbqt"
    )
    extract_vina_pose(vina_output_pdbqt, str(pose_path), mode)

    universe = mda.Universe(receptor_pdb)
    protein_atoms = universe.select_atoms("protein")
    if protein_atoms.n_atoms == 0:
        raise ValueError(f"no protein atoms selected from {receptor_pdb}")
    receptor = plf.Molecule.from_mda(
        protein_atoms, inferrer=MDAnalysisInferrer(sanitize=False)
    )

    rdkit_poses = _read_pose_molecules(str(pose_path))
    poses = [plf.Molecule.from_rdkit(m) for m in rdkit_poses if m is not None]
    if not poses:
        raise ValueError(f"No valid poses parsed from {vina_output_pdbqt}")

    fingerprint = plf.Fingerprint()
    fingerprint.run_from_iterable(poses, receptor, n_jobs=1)
    frame = fingerprint.to_dataframe()

    interactions_by_type = _reshape_fingerprint(frame)
    residues = sorted(
        {entry.residue for entries in interactions_by_type.values() for entry in entries}
    )
    report = InteractionReport(
        status="success",
        protein=protein,
        ligand=ligand_label,
        mode=mode,
        interactions_by_type=interactions_by_type,
        binding_site_residues=residues,
        interaction_count=sum(len(entries) for entries in interactions_by_type.values()),
    )

    destination.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(), handle, indent=2)
    report.report_file = str(report_path)

    langfuse_context.update_current_observation(
        output={
            "interaction_types": list(interactions_by_type.keys()),
            "interaction_count": report.interaction_count,
        }
    )
    return report


def summarize_interactions(reports: List[InteractionReport]) -> Dict[str, Any]:
    """Roll several pose reports up into per-type and per-residue totals."""
    by_type: Dict[str, int] = {}
    by_residue: Dict[str, int] = {}
    for report in reports:
        for interaction_type, entries in (report.interactions_by_type or {}).items():
            by_type[interaction_type] = by_type.get(interaction_type, 0) + len(entries)
            for entry in entries:
                by_residue[entry.residue] = by_residue.get(entry.residue, 0) + 1
    return {
        "interaction_counts_by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "contact_counts_by_residue": dict(sorted(by_residue.items(), key=lambda kv: -kv[1])),
        "poses_profiled": sum(1 for r in reports if r.status == "success"),
    }
