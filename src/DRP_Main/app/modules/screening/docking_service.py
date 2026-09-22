"""
AutoDock Vina docking service.

Two external executables are required and neither is pip-installable: `vina`
and `obabel`. Both are resolved at call time (`SCREENING_VINA_PATH` /
`SCREENING_OBABEL_PATH`, else PATH) so a missing binary fails one docking run
with a readable message instead of breaking module import — and so a Databricks
init script can drop them anywhere on the cluster.

Conversion goes through the **Open Babel executable**, not its Python bindings:
the bindings are an in-process native library that has to match the interpreter
build, while the executable is what the platform's images actually ship. Meeko
is used only when `obabel` is absent and meeko is importable.

Every subprocess call passes an argument list with `subprocess.run(timeout=...)`
rather than a shell string — the previous `timeout N obabel ...` form was a
POSIX-only shell builtin and could not run on Windows at all.
"""
import configparser
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.screening.enums import DataDirectories

logger = get_logger(__name__)

# ── Dict keys ─────────────────────────────────────────────────────────────────
key_mode = "Mode"
key_affinity = "Affinity (kcal/mol)"
key_protein_ligand = "protein_ligand"

#: Residues that are solvent/buffer rather than a bound ligand — never a site.
_NON_LIGAND_HETATMS = {
    "HOH", "DOD", "WAT", "SOL", "EDO", "GOL", "PEG", "PGE", "MPD", "DMS", "ACT",
    "SO4", "PO4", "NO3", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "CU",
    "NI", "CD", "IOD", "BR", "F", "TRS", "EPE", "MES", "IMD", "FMT", "ACY",
}


class DockingToolError(RuntimeError):
    """An external executable is missing, blocked, or failed."""


class ExecutableBlockedError(DockingToolError):
    """
    The binary exists but the OS refused to execute it.

    Endpoint-security software commonly denies execution of unsigned or
    newly-written executables (Windows reports `WinError 5 / Access is denied`).
    This is a machine policy, not a missing dependency, so it is reported
    separately — the fix is an allowlist entry or a different host, never a
    reinstall.
    """


# ── Executable resolution ─────────────────────────────────────────────────────

def _resolve_executable(configured: str, *names: str, label: str) -> str:
    """
    Find an external binary: configured path first, then PATH.

    Also checks a couple of repo-local spots, because the Windows Vina build is
    distributed as a bare .exe that developers drop next to the project.
    """
    configured = (configured or "").strip()
    if configured:
        if os.path.isfile(configured):
            return configured
        raise DockingToolError(f"{label} configured at {configured!r} but no such file exists")

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    # app/modules/screening/ -> app/ -> src/DRP_Main/ -> src/ -> repo root
    search_roots = [Path(__file__).resolve().parents[i] for i in (2, 3, 4, 5)]
    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return str(candidate)

    raise DockingToolError(
        f"{label} not found. Install it and put it on PATH, or set the matching "
        f"path setting (looked for: {', '.join(names)})"
    )


def vina_executable() -> str:
    return _resolve_executable(
        settings.SCREENING_VINA_PATH, "vina", "vina.exe", label="AutoDock Vina"
    )


def obabel_executable() -> str:
    return _resolve_executable(
        settings.SCREENING_OBABEL_PATH, "obabel", "obabel.exe", label="Open Babel"
    )


def _probe_executable(resolver, probe_args: List[str], label: str) -> Dict[str, Any]:
    """
    Resolve a binary *and* actually try to run it.

    Resolution alone is not evidence it works: a binary can be present and
    still be refused by security policy, which is the difference between
    "install it" and "get it allowlisted".
    """
    try:
        path = resolver()
    except DockingToolError as exc:
        return {"available": False, "reason": "not_found", "error": str(exc)}

    try:
        _run_tool([path] + probe_args, label=label, timeout=30)
    except ExecutableBlockedError as exc:
        return {"available": False, "reason": "blocked_by_policy", "path": path, "error": str(exc)}
    except DockingToolError as exc:
        return {"available": False, "reason": "failed_to_run", "path": path, "error": str(exc)}
    return {"available": True, "path": path}


def tool_availability() -> Dict[str, Any]:
    """
    What this environment can actually do. Used by `/screening/health`.

    Reports each external binary as runnable/missing/blocked, and each optional
    Python package separately — `obabel_in_process` is what makes structure
    conversion possible without execute permission.
    """
    status: Dict[str, Any] = {
        "vina": _probe_executable(vina_executable, ["--version"], "AutoDock Vina"),
        "obabel": _probe_executable(obabel_executable, ["-V"], "Open Babel"),
    }
    for label, module in (
        ("vina_in_process", "vina"),
        ("pdbfixer", "pdbfixer"),
        ("openmm", "openmm"),
        ("prolif", "prolif"),
        ("meeko", "meeko"),
        ("obabel_in_process", "openbabel.pybel"),
    ):
        try:
            __import__(module)
            status[label] = {"available": True}
        except Exception as exc:  # noqa: BLE001
            status[label] = {"available": False, "error": str(exc)}
    return status


def conversion_available() -> bool:
    """Whether *any* PDBQT conversion route works here."""
    tools = tool_availability()
    return bool(
        tools["obabel"]["available"]
        or tools["obabel_in_process"]["available"]
        or tools["meeko"]["available"]
    )


def docking_available() -> bool:
    """Whether Vina can run at all — as a binary or as Python bindings."""
    tools = tool_availability()
    return bool(tools["vina"]["available"] or tools["vina_in_process"]["available"])


def _run_tool(command: List[str], *, label: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run an external tool, raising `DockingToolError` with its own stderr."""
    timeout = timeout or int(settings.SCREENING_SUBPROCESS_TIMEOUT or 360)
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except PermissionError as exc:
        # The file is there; the OS refused to run it. Saying "not found" here
        # sends people off reinstalling a dependency they already have.
        raise ExecutableBlockedError(
            f"{label} exists at {command[0]} but the operating system refused to execute it "
            f"({exc.strerror or exc}). Endpoint-security policy commonly blocks unsigned or "
            f"newly-downloaded binaries — this needs an allowlist entry for that path, or a "
            f"host that permits it (a Linux cluster or the Databricks job)."
        ) from exc
    except OSError as exc:
        raise DockingToolError(f"{label} could not be executed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockingToolError(f"{label} timed out after {timeout}s") from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()[-500:]
        raise DockingToolError(f"{label} failed (exit {process.returncode}): {detail}")
    return process


# ── Directory helpers ─────────────────────────────────────────────────────────

def create_data_directories() -> None:
    dirs = [
        DataDirectories.data_dir,
        DataDirectories.minimized_molecule_dir,
        DataDirectories.vina_configs_dir,
        DataDirectories.protein_pdbqt,
        DataDirectories.ligand_pdbqt_dir,
        DataDirectories.vina_output_dir,
        DataDirectories.vina_logs_dir,
        DataDirectories.tmp_dir,
        DataDirectories.vina_csv_dir,
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


# ── Molecule minimisation ─────────────────────────────────────────────────────

@observe(name="molecule_minimization")
def minimize_mol(ip_file: str, op_file: str, overwrite: bool = False) -> str:
    """
    Minimise a ligand from SDF with RDKit's UFF force field.

    A PubChem 2D record has no usable conformer, so coordinates are embedded
    before minimising; a 3D record keeps the conformer it arrived with.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    langfuse_context.update_current_observation(input={"ip_file": ip_file})
    if os.path.exists(op_file) and not overwrite:
        return op_file

    supplier = Chem.SDMolSupplier(ip_file, removeHs=False)
    written = 0
    writer = Chem.SDWriter(op_file)
    try:
        for molecule in supplier:
            if molecule is None:
                continue
            molecule = Chem.AddHs(molecule, addCoords=True)
            if molecule.GetNumConformers() == 0:
                if AllChem.EmbedMolecule(molecule, randomSeed=0xC0FFEE) != 0:
                    logger.warning("Could not embed 3D coordinates for %s", ip_file)
                    continue
            try:
                AllChem.UFFOptimizeMolecule(molecule)
            except Exception as exc:  # noqa: BLE001 — UFF rejects exotic valences
                logger.warning("UFF minimisation skipped for %s: %s", ip_file, exc)
            writer.write(molecule)
            written += 1
    finally:
        writer.close()

    if written == 0:
        raise ValueError(f"no usable molecule parsed from {ip_file}")
    langfuse_context.update_current_observation(output={"op_file": op_file})
    return op_file


# ── PDBQT conversion ──────────────────────────────────────────────────────────

def convert_ligand_to_pdbqt_obabel(sdf_file: str, output_pdbqt: str) -> str:
    """Convert a ligand SDF to PDBQT with the Open Babel executable."""
    _run_tool(
        [
            obabel_executable(),
            "-isdf", sdf_file,
            "-opdbqt", "-O", output_pdbqt,
            "--partialcharge", "gasteiger",
        ],
        label="Open Babel (ligand)",
    )
    _assert_non_empty(output_pdbqt, "Open Babel (ligand)")
    return output_pdbqt


def _pybel():
    """
    Open Babel's in-process Python binding.

    Used when the `obabel` executable is unavailable *or* blocked from
    executing. The binding is the same Open Babel code reached through a
    library call instead of a subprocess, so it needs no execute permission —
    which is the only way conversion can run on a host whose security policy
    denies unknown binaries.
    """
    from openbabel import pybel

    # Open Babel finds its data files (element and space-group tables) through
    # BABEL_DATADIR, which the wheel does not set — hence the "Unable to open
    # data file 'space-groups.txt'" warnings, and degraded perception for any
    # format that needs those tables. The wheel ships them under bin/data.
    if not os.environ.get("BABEL_DATADIR"):
        package_dir = Path(pybel.__file__).resolve().parent
        for candidate in (package_dir / "bin" / "data", package_dir / "data",
                          package_dir / "share" / "openbabel"):
            if candidate.is_dir():
                os.environ["BABEL_DATADIR"] = str(candidate)
                logger.debug("BABEL_DATADIR set to %s", candidate)
                break

    return pybel


def convert_with_pybel(
    input_file: str,
    output_file: str,
    in_format: str,
    out_format: str = "pdbqt",
    *,
    add_hydrogens: bool = True,
    charges: bool = True,
) -> str:
    """
    Convert one structure file in-process.

    `add_hydrogens`/`charges` are off when converting a *pose*: adding atoms or
    recomputing charges on a docked or crystallographic pose changes the
    geometry being analysed.
    """
    pybel = _pybel()
    try:
        molecule = next(pybel.readfile(in_format, input_file))
    except StopIteration as exc:
        raise ValueError(f"no molecule found in {input_file}") from exc

    if add_hydrogens:
        molecule.OBMol.AddHydrogens(False, True, 7.4)
    if charges:
        try:
            molecule.calccharges(model="gasteiger")
        except Exception as exc:  # noqa: BLE001 — some residues have no parameters
            logger.warning("Gasteiger charges incomplete for %s: %s", input_file, exc)
    molecule.write(out_format, output_file, overwrite=True)
    _assert_non_empty(output_file, "Open Babel (in-process)")
    return output_file


def convert_ligand_to_pdbqt_meeko(sdf_file: str, output_pdbqt: str) -> str:
    """Convert a ligand SDF to PDBQT with Meeko. Fallback when obabel is absent."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem

    molecule = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
    if molecule is None:
        raise ValueError(f"no molecule parsed from {sdf_file}")
    molecule = Chem.AddHs(molecule, addCoords=True)
    preparator = MoleculePreparation()
    setups = preparator.prepare(molecule)
    pdbqt_string, is_ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise RuntimeError(f"Meeko ligand prep failed for {sdf_file}: {error}")
    with open(output_pdbqt, "w") as handle:
        handle.write(pdbqt_string)
    return output_pdbqt


def convert_protein_to_pdbqt_obabel(input_file: str, output_file: str) -> str:
    """Convert a receptor PDB to PDBQT with the Open Babel executable."""
    _run_tool(
        [
            obabel_executable(),
            "-ipdb", input_file,
            "-opdbqt", "-O", output_file,
            "--partialcharge", "gasteiger",
            "-xr",  # rigid receptor: no ROOT/BRANCH torsion tree
        ],
        label="Open Babel (receptor)",
    )
    _assert_non_empty(output_file, "Open Babel (receptor)")
    return output_file


def _assert_non_empty(path: str, label: str) -> None:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise DockingToolError(f"{label} produced an empty file: {path}")


def convert_to_pdbqt(
    input_file: str, output_file: str, input_file_type: str, timeout: int = 180
) -> str:
    """
    Convert SDF (ligand) or PDB (receptor) to PDBQT.

    Three routes, tried in order, because no single one works everywhere:

    1. the `obabel` **executable** — what the platform's images ship and what
       the spec calls for;
    2. Open Babel's **in-process binding** — same library, no execute
       permission needed, so it survives a host that blocks unknown binaries;
    3. **Meeko** (ligands only) — a pure-Python preparer.

    Every failure is accumulated, so if all routes fail the error says what was
    actually wrong with each rather than reporting only the last one.
    """
    routes = []
    if input_file_type == "sdf":
        routes = [
            ("obabel executable", lambda: convert_ligand_to_pdbqt_obabel(input_file, output_file)),
            ("obabel in-process", lambda: convert_with_pybel(input_file, output_file, "sdf")),
            ("meeko", lambda: convert_ligand_to_pdbqt_meeko(input_file, output_file)),
        ]
    elif input_file_type == "pdb":
        routes = [
            ("obabel executable", lambda: convert_protein_to_pdbqt_obabel(input_file, output_file)),
            ("obabel in-process", lambda: convert_with_pybel(input_file, output_file, "pdb")),
        ]
    else:
        raise ValueError("input_file_type must be 'sdf' or 'pdb'")

    problems = []
    for name, route in routes:
        try:
            return route()
        except Exception as exc:  # noqa: BLE001 — try the next route
            problems.append(f"{name}: {exc}")
            logger.warning("PDBQT conversion via %s failed: %s", name, exc)

    raise DockingToolError(
        f"could not convert {os.path.basename(input_file)} to PDBQT. " + "; ".join(problems)
    )


# ── Docking box ───────────────────────────────────────────────────────────────

def _atom_coordinates(pdb_file: str, *, records=("ATOM", "HETATM")) -> List[tuple]:
    coordinates = []
    with open(pdb_file, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith(records):
                try:
                    coordinates.append(
                        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    )
                except ValueError:
                    continue
    return coordinates


def _clamp_size(value: float) -> float:
    floor = float(settings.SCREENING_BOX_MIN_SIZE or 20.0)
    ceiling = float(settings.SCREENING_BOX_MAX_SIZE or 40.0)
    return max(floor, min(value, ceiling))


def get_protein_dimensions(protein_file: str) -> Dict[str, float]:
    """
    Whole-protein bounding box, read straight from the coordinate columns.

    This is *blind* docking: the box covers the entire structure, so poses are
    not site-specific and affinities are not comparable between receptors. It is
    the explicit fallback for `get_binding_site_box`, not the default.
    """
    coordinates = _atom_coordinates(protein_file)
    if not coordinates:
        raise ValueError(f"No ATOM/HETATM records found in {protein_file}")

    dimensions: Dict[str, float] = {}
    for index, axis in enumerate(("x", "y", "z")):
        values = [c[index] for c in coordinates]
        dimensions[f"center_{axis}"] = round((max(values) + min(values)) / 2, 3)
        dimensions[f"size_{axis}"] = round(max(values) - min(values), 3)
    return dimensions


def find_cocrystal_ligand(pdb_file: str) -> Optional[Dict[str, Any]]:
    """
    The largest non-solvent HETATM group in the *original* PDB.

    A co-crystallised ligand marks the real binding site, which is why the box
    is derived from the downloaded structure rather than the cleaned receptor —
    preparation strips exactly these records.
    """
    groups: Dict[str, List[tuple]] = {}
    with open(pdb_file, "r", errors="replace") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            residue = line[17:20].strip().upper()
            if residue in _NON_LIGAND_HETATMS:
                continue
            key = f"{residue}|{line[20:22].strip()}|{line[22:26].strip()}"
            try:
                point = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            groups.setdefault(key, []).append(point)

    if not groups:
        return None
    key, points = max(groups.items(), key=lambda item: len(item[1]))
    residue, chain, sequence = key.split("|")
    return {"residue": residue, "chain": chain, "seq": sequence, "coordinates": points}


def get_binding_site_box(
    structure_file: str, padding: Optional[float] = None
) -> Dict[str, Any]:
    """
    Docking box around the detected binding site.

    Uses the co-crystallised ligand's extent plus `padding`, clamped to the
    configured size range. Falls back to the whole-protein extent when the
    structure has no bound ligand (an apo or predicted model), and says which
    it used in `box_mode` so a blind run is never mistaken for a targeted one.
    """
    padding = float(settings.SCREENING_BOX_PADDING if padding is None else padding)
    ligand = find_cocrystal_ligand(structure_file)
    if not ligand:
        box = get_protein_dimensions(structure_file)
        box.update({"box_mode": "protein", "box_source": "whole-protein extent (no bound ligand)"})
        for axis in ("x", "y", "z"):
            box[f"size_{axis}"] = round(box[f"size_{axis}"], 3)
        return box

    box: Dict[str, Any] = {}
    for index, axis in enumerate(("x", "y", "z")):
        values = [point[index] for point in ligand["coordinates"]]
        box[f"center_{axis}"] = round((max(values) + min(values)) / 2, 3)
        box[f"size_{axis}"] = round(_clamp_size((max(values) - min(values)) + 2 * padding), 3)
    box["box_mode"] = "site"
    box["box_source"] = (
        f"co-crystal ligand {ligand['residue']} "
        f"{ligand['chain']}{ligand['seq']} + {padding:g} Å padding"
    )
    return box


def compute_docking_box(
    structure_file: str, mode: Optional[str] = None, padding: Optional[float] = None
) -> Dict[str, Any]:
    """Pick a box per `mode` ("site" | "protein"), defaulting to settings."""
    mode = (mode or settings.SCREENING_BOX_MODE or "site").lower()
    if mode == "protein":
        box = get_protein_dimensions(structure_file)
        box.update({"box_mode": "protein", "box_source": "whole-protein extent (requested)"})
        return box
    return get_binding_site_box(structure_file, padding)


# ── Vina config helpers ───────────────────────────────────────────────────────

def generate_vina_config(conf_file: str, **kwargs) -> None:
    with open(conf_file, "w") as handle:
        for key, value in kwargs.items():
            handle.write(f"{key} = {value}\n")


def load_config(file_path: str) -> configparser.ConfigParser:
    with open(file_path, "r") as handle:
        config_string = "[DEFAULT]\n" + handle.read()
    config = configparser.ConfigParser()
    config.read_string(config_string)
    return config


def remove_lines(input_file: str, output_file: str, keywords: list = None) -> str:
    """Strip a ligand torsion tree from a receptor PDBQT (ROOT/BRANCH/TORSDOF)."""
    if keywords is None:
        keywords = ["ROOT", "BRANCH", "TORSDOF"]
    with open(input_file, "r", errors="replace") as source, open(output_file, "w") as destination:
        for line in source:
            if not any(keyword in line for keyword in keywords):
                destination.write(line)
    return output_file


def generate_vina_config_file(
    protein_name: str,
    protein_file: str,
    ligand_name: str,
    ligand_file: str,
    op_conf_file: str,
    op_result_file: str,
    box: Optional[Dict[str, Any]] = None,
    num_modes: Optional[int] = None,
    exhaustiveness: Optional[int] = None,
) -> str:
    """
    Write a Vina config.

    `box` comes from `compute_docking_box`; when omitted the whole-protein
    extent is used, which is blind docking — see `get_protein_dimensions`.
    """
    box = dict(box or get_protein_dimensions(protein_file))
    box.pop("box_mode", None)
    box.pop("box_source", None)

    vina_config = {
        "receptor": protein_file,
        "ligand": ligand_file,
        "out": op_result_file,
        "num_modes": int(num_modes or settings.SCREENING_NUM_MODES or 9),
        "exhaustiveness": int(exhaustiveness or settings.SCREENING_EXHAUSTIVENESS or 8),
    }
    vina_config.update(box)
    generate_vina_config(op_conf_file, **vina_config)
    logger.info("Generated Vina config for %s docked to %s", ligand_name, protein_name)
    return op_conf_file


# ── Vina docking ──────────────────────────────────────────────────────────────

def vina_python_available() -> bool:
    """Whether Vina's Python bindings can be imported."""
    try:
        import vina  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _config_values(config_file: str, protein_file: str = None) -> Dict[str, Any]:
    """Read a Vina config file into the values both docking routes need."""
    conf = load_config(config_file)["DEFAULT"]
    receptor = conf.get("receptor", protein_file)
    ligand = conf.get("ligand", "")
    return {
        "receptor": receptor,
        "ligand": ligand,
        "out": conf.get("out", ""),
        "protein_name": Path(receptor).stem if receptor else "",
        "ligand_name": Path(ligand).stem if ligand else "",
        "center": [float(conf[f"center_{axis}"]) for axis in ("x", "y", "z")],
        "box_size": [float(conf[f"size_{axis}"]) for axis in ("x", "y", "z")],
        "num_modes": int(conf.get("num_modes", settings.SCREENING_NUM_MODES or 9)),
        "exhaustiveness": int(
            conf.get("exhaustiveness", settings.SCREENING_EXHAUSTIVENESS or 8)
        ),
    }


def run_vina_python(config_file: str, protein_file: str = None, vina_log_file: str = None) -> list:
    """
    Dock with Vina's Python bindings instead of the `vina` command.

    The `vina` PyPI wheel (Linux only, cp38-cp312) ships **no console script** —
    it is an importable module and nothing else. So on a host installing Vina
    from pip, including Databricks serverless, there is no executable to call
    and this is the only route. It also needs no execute permission, so it works
    where policy blocks unknown binaries.

    Produces the same records as the executable path, and writes an equivalent
    log file so a run's artifacts look the same either way.
    """
    from vina import Vina

    values = _config_values(config_file, protein_file)
    engine = Vina(sf_name="vina", cpu=1, seed=0, verbosity=0)
    engine.set_receptor(values["receptor"])
    engine.set_ligand_from_file(values["ligand"])
    engine.compute_vina_maps(center=values["center"], box_size=values["box_size"])
    engine.dock(
        exhaustiveness=values["exhaustiveness"], n_poses=values["num_modes"]
    )

    if values["out"]:
        os.makedirs(os.path.dirname(os.path.abspath(values["out"])), exist_ok=True)
        engine.write_poses(values["out"], n_poses=values["num_modes"], overwrite=True)

    # energies() rows are [total, inter, intra, torsion, -intra]; the first
    # column is the affinity the executable prints in its table.
    energies = engine.energies(n_poses=values["num_modes"])
    records = []
    log_lines = [
        "mode |   affinity | dist from best mode",
        "     | (kcal/mol) | rmsd l.b.| rmsd u.b.",
        "-----+------------+----------+----------",
    ]
    for index, row in enumerate(energies, start=1):
        affinity = round(float(row[0]), 3)
        log_lines.append(f"{index:>4}    {affinity:>10.3f}      0.000      0.000")
        records.append(
            {
                key_mode: index,
                "Affinity_kcal_per_mol": affinity,
                key_protein_ligand: f"{values['protein_name']}_{values['ligand_name']}",
                "protein": values["protein_name"],
                "ligand": values["ligand_name"],
                "out_pdbqt_file": values["out"],
            }
        )

    if vina_log_file:
        os.makedirs(os.path.dirname(os.path.abspath(vina_log_file)), exist_ok=True)
        with open(vina_log_file, "w") as handle:
            handle.write("AutoDock Vina (python bindings)\n")
            handle.write("\n".join(log_lines) + "\n")

    logger.info(
        "Docked %s vs %s in-process: %d poses, best %.2f kcal/mol",
        values["protein_name"], values["ligand_name"], len(records),
        records[0]["Affinity_kcal_per_mol"] if records else float("nan"),
    )
    return records


@observe(name="vina_docking")
def run_vina(config_file: str, protein_file: str, vina_log_file: str = None) -> list:
    """
    Dock one ligand, by whichever Vina route this host offers.

    The `vina` executable first — it is what the official release binaries and
    the platform images provide — then the Python bindings, which are what `pip
    install vina` actually gives you. Both produce identical records.
    """
    langfuse_context.update_current_observation(
        input={"config_file": config_file, "protein_file": protein_file}
    )
    values = _config_values(config_file, protein_file)

    if not vina_log_file:
        conf_name = Path(config_file).stem
        vina_log_file = os.path.join(
            DataDirectories.vina_logs_dir, f"result_{conf_name}.log"
        )
    os.makedirs(os.path.dirname(os.path.abspath(vina_log_file)), exist_ok=True)

    problems = []
    try:
        executable = vina_executable()
        process = _run_tool(
            [executable, "--config", config_file, "--cpu", "1"], label="AutoDock Vina"
        )
        with open(vina_log_file, "w") as handle:
            handle.write((process.stdout or "") + (process.stderr or ""))
        affinity_list = parse_vina_log(
            vina_log_file, values["protein_name"], values["ligand_name"], values["out"]
        )
        if affinity_list:
            langfuse_context.update_current_observation(
                output={
                    "affinity_records": len(affinity_list),
                    "route": "executable",
                    "log_file": vina_log_file,
                }
            )
            return affinity_list
        problems.append("executable: ran but produced no affinity table")
    except DockingToolError as exc:
        problems.append(f"executable: {exc}")
        logger.info("Vina executable unavailable (%s); trying the Python bindings", exc)

    try:
        affinity_list = run_vina_python(config_file, protein_file, vina_log_file)
    except ImportError as exc:
        problems.append(f"python bindings: not installed ({exc})")
        affinity_list = []
    except Exception as exc:  # noqa: BLE001 — report both routes, not just this one
        problems.append(f"python bindings: {exc}")
        affinity_list = []

    if not affinity_list:
        with open(vina_log_file, "w") as handle:
            handle.write("; ".join(problems))
        raise DockingToolError(
            "AutoDock Vina could not run. " + "; ".join(problems)
        )

    langfuse_context.update_current_observation(
        output={
            "affinity_records": len(affinity_list),
            "route": "python-bindings",
            "log_file": vina_log_file,
        }
    )
    return affinity_list


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_vina_log(
    log_file: str, protein: str = None, ligand: str = None, output_file_path: str = None
) -> list:
    """Extract the ranked binding affinities from a Vina log."""
    with open(log_file, "r", errors="replace") as handle:
        lines = handle.readlines()

    log_file_name = Path(log_file).stem
    if not ligand:
        ligand_conf = log_file_name.split(f"{protein}_")[-1]
        ligand = ligand_conf.split("_config")[0]

    affinity_mode_data = []
    affinity_regex = r"^\s*\d+\s+[-+]?\d+\.\d+"
    for line in lines:
        if re.match(affinity_regex, line):
            columns = line.split()
            affinity_mode_data.append(
                {
                    key_mode: int(columns[0]),
                    "Affinity_kcal_per_mol": float(columns[1]),
                    key_protein_ligand: log_file_name,
                    "protein": protein,
                    "ligand": ligand,
                    "out_pdbqt_file": output_file_path,
                }
            )
    return affinity_mode_data


def parse_vina_log_files(protein_name: str, vina_log_dir: str = None) -> list:
    if vina_log_dir is None:
        vina_log_dir = DataDirectories.vina_logs_dir
    log_files = [
        os.path.join(vina_log_dir, f)
        for f in os.listdir(vina_log_dir)
        if f.endswith(".log")
    ]
    affinity_list = []
    for log_file in log_files:
        affinity_list.extend(parse_vina_log(log_file, protein=protein_name))
    return affinity_list


def get_top_affinity(
    affinity_list: list, affinity_file: str, fraction: Optional[float] = None
) -> list:
    """
    Top fraction of ligands, each represented by its own best pose.

    Ranking is per call and never pooled across receptors — a batch ranks each
    protein independently, because affinities from different boxes are not
    comparable.
    """
    if not affinity_list:
        raise ValueError("affinity_list must be provided")

    fraction = float(settings.SCREENING_TOP_FRACTION if fraction is None else fraction)
    frame = pd.DataFrame(affinity_list)
    best_pose_per_ligand = (
        frame.loc[frame.groupby("ligand")["Affinity_kcal_per_mol"].idxmin()]
        .sort_values(by="Affinity_kcal_per_mol")
        .reset_index(drop=True)
    )
    top_count = max(int(len(best_pose_per_ligand) * fraction), 1)
    top_records = best_pose_per_ligand.head(top_count)
    os.makedirs(os.path.dirname(os.path.abspath(affinity_file)), exist_ok=True)
    top_records.to_csv(affinity_file, index=False)
    return top_records.to_dict(orient="records")


def extract_vina_pose(input_pdbqt: str, output_pdbqt: str, mode: int) -> str:
    """Write a single Vina MODEL to its own PDBQT for interaction analysis."""
    with open(input_pdbqt, "r", errors="replace") as handle:
        lines = handle.readlines()

    models = []
    current_model = None
    for line in lines:
        if line.startswith("MODEL"):
            current_model = [line]
        elif current_model is not None:
            current_model.append(line)
            if line.startswith("ENDMDL"):
                models.append(current_model)
                current_model = None

    if not models:
        if mode != 1:
            raise ValueError(f"Pose {mode} is not available in {input_pdbqt}")
        pose_lines = lines
    elif mode < 1 or mode > len(models):
        raise ValueError(f"Pose {mode} is not available in {input_pdbqt}")
    else:
        pose_lines = models[mode - 1]

    with open(output_pdbqt, "w") as handle:
        handle.writelines(pose_lines)
    return output_pdbqt


def is_top_affinity_record(affinity_record: dict) -> bool:
    """Whether an affinity record appears in the saved top-N CSV next to it."""
    output_path = Path(affinity_record["out_pdbqt_file"])
    top_file = output_path.parent / "affinity_results_top_n.csv"
    if not top_file.exists():
        return False

    top_records = pd.read_csv(top_file)
    required = {"Mode", "Affinity_kcal_per_mol", "ligand", "out_pdbqt_file"}
    if not required.issubset(top_records.columns):
        return False

    matches = top_records[
        (top_records["Mode"] == affinity_record["Mode"])
        & (top_records["ligand"] == affinity_record["ligand"])
        & (top_records["out_pdbqt_file"] == affinity_record["out_pdbqt_file"])
    ]
    return not matches.empty


# ── Batch helpers ─────────────────────────────────────────────────────────────

def minimize_list_of_mol(
    sdf_file_list: list, op_dir: str = None, overwrite: bool = False
) -> list:
    results = []
    for sdf_file in sdf_file_list:
        if op_dir:
            op_file = os.path.join(op_dir, f"{Path(sdf_file).stem}_min.sdf")
        else:
            op_file = sdf_file.replace(".sdf", "_min.sdf")
        results.append(minimize_mol(sdf_file, op_file, overwrite=overwrite))
    return results


def convert_list_of_mol_to_pdbqt(file_list: list, file_type: str) -> list:
    pdbqt_list = []
    for ip_file_path in file_list:
        ip_file = os.path.basename(ip_file_path)
        if file_type == "sdf":
            op_file_path = os.path.join(
                DataDirectories.ligand_pdbqt_dir, ip_file.replace(".sdf", ".pdbqt")
            )
        else:
            op_file_path = os.path.join(
                DataDirectories.protein_pdbqt, ip_file.replace(".pdb", ".pdbqt")
            )
        pdbqt_list.append(convert_to_pdbqt(ip_file_path, op_file_path, file_type))
    return pdbqt_list


def cleanup_protein_files(protein_file_list: list) -> list:
    cleaned = []
    for protein_file_path in protein_file_list:
        op_file_path = os.path.join(
            DataDirectories.protein_pdbqt, f"clean_{os.path.basename(protein_file_path)}"
        )
        cleaned.append(remove_lines(protein_file_path, op_file_path))
    return cleaned
