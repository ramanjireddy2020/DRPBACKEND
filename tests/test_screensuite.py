"""
ScreenSuite tests — offline. No network, no Vina, no Open Babel, no LLM.

Everything that reaches outside the process is stubbed, so these pin behaviour
rather than the state of RCSB/PubChem or whether the two external executables
happen to be installed on the machine running them.

`tests/conftest.py` initialises Databricks Connect, so run this file with
`--noconftest` locally:

    cd src && python -m pytest ../tests/test_screensuite.py --noconftest -q
"""
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from DRP_Main.app.modules.screening import (  # noqa: E402
    docking_service,
    download_service,
    interaction_service,
    pipeline_service,
    resolution_service,
)
from DRP_Main.app.modules.screening.enums import ResolutionStage, safe_name  # noqa: E402
from DRP_Main.app.modules.screening.router import router  # noqa: E402
from DRP_Main.app.modules.screening.schemas import (  # noqa: E402
    DrugQuery,
    DrugSource,
    ProteinQuery,
    ProteinSource,
    ResolvedDrug,
    ResolvedProtein,
)


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router, prefix="/screening")
    return TestClient(app)


@pytest.fixture()
def isolated_output(tmp_path, monkeypatch):
    """Point every run at a temp directory instead of the repo's data dir."""
    monkeypatch.setattr(
        "DRP_Main.app.core.config.settings.SCREENING_OUTPUT_ROOT", str(tmp_path), raising=False
    )
    return tmp_path


# ── Resolution: identifiers supplied ──────────────────────────────────────────

def test_supplied_identifier_is_not_looked_up(monkeypatch):
    """An identifier the caller already has must cost no network call."""
    def fail(*args, **kwargs):
        raise AssertionError("resolution must not call out when an identifier is supplied")

    monkeypatch.setattr(resolution_service, "_rcsb_search", fail)
    monkeypatch.setattr(resolution_service, "_pubchem_cids", fail)

    result = resolution_service.resolve(
        [ProteinQuery(name="AcrB", identifier="6abj", source=ProteinSource.pdb)],
        [DrugQuery(name="Aspirin", identifier="2244", source=DrugSource.pubchem)],
    )

    assert result.stage == ResolutionStage.resolved
    # PDB codes are normalised upward so downstream file naming is stable.
    assert result.proteins[0].identifier == "6ABJ"
    assert result.proteins[0].source == ProteinSource.pdb
    assert result.drugs[0].identifier == "2244"


def test_source_inferred_from_identifier_shape(monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda *a, **k: [])
    pdb = resolution_service.resolve_protein(ProteinQuery(name="x", identifier="6ABJ"))
    alphafold = resolution_service.resolve_protein(ProteinQuery(name="y", identifier="P00533"))
    assert pdb["resolved"].source == ProteinSource.pdb
    assert alphafold["resolved"].source == ProteinSource.alphafold


def test_unusable_identifier_is_reported_not_guessed():
    outcome = resolution_service.resolve_protein(
        ProteinQuery(name="mystery", identifier="not-an-id")
    )
    assert "unresolved" in outcome
    assert "source" in outcome["unresolved"].reason


def test_zinc_only_when_explicitly_supplied():
    """Resolution never *chooses* ZINC — only honours a supplied ZINC id."""
    outcome = resolution_service.resolve_drug(
        DrugQuery(name="something", identifier="ZINC000123")
    )
    assert outcome["resolved"].source == DrugSource.zinc


# ── Resolution: the confirmation checkpoint ───────────────────────────────────

def test_single_match_resolves_without_pausing(monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A"])
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    result = resolution_service.resolve([ProteinQuery(name="JAK2")], [DrugQuery(name="Chlorthalidone")])

    assert result.stage == ResolutionStage.resolved
    assert not result.ambiguous
    assert result.proteins[0].identifier == "2B7A"


def test_multiple_matches_pause_for_confirmation(monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A", "3KRR"])
    monkeypatch.setattr(resolution_service, "_rcsb_title", lambda pdb_id: f"title {pdb_id}")
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    result = resolution_service.resolve(
        [ProteinQuery(name="JAK2")], [DrugQuery(name="Chlorthalidone")]
    )

    assert result.stage == ResolutionStage.awaiting_confirmation
    assert [c.identifier for c in result.ambiguous[0].candidates] == ["2B7A", "3KRR"]
    # Nothing is auto-selected from an ambiguous shortlist.
    assert result.proteins == []
    assert "Confirm which structure" in result.message


def test_partial_resolution_keeps_what_resolved(monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A", "3KRR"])
    monkeypatch.setattr(resolution_service, "_rcsb_title", lambda pdb_id: None)
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    result = resolution_service.resolve(
        [ProteinQuery(name="JAK2")], [DrugQuery(name="Chlorthalidone")], allow_partial=True
    )

    # Stage still says confirmation is outstanding, but the clean half survives.
    assert result.stage == ResolutionStage.awaiting_confirmation
    assert [d.name for d in result.drugs] == ["Chlorthalidone"]


def test_confirmation_applies_the_choice(monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A", "3KRR"])
    monkeypatch.setattr(resolution_service, "_rcsb_title", lambda pdb_id: None)
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    confirmed = resolution_service.confirm(
        [{"name": "JAK2", "kind": "protein", "identifier": "3KRR", "source": "pdb"}],
        [ProteinQuery(name="JAK2")],
        [DrugQuery(name="Chlorthalidone")],
    )

    assert confirmed.stage == ResolutionStage.resolved
    assert confirmed.proteins[0].identifier == "3KRR"


def test_no_match_is_reported_not_dropped(monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: [])
    monkeypatch.setattr(resolution_service, "_uniprot_accession", lambda name: None)
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: [])

    result = resolution_service.resolve([ProteinQuery(name="zzz")], [DrugQuery(name="qqq")])

    assert result.stage == ResolutionStage.resolved      # nothing ambiguous to confirm
    assert {u.name for u in result.unresolved} == {"zzz", "qqq"}
    assert result.proteins == [] and result.drugs == []


def test_alphafold_fallback_when_no_experimental_structure(monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: [])
    monkeypatch.setattr(resolution_service, "_uniprot_accession", lambda name: ("P00533", "EGFR"))
    monkeypatch.setattr(resolution_service, "_alphafold_exists", lambda accession: True)

    outcome = resolution_service.resolve_protein(ProteinQuery(name="EGFR"))
    assert outcome["resolved"].source == ProteinSource.alphafold
    assert outcome["resolved"].identifier == "P00533"


# ── Download: the hard gate ───────────────────────────────────────────────────

def test_html_error_page_is_a_download_failure(tmp_path, monkeypatch, isolated_output):
    """RCSB answers 200 with HTML for some bad ids — that is not a structure."""
    class FakeResponse:
        status_code = 200
        content = b"<!DOCTYPE html><html><body>" + b"x" * 400 + b"</body></html>"

    monkeypatch.setattr(download_service.requests, "get", lambda *a, **k: FakeResponse())

    _, _, failures = download_service.fetch_resolved_structures(
        [ResolvedProtein(name="Bad", identifier="9ZZZ", source=ProteinSource.pdb)], []
    )

    assert len(failures) == 1
    assert failures[0].stage == "download"
    assert "HTML" in failures[0].error


def test_truncated_response_is_a_download_failure(monkeypatch, isolated_output):
    class FakeResponse:
        status_code = 200
        content = b"ATOM      1  N"          # far too short to be a structure

    monkeypatch.setattr(download_service.requests, "get", lambda *a, **k: FakeResponse())
    _, _, failures = download_service.fetch_resolved_structures(
        [], [ResolvedDrug(name="Bad", identifier="0", source=DrugSource.pubchem)]
    )
    assert failures and "not a structure" in failures[0].error


def test_alphafold_url_comes_from_the_api(monkeypatch, tmp_path):
    """
    A hardcoded model version silently stops working when AlphaFold rolls it,
    so the URL must be read from the prediction API.
    """
    monkeypatch.setattr(
        download_service,
        "get_json",
        lambda url, **kwargs: [{"pdbUrl": "https://example.invalid/AF-P00533-F1-model_v9.pdb"}],
    )
    requested = {}

    class FakeResponse:
        status_code = 200
        content = b"ATOM  " + b"x" * 400

    def fake_get(url, **kwargs):
        requested["url"] = url
        return FakeResponse()

    monkeypatch.setattr(download_service.requests, "get", fake_get)
    download_service.download_alphafold("P00533", str(tmp_path / "af.pdb"), "EGFR")

    assert requested["url"].endswith("model_v9.pdb")


def test_pubchem_falls_back_to_2d_record(monkeypatch, tmp_path):
    """Not every CID has a 3D conformer; RDKit embeds the 2D record instead."""
    attempts = []

    class FakeResponse:
        def __init__(self, status_code, content=b""):
            self.status_code = status_code
            self.content = content

    def fake_get(url, **kwargs):
        attempts.append(url)
        if "record_type=3d" in url:
            return FakeResponse(404)
        return FakeResponse(200, b"\n  test sdf  \n" + b"x" * 400)

    monkeypatch.setattr(download_service.requests, "get", fake_get)
    download_service.download_pubchem("2244", str(tmp_path / "d.sdf"), "Aspirin")

    assert len(attempts) == 2
    assert "record_type=2d" in attempts[1]


def test_failed_download_leaves_no_partial_file(monkeypatch, isolated_output):
    class FakeResponse:
        status_code = 404
        content = b""

    monkeypatch.setattr(download_service.requests, "get", lambda *a, **k: FakeResponse())
    protein_files, _, failures = download_service.fetch_resolved_structures(
        [ResolvedProtein(name="Missing", identifier="9ZZZ", source=ProteinSource.pdb)], []
    )

    # The name must be absent from the files map, so no later stage can use it.
    assert "Missing" not in protein_files
    assert failures[0].kind == "protein"
    protein_dir, _ = download_service.structure_dirs()
    assert not any(Path(protein_dir).glob("*.part"))


# ── Docking box ───────────────────────────────────────────────────────────────

def _pdb_line(
    record: str, serial: int, name: str, resname: str, chain: str, resseq: int, xyz
) -> str:
    """
    One coordinate record at the exact PDB column positions.

    The parser reads fixed columns (resName at 18-20, coordinates at 31-54), so
    a fixture built with loose spacing tests the wrong thing.
    """
    x, y, z = xyz
    return (
        f"{record:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
    )


def _write_pdb(path: Path, *, with_ligand: bool) -> Path:
    """A minimal PDB: a spread-out protein plus an optional bound ligand."""
    lines = [
        _pdb_line("ATOM", index, "CA", "ALA", "A", index, point)
        for index, point in enumerate(
            [(0.0, 0.0, 0.0), (40.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, 0.0, 40.0)], start=1
        )
    ]
    if with_ligand:
        # A tight cluster away from the protein centroid — the "binding site".
        # Every atom shares one residue number: they are one ligand, not three.
        lines += [
            _pdb_line("HETATM", 100 + offset, f"C{offset + 1}", "LIG", "B", 500, point)
            for offset, point in enumerate(
                [(20.0, 20.0, 20.0), (21.0, 20.0, 20.0), (20.0, 21.0, 20.0)]
            )
        ]
        # Solvent must never be mistaken for a ligand.
        lines.append(_pdb_line("HETATM", 200, "O", "HOH", "C", 200, (5.0, 5.0, 5.0)))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_site_box_uses_the_cocrystal_ligand(tmp_path):
    pdb = _write_pdb(tmp_path / "holo.pdb", with_ligand=True)
    box = docking_service.get_binding_site_box(str(pdb), padding=8.0)

    assert box["box_mode"] == "site"
    assert "LIG" in box["box_source"]
    # Centred on the ligand cluster, not the protein centroid (20,20,20).
    assert box["center_x"] == pytest.approx(20.5, abs=0.1)
    assert box["center_z"] == pytest.approx(20.0, abs=0.1)
    # Clamped up to the configured floor rather than a 1 Å box.
    assert box["size_x"] >= 20.0


def test_solvent_is_not_treated_as_a_binding_site(tmp_path):
    pdb = _write_pdb(tmp_path / "waters.pdb", with_ligand=False)
    pdb.write_text(
        pdb.read_text().replace("END", "")
        + _pdb_line("HETATM", 300, "O", "HOH", "C", 300, (1.0, 1.0, 1.0))
        + "\nEND\n"
    )
    assert docking_service.find_cocrystal_ligand(str(pdb)) is None


def test_apo_structure_falls_back_to_blind_box(tmp_path):
    pdb = _write_pdb(tmp_path / "apo.pdb", with_ligand=False)
    box = docking_service.get_binding_site_box(str(pdb))

    assert box["box_mode"] == "protein"
    assert "no bound ligand" in box["box_source"]
    assert box["size_x"] == pytest.approx(40.0, abs=0.01)


def test_box_size_is_clamped(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "DRP_Main.app.core.config.settings.SCREENING_BOX_MAX_SIZE", 30.0, raising=False
    )
    pdb = _write_pdb(tmp_path / "holo.pdb", with_ligand=True)
    box = docking_service.get_binding_site_box(str(pdb), padding=100.0)
    assert box["size_x"] == 30.0


def test_config_written_without_box_metadata(tmp_path):
    """`box_mode`/`box_source` are ours — Vina would reject them as keys."""
    pdb = _write_pdb(tmp_path / "holo.pdb", with_ligand=True)
    config = tmp_path / "conf.txt"
    docking_service.generate_vina_config_file(
        "P", str(pdb), "L", str(tmp_path / "lig.pdbqt"), str(config), str(tmp_path / "out.pdbqt"),
        box=docking_service.get_binding_site_box(str(pdb)),
        num_modes=5,
        exhaustiveness=4,
    )
    text = config.read_text()
    assert "box_mode" not in text and "box_source" not in text
    assert "num_modes = 5" in text and "exhaustiveness = 4" in text
    assert "center_x" in text


# ── Log parsing and ranking ───────────────────────────────────────────────────

VINA_LOG = """
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -9.234      0.000      0.000
   2       -8.100      1.234      2.345
   3       -7.050      2.100      3.400
"""


def test_vina_log_parsing(tmp_path):
    log = tmp_path / "log_P_L.log"
    log.write_text(VINA_LOG)
    records = docking_service.parse_vina_log(str(log), "P", "L", "out.pdbqt")

    assert [r["Mode"] for r in records] == [1, 2, 3]
    assert records[0]["Affinity_kcal_per_mol"] == -9.234
    # The header rows must not be mistaken for affinity rows.
    assert len(records) == 3


def test_ranking_keeps_best_pose_per_ligand(tmp_path):
    records = [
        {"Mode": 1, "Affinity_kcal_per_mol": -7.0, "ligand": "A", "protein": "P", "out_pdbqt_file": "a"},
        {"Mode": 2, "Affinity_kcal_per_mol": -5.0, "ligand": "A", "protein": "P", "out_pdbqt_file": "a"},
        {"Mode": 1, "Affinity_kcal_per_mol": -9.5, "ligand": "B", "protein": "P", "out_pdbqt_file": "b"},
    ]
    top = docking_service.get_top_affinity(records, str(tmp_path / "top.csv"), fraction=0.5)

    # One row per ligand at most, strongest first, and never the weaker pose.
    assert [r["ligand"] for r in top] == ["B"]
    assert top[0]["Affinity_kcal_per_mol"] == -9.5


def test_ranking_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        docking_service.get_top_affinity([], str(tmp_path / "top.csv"))


def test_pose_extraction(tmp_path):
    pdbqt = tmp_path / "poses.pdbqt"
    pdbqt.write_text(
        "MODEL 1\nATOM      1  C   LIG     1       0.000   0.000   0.000\nENDMDL\n"
        "MODEL 2\nATOM      1  C   LIG     1       1.000   1.000   1.000\nENDMDL\n"
    )
    out = docking_service.extract_vina_pose(str(pdbqt), str(tmp_path / "m2.pdbqt"), 2)
    assert "1.000" in Path(out).read_text()

    with pytest.raises(ValueError):
        docking_service.extract_vina_pose(str(pdbqt), str(tmp_path / "m9.pdbqt"), 9)


# ── Missing executables ───────────────────────────────────────────────────────

def test_missing_executable_raises_readable_error(monkeypatch):
    monkeypatch.setattr(docking_service.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "DRP_Main.app.core.config.settings.SCREENING_VINA_PATH", "", raising=False
    )
    monkeypatch.setattr(docking_service.Path, "is_file", lambda self: False)

    with pytest.raises(docking_service.DockingToolError) as excinfo:
        docking_service.vina_executable()
    assert "AutoDock Vina not found" in str(excinfo.value)


def test_blocked_executable_is_not_reported_as_missing(monkeypatch, tmp_path):
    """
    A binary that exists but is refused by the OS is a *policy* problem.

    Endpoint security denies execution of unsigned/newly-written binaries
    (`WinError 5`). Calling that "not found" sends people off reinstalling a
    dependency they already have, so it gets its own error type.
    """
    def deny(*args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(docking_service.subprocess, "run", deny)

    with pytest.raises(docking_service.ExecutableBlockedError) as excinfo:
        docking_service._run_tool(["/bin/whatever", "-V"], label="Open Babel")

    message = str(excinfo.value)
    assert "refused to execute" in message
    assert "allowlist" in message
    assert "not found" not in message
    # It is still a DockingToolError, so existing handlers keep working.
    assert isinstance(excinfo.value, docking_service.DockingToolError)


def test_conversion_falls_back_to_in_process_when_the_binary_is_blocked(monkeypatch, tmp_path):
    """
    Open Babel's binding is the same library reached without execute
    permission, so conversion must survive a blocked executable.
    """
    used = []

    def blocked(*args, **kwargs):
        raise docking_service.ExecutableBlockedError("obabel blocked by policy")

    def fake_pybel(input_file, output_file, in_format, out_format="pdbqt", **kwargs):
        used.append(("pybel", in_format))
        Path(output_file).write_text("ATOM      1  C   LIG\n")
        return output_file

    monkeypatch.setattr(docking_service, "_run_tool", blocked)
    monkeypatch.setattr(docking_service, "convert_with_pybel", fake_pybel)

    for file_type in ("pdb", "sdf"):
        source = tmp_path / f"in.{file_type}"
        source.write_text("ATOM\n")
        result = docking_service.convert_to_pdbqt(
            str(source), str(tmp_path / f"out_{file_type}.pdbqt"), file_type
        )
        assert Path(result).exists()

    assert [route for route, _ in used] == ["pybel", "pybel"]


def test_conversion_error_names_every_route_it_tried(monkeypatch, tmp_path):
    """Reporting only the last failure hides why the other routes didn't work."""
    monkeypatch.setattr(
        docking_service,
        "_run_tool",
        lambda *a, **k: (_ for _ in ()).throw(
            docking_service.ExecutableBlockedError("obabel blocked")
        ),
    )
    monkeypatch.setattr(
        docking_service,
        "convert_with_pybel",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no openbabel")),
    )
    monkeypatch.setattr(
        docking_service,
        "convert_ligand_to_pdbqt_meeko",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no meeko")),
    )

    source = tmp_path / "lig.sdf"
    source.write_text("mock\n")
    with pytest.raises(docking_service.DockingToolError) as excinfo:
        docking_service.convert_to_pdbqt(str(source), str(tmp_path / "o.pdbqt"), "sdf")

    message = str(excinfo.value)
    assert "obabel blocked" in message
    assert "no openbabel" in message
    assert "no meeko" in message


def _write_config(tmp_path) -> str:
    config = tmp_path / "conf.txt"
    config.write_text(
        f"receptor = {tmp_path / 'rec.pdbqt'}\n"
        f"ligand = {tmp_path / 'lig.pdbqt'}\n"
        f"out = {tmp_path / 'poses.pdbqt'}\n"
        "num_modes = 3\nexhaustiveness = 4\n"
        "center_x = 1.0\nsize_x = 20.0\n"
        "center_y = 2.0\nsize_y = 21.0\n"
        "center_z = 3.0\nsize_z = 22.0\n"
    )
    return str(config)


def test_docking_falls_back_to_python_bindings(monkeypatch, tmp_path):
    """
    The `vina` pip wheel ships **no console script** — it is an importable
    module only. So wherever Vina comes from pip (Databricks serverless
    included) there is no executable, and the bindings are the only route.
    """
    monkeypatch.setattr(
        docking_service,
        "vina_executable",
        lambda: (_ for _ in ()).throw(docking_service.DockingToolError("Vina not found")),
    )
    called = {}

    def fake_python_route(config_file, protein_file=None, vina_log_file=None):
        called["config"] = config_file
        return [
            {
                "Mode": 1, "Affinity_kcal_per_mol": -9.4, "protein_ligand": "rec_lig",
                "protein": "rec", "ligand": "lig", "out_pdbqt_file": "poses.pdbqt",
            }
        ]

    monkeypatch.setattr(docking_service, "run_vina_python", fake_python_route)

    records = docking_service.run_vina(
        _write_config(tmp_path), "rec.pdb", vina_log_file=str(tmp_path / "v.log")
    )

    assert records[0]["Affinity_kcal_per_mol"] == -9.4
    assert called["config"].endswith("conf.txt")


def test_docking_error_names_both_routes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        docking_service,
        "vina_executable",
        lambda: (_ for _ in ()).throw(
            docking_service.ExecutableBlockedError("vina blocked by policy")
        ),
    )
    monkeypatch.setattr(
        docking_service,
        "run_vina_python",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'vina'")),
    )

    log = tmp_path / "v.log"
    with pytest.raises(docking_service.DockingToolError) as excinfo:
        docking_service.run_vina(_write_config(tmp_path), "rec.pdb", vina_log_file=str(log))

    message = str(excinfo.value)
    assert "vina blocked by policy" in message
    assert "not installed" in message
    # The diagnostic must survive on disk for the run's artifacts.
    assert "blocked by policy" in log.read_text()


def test_python_route_reads_the_same_config(monkeypatch, tmp_path):
    """
    The bindings take explicit center/box arguments, not a config file, so the
    config has to be parsed into exactly what the executable would have used.
    """
    import sys as _sys
    import types

    captured = {}

    class FakeVina:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def set_receptor(self, path):
            captured["receptor"] = path

        def set_ligand_from_file(self, path):
            captured["ligand"] = path

        def compute_vina_maps(self, center, box_size):
            captured["center"] = center
            captured["box_size"] = box_size

        def dock(self, exhaustiveness, n_poses):
            captured["dock"] = {"exhaustiveness": exhaustiveness, "n_poses": n_poses}

        def write_poses(self, path, n_poses, overwrite):
            captured["out"] = path
            Path(path).write_text("MODEL 1\nENDMDL\n")

        def energies(self, n_poses):
            return [[-9.876, 0, 0, 0, 0], [-8.123, 0, 0, 0, 0]]

    fake_module = types.ModuleType("vina")
    fake_module.Vina = FakeVina
    monkeypatch.setitem(_sys.modules, "vina", fake_module)

    log = tmp_path / "v.log"
    records = docking_service.run_vina_python(
        _write_config(tmp_path), vina_log_file=str(log)
    )

    assert captured["center"] == [1.0, 2.0, 3.0]
    assert captured["box_size"] == [20.0, 21.0, 22.0]
    assert captured["dock"] == {"exhaustiveness": 4, "n_poses": 3}
    # Records must match the executable path's shape exactly.
    assert [r["Mode"] for r in records] == [1, 2]
    assert records[0]["Affinity_kcal_per_mol"] == -9.876
    assert set(records[0]) == {
        "Mode", "Affinity_kcal_per_mol", "protein_ligand", "protein", "ligand",
        "out_pdbqt_file",
    }
    # A log is written either way, so run artifacts look the same.
    assert "-9.876" in log.read_text()
    # And it must be parseable by the same log parser.
    assert len(docking_service.parse_vina_log(str(log), "rec", "lig", "o.pdbqt")) == 2


def test_tool_probe_distinguishes_missing_from_blocked(monkeypatch):
    """Resolution alone is not proof a binary runs, so the probe executes it."""
    monkeypatch.setattr(
        docking_service,
        "_run_tool",
        lambda *a, **k: (_ for _ in ()).throw(
            docking_service.ExecutableBlockedError("denied")
        ),
    )
    blocked = docking_service._probe_executable(lambda: "/path/to/vina", ["--version"], "Vina")
    assert blocked == {
        "available": False,
        "reason": "blocked_by_policy",
        "path": "/path/to/vina",
        "error": "denied",
    }

    def missing():
        raise docking_service.DockingToolError("Vina not found")

    assert docking_service._probe_executable(missing, ["--version"], "Vina")["reason"] == "not_found"


def test_configured_but_absent_path_is_explicit(monkeypatch):
    monkeypatch.setattr(
        "DRP_Main.app.core.config.settings.SCREENING_OBABEL_PATH",
        "/nope/obabel",
        raising=False,
    )
    with pytest.raises(docking_service.DockingToolError) as excinfo:
        docking_service.obabel_executable()
    assert "no such file" in str(excinfo.value)


# ── Interaction reshaping ─────────────────────────────────────────────────────

def test_fingerprint_reshaped_by_interaction_type():
    """
    The §6 fix: rows grouped by interaction type, not column labels.

    ProLIF hands back a MultiIndex frame keyed `(ligand, residue, interaction)`;
    only True cells are real interactions.
    """
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        [[True, False, True], [True, True, False]],
        columns=pandas.MultiIndex.from_tuples(
            [
                ("LIG1", "TYR100.A", "Hydrophobic"),
                ("LIG1", "ASP50.A", "HBDonor"),
                ("LIG1", "PHE20.A", "PiStacking"),
            ]
        ),
    )

    grouped = interaction_service._reshape_fingerprint(frame)

    assert set(grouped) == {"hydrophobic", "hbond_donor", "pi_stacking"}
    # Aliases are normalised: HBDonor -> hbond_donor.
    assert grouped["hbond_donor"][0].residue == "ASP50.A"
    # A False cell is not an interaction.
    assert len(grouped["pi_stacking"]) == 1
    assert len(grouped["hydrophobic"]) == 2
    assert {e.ligand_pose for e in grouped["hydrophobic"]} == {1, 2}


def test_interaction_summary_rolls_up():
    from DRP_Main.app.modules.screening.schemas import InteractionEntry, InteractionReport

    report = InteractionReport(
        status="success",
        interactions_by_type={
            "hydrophobic": [
                InteractionEntry(residue="TYR100.A", interaction_type="hydrophobic", ligand_pose=1),
                InteractionEntry(residue="TYR100.A", interaction_type="hydrophobic", ligand_pose=2),
            ]
        },
    )
    summary = interaction_service.summarize_interactions([report])
    assert summary["interaction_counts_by_type"]["hydrophobic"] == 2
    assert summary["contact_counts_by_residue"]["TYR100.A"] == 2
    assert summary["poses_profiled"] == 1


def test_pose_parser_falls_back_when_meeko_finds_nothing(monkeypatch, tmp_path):
    """
    Regression: only Meeko writes the `REMARK SMILES` header its reader needs.

    An obabel-prepared ligand — the route the platform mandates — yields a Vina
    pose with no such header, and Meeko then returns an empty list. Depending on
    it alone silently broke interaction profiling for every such ligand.
    """
    import sys as _sys
    import types

    # A Meeko that parses the file but recognises no molecule in it.
    fake_meeko = types.ModuleType("meeko")
    fake_meeko.PDBQTMolecule = type(
        "PDBQTMolecule", (), {"from_file": staticmethod(lambda *a, **k: object())}
    )
    fake_meeko.RDKitMolCreate = type(
        "RDKitMolCreate", (), {"from_pdbqt_mol": staticmethod(lambda mol: [None])}
    )
    monkeypatch.setitem(_sys.modules, "meeko", fake_meeko)

    converted = []

    def fake_convert(pose, sdf, in_format, out_format="pdbqt", **kwargs):
        converted.append(kwargs)
        # A minimal valid SDF: one carbon atom.
        Path(sdf).write_text(
            "pose\n     RDKit          3D\n\n"
            "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0\n"
            "M  END\n$$$$\n"
        )
        return sdf

    monkeypatch.setattr(
        "DRP_Main.app.modules.screening.docking_service.convert_with_pybel", fake_convert
    )

    pose = tmp_path / "pose.pdbqt"
    pose.write_text("ATOM      1  C   LIG A   1       0.000   0.000   0.000\n")
    molecules = interaction_service._read_pose_molecules(str(pose))

    assert len(molecules) == 1
    # A pose must not be altered on the way in.
    assert converted[0] == {"add_hydrogens": False, "charges": False}


def test_interaction_report_unavailable_without_prolif(monkeypatch, tmp_path):
    monkeypatch.setattr(interaction_service, "prolif_available", lambda: False)
    report = interaction_service.get_interaction_report(
        str(tmp_path / "rec.pdb"), str(tmp_path / "pose.pdbqt"), mode=1, ligand="L"
    )
    assert report.status == "unavailable"
    assert "prolif" in report.error


# ── Name collisions ───────────────────────────────────────────────────────────

def test_name_collision_cannot_escape_its_run_directory():
    """The invariant is no separators and no traversal, not an exact spelling."""
    for hostile in ("../../etc/passwd", r"..\..\windows", "JAK2/variant B", "a:b*c?"):
        cleaned = safe_name(hostile)
        assert not {"/", "\\", ":", "*", "?"} & set(cleaned)
        assert ".." not in cleaned
        assert cleaned and cleaned not in (".", "..")

    assert safe_name("JAK2/variant B") == "JAK2_variant_B"
    assert safe_name("   ") == "unnamed"


def test_runs_are_scoped_by_id(isolated_output):
    from DRP_Main.app.modules.screening.enums import run_protein_directory

    first = run_protein_directory("run_a", "JAK2")
    second = run_protein_directory("run_b", "JAK2")
    assert first != second
    assert os.path.isdir(first) and os.path.isdir(second)


# ── Batch pipeline ────────────────────────────────────────────────────────────

def test_cost_guard_refuses_oversized_batch(monkeypatch, isolated_output):
    monkeypatch.setattr(
        "DRP_Main.app.core.config.settings.SCREENING_MAX_COMBINATIONS", 4, raising=False
    )
    proteins = [ResolvedProtein(name=f"P{i}", identifier=f"100{i}", source=ProteinSource.pdb) for i in range(3)]
    drugs = [ResolvedDrug(name=f"D{i}", identifier=str(i), source=DrugSource.pubchem) for i in range(3)]

    with pytest.raises(pipeline_service.ScreeningError) as excinfo:
        pipeline_service.screen_batch(proteins, drugs)

    assert "9 docking runs" in str(excinfo.value)
    assert "SCREENING_MAX_COMBINATIONS" in str(excinfo.value)


def test_empty_batch_refused(isolated_output):
    with pytest.raises(pipeline_service.ScreeningError):
        pipeline_service.screen_batch([], [])


def test_download_failure_excludes_protein_and_is_reported(monkeypatch, isolated_output):
    """A failed download must gate the entity out, not flow into docking."""
    monkeypatch.setattr(
        download_service,
        "fetch_resolved_structures",
        lambda proteins, drugs: (
            {},
            {"D": "/tmp/d.sdf"},
            [
                type(
                    "F", (), {}
                )  # placeholder replaced below
            ],
        ),
    )
    # Use the real failure model rather than a stand-in.
    from DRP_Main.app.modules.screening.schemas import EntityFailure

    monkeypatch.setattr(
        download_service,
        "fetch_resolved_structures",
        lambda proteins, drugs: (
            {},
            {"D": "/tmp/d.sdf"},
            [EntityFailure(name="P", kind="protein", stage="download", error="404")],
        ),
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError("docking must not run for an entity that failed download")

    monkeypatch.setattr(pipeline_service, "_dock_protein", must_not_run)

    result = pipeline_service.screen_batch(
        [ResolvedProtein(name="P", identifier="1ABC", source=ProteinSource.pdb)],
        [ResolvedDrug(name="D", identifier="1", source=DrugSource.pubchem)],
    )

    assert result.status == "failed"
    assert result.proteins[0].status.startswith("Failed: structure download")
    assert [f.stage for f in result.failures] == ["download"]


def test_partial_batch_returns_results_and_failures(monkeypatch, isolated_output):
    """One protein succeeding and one failing is a `partial` run, not an error."""
    from DRP_Main.app.modules.screening.schemas import EntityFailure

    monkeypatch.setattr(
        download_service,
        "fetch_resolved_structures",
        lambda proteins, drugs: (
            {"Good": "/tmp/good.pdb"},
            {"D": "/tmp/d.sdf"},
            [EntityFailure(name="Bad", kind="protein", stage="download", error="404")],
        ),
    )

    def fake_dock(protein_name, raw_pdb, ligand_files, output_dir, **kwargs):
        records = [
            {
                "Mode": 1,
                "Affinity_kcal_per_mol": -8.5,
                "protein_ligand": "x",
                "protein": protein_name,
                "ligand": "D",
                "out_pdbqt_file": os.path.join(output_dir, "pose.pdbqt"),
            }
        ]
        return records, [], {"receptor_pdb": os.path.join(output_dir, "rec.pdb"), "box": {}}

    monkeypatch.setattr(pipeline_service, "_dock_protein", fake_dock)

    result = pipeline_service.screen_batch(
        [
            ResolvedProtein(name="Good", identifier="1ABC", source=ProteinSource.pdb),
            ResolvedProtein(name="Bad", identifier="9ZZZ", source=ProteinSource.pdb),
        ],
        [ResolvedDrug(name="D", identifier="1", source=DrugSource.pubchem)],
        skip_interactions=True,
    )

    assert result.status == "partial"
    statuses = {p.protein_name: p.status for p in result.proteins}
    assert statuses["Good"] == "Success"
    assert statuses["Bad"].startswith("Failed")
    assert [f.name for f in result.failures] == ["Bad"]
    assert "best hit D at -8.5" in result.recommendation


def test_each_protein_ranked_independently(monkeypatch, isolated_output):
    """A weak ligand must still top its own protein's table, never be pooled."""
    monkeypatch.setattr(
        download_service,
        "fetch_resolved_structures",
        lambda proteins, drugs: (
            {"Strong": "/tmp/s.pdb", "Weak": "/tmp/w.pdb"},
            {"D1": "/tmp/1.sdf", "D2": "/tmp/2.sdf"},
            [],
        ),
    )

    affinities = {"Strong": (-11.0, -10.0), "Weak": (-4.0, -3.0)}

    def fake_dock(protein_name, raw_pdb, ligand_files, output_dir, **kwargs):
        first, second = affinities[protein_name]
        records = [
            {
                "Mode": 1, "Affinity_kcal_per_mol": value, "protein_ligand": "x",
                "protein": protein_name, "ligand": ligand,
                "out_pdbqt_file": os.path.join(output_dir, f"{ligand}.pdbqt"),
            }
            for ligand, value in (("D1", first), ("D2", second))
        ]
        return records, [], {"receptor_pdb": os.path.join(output_dir, "rec.pdb"), "box": {}}

    monkeypatch.setattr(pipeline_service, "_dock_protein", fake_dock)

    result = pipeline_service.screen_batch(
        [
            ResolvedProtein(name="Strong", identifier="1ABC", source=ProteinSource.pdb),
            ResolvedProtein(name="Weak", identifier="2ABC", source=ProteinSource.pdb),
        ],
        [
            ResolvedDrug(name="D1", identifier="1", source=DrugSource.pubchem),
            ResolvedDrug(name="D2", identifier="2", source=DrugSource.pubchem),
        ],
        skip_interactions=True,
    )

    assert result.status == "success"
    for protein in result.proteins:
        assert len(protein.top_affinity_records) == 1
        assert protein.top_affinity_records[0]["ligand"] == "D1"
    # The weak receptor still reports its own best hit rather than being dropped.
    weak = next(p for p in result.proteins if p.protein_name == "Weak")
    assert weak.top_affinity_records[0]["Affinity_kcal_per_mol"] == -4.0


def test_run_manifest_written_and_readable(monkeypatch, isolated_output):
    monkeypatch.setattr(
        download_service,
        "fetch_resolved_structures",
        lambda proteins, drugs: ({"P": "/tmp/p.pdb"}, {"D": "/tmp/d.sdf"}, []),
    )

    def fake_dock(protein_name, raw_pdb, ligand_files, output_dir, **kwargs):
        records = [
            {
                "Mode": 1, "Affinity_kcal_per_mol": -7.7, "protein_ligand": "x",
                "protein": protein_name, "ligand": "D",
                "out_pdbqt_file": os.path.join(output_dir, "pose.pdbqt"),
            }
        ]
        return records, [], {"receptor_pdb": os.path.join(output_dir, "rec.pdb"), "box": {}}

    monkeypatch.setattr(pipeline_service, "_dock_protein", fake_dock)

    result = pipeline_service.screen_batch(
        [ResolvedProtein(name="P", identifier="1ABC", source=ProteinSource.pdb)],
        [ResolvedDrug(name="D", identifier="1", source=DrugSource.pubchem)],
        run_id="run_manifest_test",
        skip_interactions=True,
    )

    manifest = pipeline_service.get_run_manifest("run_manifest_test")
    assert manifest["run_id"] == "run_manifest_test"
    assert manifest["status"] == "success"
    # Affinity tables are referenced, not just produced.
    assert result.proteins[0].files["affinity_tables"]

    runs = pipeline_service.list_runs()
    assert "run_manifest_test" in [r["run_id"] for r in runs]


# ── API surface ───────────────────────────────────────────────────────────────

def test_health_reports_tool_availability(client):
    response = client.get("/screening/health")
    assert response.status_code == 200
    body = response.json()
    assert body["module"] == "ScreenSuite"
    assert set(body["tools"]) >= {
        "vina", "obabel", "pdbfixer", "openmm", "prolif", "meeko", "obabel_in_process",
    }
    # Every capability flag must be a real probe, not an assumption.
    for field in ("dockable", "structure_preparation", "interaction_profiling"):
        assert isinstance(body[field], bool)
    assert isinstance(body["blocked_by_policy"], list)


def test_health_separates_preparation_from_docking(client, monkeypatch):
    """
    Vina blocked but conversion working is the real state on a locked-down
    host: preparation and profiling still run, only docking cannot.
    """
    monkeypatch.setattr(
        docking_service,
        "tool_availability",
        lambda: {
            "vina": {"available": False, "reason": "blocked_by_policy",
                     "path": "/bin/vina", "error": "refused to execute"},
            "obabel": {"available": False, "reason": "blocked_by_policy",
                       "path": "/bin/obabel", "error": "refused to execute"},
            "vina_in_process": {"available": False, "error": "No module named 'vina'"},
            "obabel_in_process": {"available": True},
            "pdbfixer": {"available": True},
            "openmm": {"available": True},
            "prolif": {"available": True},
            "meeko": {"available": True},
        },
    )
    monkeypatch.setattr(docking_service, "conversion_available", lambda: True)
    monkeypatch.setattr(docking_service, "docking_available", lambda: False)

    body = client.get("/screening/health").json()
    assert body["dockable"] is False            # neither Vina route is available
    assert body["docking_route"] is None
    assert body["structure_preparation"] is True
    assert sorted(body["blocked_by_policy"]) == ["obabel", "vina"]
    assert "refused to execute" in body["blocker"]


def test_health_reports_the_python_binding_route(client, monkeypatch):
    """Vina from pip is a library, so `docking_route` must say which one ran."""
    monkeypatch.setattr(
        docking_service,
        "tool_availability",
        lambda: {
            "vina": {"available": False, "reason": "not_found", "error": "Vina not found"},
            "vina_in_process": {"available": True},
            "obabel": {"available": False, "reason": "not_found", "error": "not found"},
            "obabel_in_process": {"available": True},
            "pdbfixer": {"available": True},
            "openmm": {"available": True},
            "prolif": {"available": True},
            "meeko": {"available": True},
        },
    )
    monkeypatch.setattr(docking_service, "conversion_available", lambda: True)
    monkeypatch.setattr(docking_service, "docking_available", lambda: True)

    body = client.get("/screening/health").json()
    assert body["dockable"] is True
    assert body["docking_route"] == "python-bindings"
    assert body["blocker"] is None


def test_resolve_endpoint(client, monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A"])
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    response = client.post(
        "/screening/resolve",
        json={"proteins": [{"name": "JAK2"}], "drugs": [{"name": "Chlorthalidone"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "resolved"
    assert body["proteins"][0] == {"name": "JAK2", "identifier": "2B7A", "source": "pdb"}


def test_resolve_endpoint_returns_shortlist(client, monkeypatch):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A", "3KRR"])
    monkeypatch.setattr(resolution_service, "_rcsb_title", lambda pdb_id: "kinase")
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    response = client.post(
        "/screening/resolve",
        json={"proteins": [{"name": "JAK2"}], "drugs": [{"name": "Chlorthalidone"}]},
    )
    body = response.json()
    assert body["stage"] == "awaiting_structure_confirmation"
    assert len(body["ambiguous"][0]["candidates"]) == 2


def test_resolve_endpoint_rejects_empty_request(client):
    assert client.post("/screening/resolve", json={"proteins": [], "drugs": []}).status_code == 422


def test_confirm_resolution_endpoint(client, monkeypatch):
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])
    response = client.post(
        "/screening/confirm-resolution",
        json={
            "selections": [
                {"name": "JAK2", "kind": "protein", "identifier": "3KRR", "source": "pdb"}
            ],
            "proteins": [{"name": "JAK2"}],
            "drugs": [{"name": "Chlorthalidone"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "resolved"
    assert body["proteins"][0]["identifier"] == "3KRR"


def test_screen_endpoint_enforces_cost_guard(client, monkeypatch, isolated_output):
    monkeypatch.setattr(
        "DRP_Main.app.core.config.settings.SCREENING_MAX_COMBINATIONS", 1, raising=False
    )
    response = client.post(
        "/screening/screen",
        json={
            "proteins": [
                {"name": "A", "identifier": "1ABC", "source": "pdb"},
                {"name": "B", "identifier": "2ABC", "source": "pdb"},
            ],
            "drugs": [{"name": "D", "identifier": "1", "source": "pubchem"}],
        },
    )
    assert response.status_code == 422
    assert "over the configured limit" in response.json()["detail"]


def test_screen_endpoint_rejects_unresolved_input(client):
    """A record with no identifier is a contract violation, caught at the edge."""
    response = client.post(
        "/screening/screen",
        json={"proteins": [{"name": "JAK2"}], "drugs": [{"name": "Aspirin"}]},
    )
    assert response.status_code == 422


def test_run_not_found(client, isolated_output):
    assert client.get("/screening/runs/nope").status_code == 404


def test_resolve_and_screen_stops_at_confirmation(client, monkeypatch, isolated_output):
    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A", "3KRR"])
    monkeypatch.setattr(resolution_service, "_rcsb_title", lambda pdb_id: None)
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    def must_not_run(*args, **kwargs):
        raise AssertionError("no compute before confirmation")

    monkeypatch.setattr(pipeline_service, "screen_batch", must_not_run)

    response = client.post(
        "/screening/resolve-and-screen",
        json={"proteins": [{"name": "JAK2"}], "drugs": [{"name": "Chlorthalidone"}]},
    )
    assert response.status_code == 200
    assert response.json()["stage"] == "awaiting_structure_confirmation"


# ── Agent wiring ──────────────────────────────────────────────────────────────

def test_agent_surfaces_confirmation_checkpoint(monkeypatch):
    from DRP_Main.app.agents.screensuite.agent import ScreenSuiteAgent

    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A", "3KRR"])
    monkeypatch.setattr(resolution_service, "_rcsb_title", lambda pdb_id: None)
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])

    output = ScreenSuiteAgent().execute(
        {"proteins": [{"name": "JAK2"}], "drugs": [{"name": "Chlorthalidone"}]}, {}
    )

    assert output["stage"] == "awaiting_structure_confirmation"
    assert output["ambiguous"][0]["name"] == "JAK2"
    # The pending queries are carried so a confirmation can be applied later.
    assert output["session_state_update"]["screensuite_pending_proteins"]


def test_agent_carries_over_session_state(monkeypatch):
    """A target from TxKG and candidates from CurateX need no explicit input."""
    from DRP_Main.app.agents.screensuite.agent import ScreenSuiteAgent

    captured = {}

    def fake_batch(proteins, drugs, **kwargs):
        captured["proteins"] = [p.name for p in proteins]
        captured["drugs"] = [d.name for d in drugs]
        from DRP_Main.app.modules.screening.schemas import ScreenBatchResult

        return ScreenBatchResult(run_id="r1", status="success", recommendation="ok")

    monkeypatch.setattr(resolution_service, "_rcsb_search", lambda name, rows: ["2B7A"])
    monkeypatch.setattr(resolution_service, "_pubchem_cids", lambda name: ["2732"])
    monkeypatch.setattr(pipeline_service, "screen_batch", fake_batch)

    output = ScreenSuiteAgent().execute(
        {},
        {"selected_target": "JAK2", "curatex_candidates": ["Chlorthalidone", "Aspirin"]},
    )

    assert captured["proteins"] == ["JAK2"]
    assert captured["drugs"] == ["Chlorthalidone", "Aspirin"]
    assert output["run_id"] == "r1"


def test_agent_requires_both_sides():
    from DRP_Main.app.agents.screensuite.agent import ScreenSuiteAgent

    agent = ScreenSuiteAgent()
    assert "error" in agent.execute({}, {})
    assert "error" in agent.execute({"proteins": [{"name": "JAK2"}]}, {})
