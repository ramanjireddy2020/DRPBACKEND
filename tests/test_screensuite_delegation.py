"""
ScreenSuite → Databricks job delegation. Offline: no Databricks, no network.

Docking cannot run in the API process, so the `/v1` screening runner hands the
work to the ScreenSuite job and rebuilds its response from Gold. These tests pin
the parts that decide whether the frontend gets real results:

* the per-module opt-in (`DRP_DATABRICKS_MODULES`) — so ScreenSuite can move to
  its job without the global switch dragging TxKG onto a placeholder notebook
* the job parameters sent — `run_now` rejects anything undeclared
* Gold rows mapped onto the unchanged `ScreeningHit` wire shape
* failures surfaced with their reason rather than as an empty result

Run:
    cd src && python -m pytest ../tests/test_screensuite_delegation.py --noconftest -q
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from DRP_Main.app.core.config import settings  # noqa: E402
from DRP_Main.app.drp import execution, gold, runners  # noqa: E402
from DRP_Main.app.drp.schemas import ScreeningHit  # noqa: E402


class _Ctx:
    """The two things a runner uses from JobContext."""

    def __init__(self, job_id="job_test_1"):
        self.job_id = job_id
        self.session_id = None
        self.messages = []

    def progress(self, message):
        self.messages.append(message)


# ── Per-module opt-in ─────────────────────────────────────────────────────────

@pytest.fixture()
def screensuite_job_linked(monkeypatch):
    monkeypatch.setenv("JOB_ID_SCREENSUITE", "377908132020938")
    monkeypatch.setenv("JOB_ID_TXKG", "325714213488150")
    monkeypatch.setattr(settings, "DRP_EXECUTION_BACKEND", "inprocess")


def test_opt_in_delegates_only_the_listed_module(monkeypatch, screensuite_job_linked):
    """The whole point: ScreenSuite moves, TxKG (placeholder notebook) stays."""
    monkeypatch.setattr(settings, "DRP_DATABRICKS_MODULES", "ScreenSuite")
    assert execution.should_delegate("ScreenSuite") is True
    assert execution.should_delegate("TxKG") is False


def test_opt_in_is_case_and_space_insensitive(monkeypatch, screensuite_job_linked):
    monkeypatch.setattr(settings, "DRP_DATABRICKS_MODULES", " screensuite , novsearch ")
    assert execution.should_delegate("ScreenSuite") is True


def test_nothing_delegates_by_default(monkeypatch, screensuite_job_linked):
    monkeypatch.setattr(settings, "DRP_DATABRICKS_MODULES", "")
    assert execution.should_delegate("ScreenSuite") is False
    assert execution.should_delegate("TxKG") is False


def test_opt_in_without_a_linked_job_stays_in_process(monkeypatch):
    """An opted-in module with no job id must keep working, not start failing."""
    monkeypatch.delenv("JOB_ID_SCREENSUITE", raising=False)
    monkeypatch.setattr(settings, "DRP_EXECUTION_BACKEND", "inprocess")
    monkeypatch.setattr(settings, "DRP_DATABRICKS_MODULES", "ScreenSuite")
    assert execution.should_delegate("ScreenSuite") is False


def test_global_switch_still_delegates_everything_linked(monkeypatch, screensuite_job_linked):
    monkeypatch.setattr(settings, "DRP_EXECUTION_BACKEND", "databricks")
    monkeypatch.setattr(settings, "DRP_DATABRICKS_MODULES", "")
    assert execution.should_delegate("ScreenSuite") is True
    assert execution.should_delegate("TxKG") is True


# ── Job query payload ─────────────────────────────────────────────────────────

def test_query_payload_from_names():
    query = runners._screensuite_job_query(
        "JAK2", {"compounds": [{"drug_name": "Ruxolitinib"}, {"drug_name": "  "}]}
    )
    assert query == {"proteins": [{"name": "JAK2"}], "drugs": [{"name": "Ruxolitinib"}]}


def test_pdb_id_target_is_sent_as_an_identifier():
    """A PDB code docks that exact structure — the way past an ambiguous gene name."""
    query = runners._screensuite_job_query("6vgl", {"compounds": [{"drug_name": "X"}]})
    assert query["proteins"] == [{"name": "6vgl", "identifier": "6VGL", "source": "pdb"}]


def test_compound_identifier_passes_through():
    query = runners._screensuite_job_query(
        "JAK2",
        {"compounds": [{"drug_name": "Ruxolitinib", "identifier": "25126798", "source": "PubChem"}]},
    )
    assert query["drugs"] == [
        {"name": "Ruxolitinib", "identifier": "25126798", "source": "pubchem"}
    ]


# ── Runner ────────────────────────────────────────────────────────────────────

@pytest.fixture()
def delegated(monkeypatch):
    """Delegation on, with the job and Gold stubbed."""
    calls = {}

    async def fake_run_module(module, params, ctx):
        calls["module"] = module
        calls["params"] = params
        return 12345

    monkeypatch.setattr(execution, "should_delegate", lambda module: module == "ScreenSuite")
    monkeypatch.setattr(execution, "run_module", fake_run_module)
    monkeypatch.setattr(runners, "_persist_screensuite_hits", lambda *a, **k: None)
    return calls


def _gold_rows(rows):
    def reader(table, drp_job_id, **kwargs):
        assert table == "docking_results"
        return rows
    return reader


def test_runner_sends_only_the_declared_query_parameter(monkeypatch, delegated):
    """`run_now` rejects undeclared job parameters, so target/compounds must not leak."""
    monkeypatch.setattr(gold, "read_job_rows", _gold_rows([
        {"run_id": "r1", "protein": "JAK2", "ligand": "Ruxolitinib", "mode": "1",
         "affinity_kcal_mol": "-8.342", "pose_file": "/tmp/p.pdbqt", "status": "Success"},
    ]))

    asyncio.run(runners.run_screensuite_screen(
        {"target": "JAK2", "compoundLibrary": "lib", "compounds": [{"drug_name": "Ruxolitinib"}]},
        _Ctx(),
    ))

    assert delegated["module"] == "ScreenSuite"
    assert set(delegated["params"]) == {"query"}
    assert json.loads(delegated["params"]["query"]) == {
        "proteins": [{"name": "JAK2"}], "drugs": [{"name": "Ruxolitinib"}],
    }


def test_runner_maps_gold_rows_onto_the_unchanged_wire_shape(monkeypatch, delegated):
    """Gold returns text; the frontend's ScreeningHit expects typed, camelCase fields."""
    monkeypatch.setattr(gold, "read_job_rows", _gold_rows([
        {"run_id": "r1", "protein": "JAK2", "ligand": "Weak", "mode": "2",
         "affinity_kcal_mol": "-5.1", "pose_file": "/tmp/w.pdbqt", "status": "Success"},
        {"run_id": "r1", "protein": "JAK2", "ligand": "Ruxolitinib", "mode": "1",
         "affinity_kcal_mol": "-8.342", "pose_file": "/tmp/r.pdbqt", "status": "Success"},
    ]))

    result = asyncio.run(runners.run_screensuite_screen(
        {"target": "JAK2", "compounds": [{"drug_name": "Ruxolitinib"}, {"drug_name": "Weak"}]},
        _Ctx(),
    ))

    # Strongest binder first, and every hit valid for the wire model.
    assert [h["compound"] for h in result["hits"]] == ["Ruxolitinib", "Weak"]
    first = ScreeningHit(**result["hits"][0])
    assert first.affinityKcalPerMol == -8.342
    assert first.mode == 1
    assert first.outputFile == "/tmp/r.pdbqt"
    assert result["status"] == "success"
    assert result["runId"] == "r1"


def test_runner_keeps_partial_results_and_reports_failures(monkeypatch, delegated):
    monkeypatch.setattr(gold, "read_job_rows", _gold_rows([
        {"run_id": "r1", "protein": "JAK2", "ligand": "Ruxolitinib", "mode": "1",
         "affinity_kcal_mol": "-8.3", "pose_file": "p", "status": "Success"},
        {"run_id": "r1", "protein": "", "ligand": "Bogus",
         "status": "Failed[resolution]: 'Bogus' — no PubChem compound found for this name"},
    ]))

    result = asyncio.run(runners.run_screensuite_screen(
        {"target": "JAK2", "compounds": [{"drug_name": "Ruxolitinib"}, {"drug_name": "Bogus"}]},
        _Ctx(),
    ))

    assert result["status"] == "partial"
    assert len(result["hits"]) == 1
    assert "no PubChem compound" in result["failures"][0]


def test_runner_surfaces_the_reason_when_nothing_docked(monkeypatch, delegated):
    """An ambiguous gene name must come back as the shortlist, not an empty list."""
    monkeypatch.setattr(gold, "read_job_rows", _gold_rows([
        {"run_id": "", "protein": "JAK2", "ligand": "",
         "status": "Failed[resolution]: 'JAK2' matches several structures — resend with one "
                   "identifier. Candidates: 8C08, 6VGL (JAK2 JH1 in complex with ruxolitinib)"},
    ]))

    with pytest.raises(runners.RunnerError) as excinfo:
        asyncio.run(runners.run_screensuite_screen(
            {"target": "JAK2", "compounds": [{"drug_name": "Ruxolitinib"}]}, _Ctx(),
        ))
    assert "6VGL" in str(excinfo.value)
    assert "resend with one identifier" in str(excinfo.value)


def test_runner_requires_compounds_before_starting_the_job(monkeypatch, delegated):
    """No compounds means nothing to dock — refuse before spending compute."""
    with pytest.raises(runners.RunnerError) as excinfo:
        asyncio.run(runners.run_screensuite_screen({"target": "JAK2"}, _Ctx()))
    assert "compound" in str(excinfo.value)
    assert "module" not in delegated


def test_runner_reports_unreadable_gold_distinctly(monkeypatch, delegated):
    def unavailable(*args, **kwargs):
        raise gold.GoldUnavailable("warehouse not reachable")

    monkeypatch.setattr(gold, "read_job_rows", unavailable)
    with pytest.raises(runners.RunnerError) as excinfo:
        asyncio.run(runners.run_screensuite_screen(
            {"target": "JAK2", "compounds": [{"drug_name": "Ruxolitinib"}]}, _Ctx(),
        ))
    assert "could not be read" in str(excinfo.value)


def test_in_process_path_is_untouched_when_not_delegated(monkeypatch):
    """Without opt-in, the runner must not reach for Databricks at all."""
    monkeypatch.setattr(execution, "should_delegate", lambda module: False)

    async def must_not_run(*args, **kwargs):
        raise AssertionError("run_module must not be called in-process")

    monkeypatch.setattr(execution, "run_module", must_not_run)
    # No previous results on disk for this target → the existing in-process error.
    with pytest.raises(runners.RunnerError) as excinfo:
        asyncio.run(runners.run_screensuite_screen({"target": "NoSuchTargetXYZ"}, _Ctx()))
    assert "No screening results exist" in str(excinfo.value)
