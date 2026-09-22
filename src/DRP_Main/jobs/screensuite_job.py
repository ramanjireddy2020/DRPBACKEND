# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4 — Screening Suite (Docking)
# MAGIC
# MAGIC Runs the **real** `app/modules/screening` pipeline:
# MAGIC resolution → structure retrieval → receptor prep (PDBFixer) → PDBQT conversion
# MAGIC (Open Babel) → AutoDock Vina → per-protein ranking → ProLIF interaction profiling.
# MAGIC Writes `gold.docking_results`.
# MAGIC
# MAGIC **Why this runs here and not in the API process:** Vina is a native dependency.
# MAGIC `pip install vina` publishes manylinux/musllinux wheels for cp38–cp312 and **no
# MAGIC Windows or macOS wheel at all**, so a Linux job is the only place the pip route
# MAGIC works. The wheel also ships *no console script* — it is an importable module —
# MAGIC which is why `docking_service.run_vina` falls back to the Python bindings when no
# MAGIC `vina` executable is on PATH.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dependencies
# MAGIC
# MAGIC The application code is **not** installed as a wheel. The bundle already syncs
# MAGIC the source tree next to this notebook (`<bundle root>/files/src/`), so the code
# MAGIC below puts that directory on `sys.path` — derived from this notebook's own
# MAGIC location, which keeps the job correct for any user and any bundle target
# MAGIC instead of pinning one person's workspace folder. It also means the job always
# MAGIC runs exactly the code that was deployed with it.
# MAGIC
# MAGIC The app's full dependency list is deliberately not installed: it includes
# MAGIC `databricks-connect`, which must never be installed on Databricks compute (it
# MAGIC collides with the runtime's own PySpark). Only what the screening module
# MAGIC imports is installed below.

# COMMAND ----------
# MAGIC %md
# MAGIC Installed one group per cell, deliberately. A single combined `%pip install`
# MAGIC reports only `returned non-zero exit status 1` without naming the package that
# MAGIC could not be resolved against the runtime's immutable constraints, which makes
# MAGIC a dependency conflict untraceable from the run output alone.

# COMMAND ----------
# MAGIC %pip install "langfuse==2.60.10" pydantic-settings xmltodict

# COMMAND ----------
# MAGIC %pip install rdkit

# COMMAND ----------
# MAGIC %md
# MAGIC `vina` is deliberately **not** pip-installed. It publishes wheels for x86_64
# MAGIC only, and this compute is aarch64 — pip therefore falls back to the source
# MAGIC distribution and dies on `Boost library location was not found`. The official
# MAGIC release does ship a prebuilt `linux_aarch64` binary, which is downloaded below
# MAGIC and runs fine from `/tmp`.

# COMMAND ----------
# MAGIC %pip install openbabel-wheel

# COMMAND ----------
# MAGIC %pip install pdbfixer openmm

# COMMAND ----------
# MAGIC %pip install prolif MDAnalysis meeko

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %run ./_common

# COMMAND ----------
import json
import os
import platform
import stat
import subprocess
import sys
import urllib.request

# ── Application code ─────────────────────────────────────────────────────────
# This notebook is synced to <bundle root>/files/src/DRP_Main/jobs/, so the
# importable `src/` directory is three levels up from it. Derived at runtime
# rather than hardcoded, so the job works for whichever user and target
# deployed it.
notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
if not os.path.isdir(os.path.join(src_dir, "DRP_Main", "app")):
    raise RuntimeError(
        f"application source not found under {src_dir} (derived from {notebook_path}). "
        "Is this notebook running from a bundle deployment?"
    )
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
print(f"[screensuite] application source: {src_dir}")

p = get_params()
# `.get` rather than indexing: `drp_job_id` was added to `_common.get_params()`
# later than the first deployment, and a workspace copy can lag the repo. A
# manually started run has no job id either.
catalog, schema = p["catalog"], p["schema"]
drp_job_id = p.get("drp_job_id") or ""

# ── AutoDock Vina ────────────────────────────────────────────────────────────
# Fetched as the official prebuilt binary for this machine's architecture.
# `pip install vina` is not an option here: wheels exist for x86_64 only, so on
# aarch64 pip builds from source and fails for want of Boost. The binary is
# written to /tmp — the only writable-and-executable location on serverless
# (/local_disk0 refuses writes).
VINA_VERSION = "1.2.7"
VINA_ASSETS = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
    "arm64": "linux_aarch64",
}


def ensure_vina() -> str:
    """Download the matching Vina binary and return its path."""
    machine = platform.machine()
    asset = VINA_ASSETS.get(machine)
    if not asset:
        raise RuntimeError(
            f"no prebuilt AutoDock Vina binary for architecture {machine!r}. "
            "See https://github.com/ccsb-scripps/AutoDock-Vina/releases"
        )

    destination = f"/tmp/vina-{VINA_VERSION}"
    if not os.path.exists(destination):
        url = (
            f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/"
            f"v{VINA_VERSION}/vina_{VINA_VERSION}_{asset}"
        )
        print(f"[vina] downloading {url}")
        with urllib.request.urlopen(url, timeout=180) as response:
            payload = response.read()
        # Write through a temp name so an interrupted download cannot leave a
        # truncated binary behind at the real path.
        with open(destination + ".part", "wb") as handle:
            handle.write(payload)
        os.replace(destination + ".part", destination)
        print(f"[vina] wrote {len(payload):,} bytes -> {destination}")

    os.chmod(destination, os.stat(destination).st_mode | stat.S_IEXEC | stat.S_IXGRP)
    check = subprocess.run(
        [destination, "--version"], capture_output=True, text=True, timeout=60
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"{destination} would not run: {(check.stderr or check.stdout).strip()[:300]}"
        )
    print(f"[vina] {check.stdout.strip()} ({machine})")
    return destination


# Must be set before the app is imported: `core.config.Settings` reads the
# environment once, at import time.
os.environ["SCREENING_VINA_PATH"] = ensure_vina()

# Where run output lands. Cluster-local storage does not survive the session, so
# production should point this at a Unity Catalog volume; /tmp is the default so a
# smoke run works before a volume exists.
dbutils.widgets.text("output_root", "/tmp/screensuite_runs")
output_root = dbutils.widgets.get("output_root")

# Set before importing the app: `core.config.Settings` reads the environment once,
# at import time.
os.environ["SCREENING_OUTPUT_ROOT"] = output_root
os.environ.setdefault("SCREENING_MAX_COMBINATIONS", "50")

os.makedirs(output_root, exist_ok=True)
print(f"[screensuite] output_root={output_root}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Input
# MAGIC
# MAGIC Accepts either shape on the `input`/`query` parameter:
# MAGIC
# MAGIC * JSON — `{"proteins": ["JAK2"], "drugs": ["ruxolitinib", "chlorthalidone"]}`,
# MAGIC   where an entry may also be an object carrying a known identifier:
# MAGIC   `{"name": "JAK2", "identifier": "6VGL", "source": "pdb"}`
# MAGIC * Shorthand — `"JAK2: ruxolitinib, chlorthalidone"`
# MAGIC
# MAGIC An empty parameter runs a known-good pair so a manually started run still
# MAGIC proves the stack end to end.

# COMMAND ----------
DEFAULT_INPUT = {
    "proteins": [{"name": "JAK2", "identifier": "6VGL", "source": "pdb"}],
    "drugs": [{"name": "Ruxolitinib", "identifier": "25126798", "source": "pubchem"}],
}


def parse_input(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        print("[screensuite] no input given — using the default JAK2/ruxolitinib pair")
        return DEFAULT_INPUT
    if raw.startswith("{"):
        return json.loads(raw)
    if ":" in raw:
        protein_part, drug_part = raw.split(":", 1)
        return {
            "proteins": [{"name": protein_part.strip()}],
            "drugs": [{"name": d.strip()} for d in drug_part.split(",") if d.strip()],
        }
    # A bare protein name with no drugs is not screenable — say so rather than
    # silently docking nothing.
    raise ValueError(
        f"input {raw!r} names no drugs. Use 'PROTEIN: drug1, drug2' or JSON."
    )


parsed = parse_input(p["input"])
print(f"[screensuite] input: {json.dumps(parsed)[:400]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Environment check
# MAGIC
# MAGIC Reports what this cluster can actually do before spending compute, so a failed
# MAGIC run says *why* instead of only *that* it failed.

# COMMAND ----------
from DRP_Main.app.modules.screening import docking_service

tools = docking_service.tool_availability()
for name, state in tools.items():
    mark = "OK  " if state["available"] else "MISS"
    detail = state.get("path") or state.get("reason") or state.get("error", "")
    print(f"  {mark} {name:<20} {str(detail)[:90]}")

print(f"\n  docking available:    {docking_service.docking_available()}")
print(f"  conversion available: {docking_service.conversion_available()}")

if not docking_service.docking_available():
    raise RuntimeError(
        "AutoDock Vina is not available on this cluster — neither the executable nor "
        "the Python bindings. Check the %pip install cell above."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Resolve names to structures
# MAGIC
# MAGIC Skipped for any entry that already carries an identifier. An ambiguous name does
# MAGIC **not** pause here — a batch job has nobody to ask — so it is reported and
# MAGIC excluded; interactive confirmation belongs to the agent path.

# COMMAND ----------
from DRP_Main.app.modules.screening import resolution_service
from DRP_Main.app.modules.screening.schemas import DrugQuery, ProteinQuery


def as_queries(entries, model):
    out = []
    for entry in entries or []:
        out.append(model(name=entry) if isinstance(entry, str) else model(**entry))
    return out


resolution = resolution_service.resolve(
    as_queries(parsed.get("proteins"), ProteinQuery),
    as_queries(parsed.get("drugs"), DrugQuery),
    allow_partial=True,
)
print(f"[resolution] stage={resolution.stage}")
print(f"[resolution] proteins={[p_.model_dump() for p_ in resolution.proteins]}")
print(f"[resolution] drugs={[d.model_dump() for d in resolution.drugs]}")
if resolution.ambiguous:
    for entity in resolution.ambiguous:
        print(
            f"[resolution] AMBIGUOUS {entity.name}: "
            f"{[c.identifier for c in entity.candidates]} — excluded from this batch"
        )
if resolution.unresolved:
    for entity in resolution.unresolved:
        print(f"[resolution] UNRESOLVED {entity.name}: {entity.reason}")

def resolution_failure_rows() -> list:
    """
    Entities that never reached docking: ambiguous names and names that matched
    nothing. Written to Gold so the API can tell the user *why* nothing came
    back, including the shortlist to choose from.

    Every value is a string, never None: Spark cannot infer a column type when a
    column is None in every row, and a batch of only-protein failures would
    otherwise leave `ligand` entirely null and crash the write.
    """
    out = []
    for entity in resolution.ambiguous:
        shortlist = ", ".join(
            c.identifier + (f" ({c.title})" if c.title else "") for c in entity.candidates
        )
        out.append({
            "run_id": "",
            "protein": entity.name if entity.kind == "protein" else "",
            "ligand": entity.name if entity.kind == "drug" else "",
            "status": (
                f"Failed[resolution]: '{entity.name}' matches several structures — "
                f"resend with one identifier. Candidates: {shortlist}"
            )[:900],
        })
    for entity in resolution.unresolved:
        out.append({
            "run_id": "",
            "protein": entity.name if entity.kind == "protein" else "",
            "ligand": entity.name if entity.kind == "drug" else "",
            "status": f"Failed[resolution]: '{entity.name}' — {entity.reason}"[:900],
        })
    return out


if not resolution.proteins or not resolution.drugs:
    # A batch job has nobody to ask, so this is not raised as an error: raising
    # fails the run and the reason is lost behind "Workload failed". Recording
    # it lets the API surface the shortlist to the user instead.
    ensure_medallion(catalog, schema)
    nothing_rows = resolution_failure_rows() or [{
        "run_id": "",
        "protein": "",
        "ligand": "",
        "status": f"Failed[resolution]: nothing screenable. {resolution.message}"[:900],
    }]
    write_gold(catalog, schema, "docking_results", nothing_rows, drp_job_id)
    summary = f"screensuite: nothing_screenable rows={len(nothing_rows)} drp_job_id={drp_job_id}"
    print(summary)
    dbutils.notebook.exit(summary)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dock
# MAGIC
# MAGIC Every protein against every drug, each protein ranked independently.

# COMMAND ----------
from DRP_Main.app.modules.screening import pipeline_service

step("AutoDock Vina docking + ProLIF interaction analysis")
result = pipeline_service.screen_batch(
    resolution.proteins,
    resolution.drugs,
    run_id=f"job_{drp_job_id}" if drp_job_id else None,
)

print(f"[screensuite] run_id={result.run_id} status={result.status} "
      f"combinations={result.combinations} time={result.time_taken}")
print(f"[screensuite] {result.recommendation}")

for protein in result.proteins:
    print(f"\n  {protein.protein_name} ({protein.identifier}) — {protein.status}")
    for record in protein.top_affinity_records:
        print(f"      {record['ligand']:<20} {record['Affinity_kcal_per_mol']:>8} kcal/mol "
              f"(mode {record['Mode']})")
    for report in protein.interaction_reports:
        if report.status == "success":
            print(f"      interactions for {report.ligand}: {report.interaction_count} "
                  f"across {len(report.binding_site_residues)} residue(s)")
            for kind, entries in report.interactions_by_type.items():
                residues = sorted({e.residue for e in entries})
                print(f"        {kind:<16} {residues}")
        else:
            print(f"      interactions for {report.ligand}: {report.status} — {report.error}")

for failure in result.failures:
    print(f"  FAILED {failure.kind} {failure.name} [{failure.stage}]: {failure.error[:160]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Persist to Gold
# MAGIC
# MAGIC One row per ranked pose, tagged with the DRP job id the API reads back on.

# COMMAND ----------
ensure_medallion(catalog, schema)
mlflow_log(
    "screensuite",
    {"proteins": len(resolution.proteins), "drugs": len(resolution.drugs)},
    {"combinations": result.combinations},
)

rows = []
for protein in result.proteins:
    interactions_by_ligand = {
        report.ligand: {
            kind: sorted({e.residue for e in entries})
            for kind, entries in (report.interactions_by_type or {}).items()
        }
        for report in protein.interaction_reports
        if report.status == "success"
    }
    for record in protein.top_affinity_records:
        rows.append(
            {
                "run_id": result.run_id,
                "protein": protein.protein_name,
                "protein_identifier": protein.identifier,
                "protein_source": protein.source,
                "ligand": record.get("ligand"),
                "mode": record.get("Mode"),
                "affinity_kcal_mol": record.get("Affinity_kcal_per_mol"),
                "pose_file": record.get("out_pdbqt_file"),
                "receptor_pdb": protein.receptor_pdb,
                "interactions": interactions_by_ligand.get(record.get("ligand"), {}),
                "status": protein.status,
            }
        )

# Failures are recorded too — a run that produced nothing must still be explainable
# from the table alone, not only from the driver log. Values are "" rather than None:
# a run with only failures would otherwise leave a column null in every row, and
# Spark cannot infer a type for it, so the write itself would crash.
for failure in result.failures:
    rows.append(
        {
            "run_id": result.run_id,
            "protein": failure.name if failure.kind == "protein" else "",
            "ligand": failure.name if failure.kind == "drug" else "",
            "status": f"Failed[{failure.stage}]: {failure.error[:300]}",
        }
    )

# Names excluded before docking (ambiguous, or matched nothing) in a run that still
# docked the rest — without these they would vanish without a trace.
for row in resolution_failure_rows():
    row["run_id"] = result.run_id
    rows.append(row)

write_gold(catalog, schema, "docking_results", rows, drp_job_id)

# COMMAND ----------
summary = (
    f"screensuite: {result.status} run_id={result.run_id} "
    f"proteins={len(result.proteins)} rows={len(rows)} "
    f"failures={len(result.failures)} drp_job_id={drp_job_id}"
)
print(summary)
dbutils.notebook.exit(summary)
