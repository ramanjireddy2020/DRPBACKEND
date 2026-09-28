"""
CurateX Tool 1 (§5) — target resolution and profile builder.

Steps 1-8: resolve inputs → pull known ligands from all six §2.1 sources →
extract §2.2 fields → dedup and merge on InChIKey → DailyMed enrichment on the
deduped set only → compute the per-criterion baselines the scorer normalizes
against → set up the exclusion flag → emit an editable profile object.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langfuse.decorators import langfuse_context, observe

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation import dailymed_service, ligand_sources
from DRP_Main.app.modules.drug_curation.criteria_config import (
    ALL_ROWS,
    CATEGORICAL,
    DAILYMED_CRITERIA,
    EXCLUSION,
    METADATA,
    NUMERIC,
    default_profile_weights,
)
from DRP_Main.app.modules.drug_curation.identifier_service import (
    ResolvedDisease,
    ResolvedTarget,
    ResolutionError,
    resolve_disease,
    resolve_structure,
    resolve_target,
)
from DRP_Main.app.modules.drug_curation.scoring_service import (
    CriterionBaseline,
    apply_value_overrides,
    build_baselines,
)

logger = get_logger(__name__)

# Fields where ChEMBL and Open Targets both speak. §5 step 4: Open Targets wins,
# the ChEMBL value is kept as a secondary reference rather than discarded.
OPEN_TARGETS_AUTHORITATIVE = ("max_clinical_phase", "withdrawn_status", "black_box_warning")

# Sources that contribute structure/bioactivity but never criteria fields.
DISCOVERY_ONLY_SOURCES = ("bindingdb", "pubchem_bioassay", "iuphar", "dgidb")


@dataclass
class Ligand:
    """One deduped reference ligand, merged across sources."""

    inchikey: str
    name: str
    smiles: str = ""
    chembl_id: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)
    field_sources: Dict[str, str] = field(default_factory=dict)
    secondary_values: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, str]] = field(default_factory=list)
    known_indications: List[Dict[str, Any]] = field(default_factory=list)
    bioactivity: List[Dict[str, Any]] = field(default_factory=list)
    label: Optional[dailymed_service.LabelExtract] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "inchikey": self.inchikey,
            "name": self.name,
            "smiles": self.smiles,
            "chemblId": self.chembl_id,
            "fields": dict(self.fields),
            "fieldSources": dict(self.field_sources),
            "secondaryValues": dict(self.secondary_values),
            "provenance": list(self.provenance),
            "knownIndications": list(self.known_indications),
            "bioactivity": list(self.bioactivity),
            "label": self.label.as_dict() if self.label else None,
        }


@dataclass
class DrugProfile:
    """The §5 step 8 output object — shown to the user for editing before Tool 2."""

    target: Optional[ResolvedTarget]
    disease: Optional[ResolvedDisease]
    ligands: List[Ligand]
    baselines: Dict[str, CriterionBaseline]
    weights: Dict[str, float]
    exclusion: Dict[str, Any]
    source_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)

    # ── weighting ────────────────────────────────────────────────────────────
    def hidden_keys(self) -> Dict[str, str]:
        """
        Criteria that cannot affect the ranking, and why.

        Two cases. A criterion every ligand shares (324/324 not withdrawn) adds
        the same term to every composite — it cannot reorder anything, but it
        still consumed 10 of the 100 available weight, so it diluted every
        criterion that could. A criterion nothing has data for is worse: it
        contributes nothing at all while still holding weight.

        Returned rather than applied in place so the row survives in the payload,
        flagged, instead of vanishing from an array the UI indexes into.
        """
        hidden: Dict[str, str] = {}
        for key, baseline in self.baselines.items():
            if baseline.count == 0:
                hidden[key] = "No ligand had a value for this criterion."
            elif not baseline.discriminating:
                hidden[key] = (
                    "Every ligand with data shares the same value, so this "
                    "cannot separate one candidate from another."
                )
        return hidden

    def effective_weights(self) -> Dict[str, float]:
        """
        The weights actually used for scoring: hidden criteria dropped, the rest
        renormalized to total exactly 100 as whole numbers.

        Largest-remainder apportionment, so the total is 100 on the nose rather
        than 99 or 101 after rounding — the profile screen states the weights sum
        to 100, and it should be true of the numbers that did the scoring.
        """
        hidden = self.hidden_keys()
        live = {
            key: float(weight)
            for key, weight in self.weights.items()
            if key not in hidden and float(weight) > 0
        }
        total = sum(live.values())
        if not live or total <= 0:
            return {key: float(w) for key, w in self.weights.items()}

        scaled = {key: weight * 100.0 / total for key, weight in live.items()}
        floored = {key: int(value) for key, value in scaled.items()}
        shortfall = 100 - sum(floored.values())
        # Hand the leftover points to the largest fractional parts first.
        for key, _ in sorted(
            scaled.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True
        )[: max(0, shortfall)]:
            floored[key] += 1
        return {key: float(value) for key, value in floored.items()}

    # ── views ────────────────────────────────────────────────────────────────
    def criteria_rows(self) -> List[Dict[str, Any]]:
        """19 weighted rows + the exclusion row + the metadata row."""
        hidden = self.hidden_keys()
        effective = self.effective_weights()
        rows: List[Dict[str, Any]] = []
        for criterion in ALL_ROWS:
            baseline = self.baselines.get(criterion.key)
            row: Dict[str, Any] = {
                "key": criterion.key,
                "number": criterion.number,
                "label": criterion.label,
                "category": criterion.category,
                "kind": criterion.kind,
                "sources": list(criterion.sources),
                "editable": criterion.editable,
                "note": criterion.note,
                # The renormalized weight — what actually scored this run. The
                # profile screen offers these for editing, so showing the
                # pre-normalization number would invite edits against a scale
                # the backend no longer uses.
                "weight": float(
                    effective.get(
                        criterion.key,
                        self.weights.get(criterion.key, criterion.default_weight),
                    )
                ),
                "defaultWeight": criterion.default_weight,
                # Hidden rows stay in the array — the UI indexes into it — but
                # carry zero weight and say why they were set aside.
                "hidden": criterion.key in hidden,
                "hiddenReason": hidden.get(criterion.key),
                "discriminating": baseline.discriminating if baseline else None,
            }
            if criterion.key in hidden:
                row["weight"] = 0.0
                row["editable"] = False
            if criterion.kind == EXCLUSION:
                row.update(
                    {
                        "weight": None,
                        "defaultWeight": None,
                        "value": self.exclusion.get("summary", ""),
                        "excludedDrugCount": self.exclusion.get("count", 0),
                    }
                )
            elif criterion.kind == METADATA:
                years = [
                    ligand.fields.get("withdrawal_year")
                    for ligand in self.ligands
                    if ligand.fields.get("withdrawal_year")
                ]
                row.update({"weight": None, "defaultWeight": None, "value": sorted(set(years))})
            elif criterion.kind == NUMERIC and baseline:
                row.update(
                    {
                        "median": baseline.median_value,
                        "min": baseline.minimum,
                        "max": baseline.maximum,
                        # The representative band, and what scoring normalizes
                        # against. Prefer it over min-max when displaying the
                        # ideal range: min-max is one outlier wide.
                        "p25": baseline.p25,
                        "p75": baseline.p75,
                        "unit": criterion.unit,
                    }
                )
            elif criterion.kind == CATEGORICAL and baseline:
                row.update({"distribution": baseline.distribution})

            if baseline:
                # Coverage on every criterion, not just the DailyMed-derived
                # ones. ChEMBL's `oral` flag is 0 when a drug is simply not
                # annotated, so the profile reported "no: 290, yes: 10" for a set
                # of mostly-oral drugs with nothing to indicate that 290 meant
                # "unknown". A criterion with no coverage figure reads as
                # complete, which for this one was badly wrong.
                row["coverage"] = baseline.coverage
                row["coverageLabel"] = (
                    f"{baseline.count}/{baseline.total} ligands" if baseline.total else "no ligands"
                )
            rows.append(row)
        return rows

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.as_dict() if self.target else None,
            "disease": self.disease.as_dict() if self.disease else None,
            "criteria": self.criteria_rows(),
            # `weights` is the renormalized table that scored this run, so the
            # profile and the scoring agree. The pre-normalization numbers are
            # kept alongside rather than dropped, for anything reconciling
            # against the defaults.
            "weights": self.effective_weights(),
            "rawWeights": dict(self.weights),
            "hiddenCriteria": self.hidden_keys(),
            "exclusion": dict(self.exclusion),
            "ligandCount": len(self.ligands),
            "ligands": [ligand.as_dict() for ligand in self.ligands],
            "sourceCounts": dict(self.source_counts),
            "coverage": {
                key: self.baselines[key].as_dict()
                for key in DAILYMED_CRITERIA
                if key in self.baselines
            },
            "warnings": list(self.warnings),
        }


# ── Step 2: known ligand retrieval ────────────────────────────────────────────
@observe(name="curatex_ligand_retrieval")
def retrieve_known_ligands(target: ResolvedTarget) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    §5 step 2 — query all six §2.1 sources in parallel.

    DailyMed is deliberately absent: it is a per-drug label lookup with no target
    index, so it only makes sense after dedup (step 5).
    """
    max_ligands = int(getattr(settings, "CURATEX_MAX_KNOWN_LIGANDS", 300) or 300)

    def _chembl_ligands() -> List[Dict[str, Any]]:
        """Molecule properties plus the potency that made each one a ligand."""
        potency = ligand_sources.chembl_ligand_potency(
            target.chembl_target_id, limit=max_ligands
        )
        records = ligand_sources.chembl_molecules(list(potency.keys()))
        for record in records:
            measure = potency.get(record.get("chembl_id"))
            if measure:
                record["bioactivity"] = measure
        return records

    jobs = {
        "chembl": _chembl_ligands,
        "open_targets": lambda: ligand_sources.open_targets_known_drugs(target.ensembl_gene_id),
        "bindingdb": lambda: ligand_sources.bindingdb_ligands(target.uniprot_accession),
        "pubchem_bioassay": lambda: ligand_sources.pubchem_bioassay_ligands(
            target.uniprot_accession
        ),
        "iuphar": lambda: ligand_sources.iuphar_ligands(target.gene_symbol),
        "dgidb": lambda: ligand_sources.dgidb_ligands(target.gene_symbol),
    }

    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {name: executor.submit(job) for name, job in jobs.items()}
        for name, future in futures.items():
            try:
                rows = future.result() or []
            except Exception as exc:
                logger.warning("Ligand source %s failed: %s", name, exc)
                rows = []
            counts[name] = len(rows)
            records.extend(rows)

    langfuse_context.update_current_observation(output=counts)
    return records, counts


# ── Steps 3-4: field extraction, dedup and merge ──────────────────────────────
def _merge_key(record: Dict[str, Any], resolve_missing_structures: bool) -> str:
    """
    The §5 step 4 primary merge key.

    ChEMBL and PubChem hand back an InChIKey directly. Anything else (a name-only
    DGIdb/IUPHAR row, a BindingDB SMILES) goes through §3.3 first — without a
    common key those records would each become their own "ligand" and inflate the
    baseline ranges with duplicates.
    """
    inchikey = (record.get("inchikey") or "").strip()
    if inchikey:
        return inchikey
    if not resolve_missing_structures:
        return ""
    seed = record.get("smiles") or record.get("name") or ""
    if not seed:
        return ""
    resolved = resolve_structure(seed)
    if resolved.inchikey:
        record["inchikey"] = resolved.inchikey
        if resolved.smiles and not record.get("smiles"):
            record["smiles"] = resolved.smiles
        return resolved.inchikey
    return ""


def merge_records(
    records: Sequence[Dict[str, Any]], resolve_missing_structures: bool = True
) -> List[Ligand]:
    """
    §5 steps 3-4 — collapse source records into one ligand per InChIKey.

    Merge precedence for the fields ChEMBL and Open Targets both report is Open
    Targets, with the ChEMBL value retained under `secondary_values`. Records
    from the discovery-only sources never write criteria fields; they contribute
    structure, bioactivity and provenance.
    """
    by_key: Dict[str, Ligand] = {}
    # Ligands with no resolvable structure still matter (a name-only DGIdb hit is
    # real evidence), so they fall back to a name key rather than being dropped.
    for record in records:
        key = _merge_key(record, resolve_missing_structures)
        if not key:
            key = f"name:{(record.get('name') or '').strip().lower()}"
            if key == "name:":
                continue

        ligand = by_key.get(key)
        if ligand is None:
            ligand = Ligand(
                inchikey=record.get("inchikey", "") or "",
                name=record.get("name", "") or key,
                smiles=record.get("smiles", "") or "",
            )
            by_key[key] = ligand

        source = record.get("source", "")
        ligand.provenance.append({"source": source, "recordId": str(record.get("record_id", ""))})
        if not ligand.smiles and record.get("smiles"):
            ligand.smiles = record["smiles"]
        if not ligand.inchikey and record.get("inchikey"):
            ligand.inchikey = record["inchikey"]
        if record.get("chembl_id") and not ligand.chembl_id:
            ligand.chembl_id = record["chembl_id"]
        # ChEMBL/Open Targets carry the curated preferred name; prefer it over a
        # BindingDB monomer id or a PubChem title.
        if source in ("chembl", "open_targets") and record.get("name"):
            ligand.name = record["name"]
        if record.get("bioactivity"):
            ligand.bioactivity.append(record["bioactivity"])
        for indication in record.get("known_indications") or []:
            if indication not in ligand.known_indications:
                ligand.known_indications.append(indication)

        if source in DISCOVERY_ONLY_SOURCES:
            continue

        for key_name, value in (record.get("fields") or {}).items():
            if value is None or value == "":
                continue
            existing_source = ligand.field_sources.get(key_name)
            if existing_source is None:
                ligand.fields[key_name] = value
                ligand.field_sources[key_name] = source
                continue
            if existing_source == source:
                # Duplicate records from one source for one InChIKey (salt forms,
                # repeat assay entries) collapse to one. The spec says keep the
                # most recent; ChEMBL's molecule endpoint carries no record
                # timestamp, so "most recent" is unavailable and the first record
                # returned wins — a stable choice rather than an arbitrary one.
                continue
            if source == "open_targets" and key_name in OPEN_TARGETS_AUTHORITATIVE:
                ligand.secondary_values[key_name] = ligand.fields[key_name]
                ligand.fields[key_name] = value
                ligand.field_sources[key_name] = source
            elif existing_source == "open_targets" and key_name in OPEN_TARGETS_AUTHORITATIVE:
                ligand.secondary_values[key_name] = value
            elif key_name not in ligand.fields:
                ligand.fields[key_name] = value
                ligand.field_sources[key_name] = source

    return list(by_key.values())


def enrich_open_targets_drug_fields(ligands: Sequence[Ligand], max_workers: int = 6) -> None:
    """
    Fill the Open Targets per-drug fields (AE LLR, withdrawal, black box).

    `knownDrugs` gives the drug list and its indications but not the safety
    block, so criterion 14 — 7 points — is only available from `drug(chemblId:)`.
    """
    targets = [ligand for ligand in ligands if ligand.chembl_id]
    if not targets:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        details = list(
            executor.map(ligand_sources.open_targets_drug, [l.chembl_id for l in targets])
        )
    for ligand, detail in zip(targets, details):
        if not detail:
            continue
        for key, value in (detail.get("fields") or {}).items():
            if value is None or value == "":
                continue
            current_source = ligand.field_sources.get(key)
            if current_source == "open_targets" and key in ligand.fields:
                continue
            if current_source and ligand.fields.get(key) != value:
                # Open Targets wins on conflict (§5 step 4); the displaced value
                # stays visible rather than being dropped.
                ligand.secondary_values[key] = ligand.fields.get(key)
            ligand.fields[key] = value
            ligand.field_sources[key] = "open_targets"
        for indication in detail.get("known_indications") or []:
            if indication not in ligand.known_indications:
                ligand.known_indications.append(indication)


def backfill_chembl_warnings(ligands: Sequence[Ligand]) -> None:
    """Criterion 11b + the withdrawal-year metadata, for ligands flagged withdrawn."""
    withdrawn = [
        ligand
        for ligand in ligands
        if ligand.fields.get("withdrawn_status") == "withdrawn" and ligand.chembl_id
    ]
    needing = [
        ligand
        for ligand in withdrawn
        if ligand.fields.get("withdrawal_reason") in (None, "", "unknown", "not_withdrawn")
        or not ligand.fields.get("withdrawal_year")
    ]
    if not needing:
        return
    warnings = ligand_sources.chembl_drug_warnings([l.chembl_id for l in needing])
    for ligand in needing:
        extra = warnings.get(ligand.chembl_id) or {}
        for key, value in extra.items():
            if value:
                ligand.fields[key] = value
                ligand.field_sources[key] = "chembl"


# ── Step 5: DailyMed enrichment ───────────────────────────────────────────────
@observe(name="curatex_dailymed_enrichment")
def enrich_with_dailymed(ligands: Sequence[Ligand], max_workers: int = 6) -> Dict[str, int]:
    """
    §5 step 5 — post-dedup only, matched by drug name (DailyMed has no InChIKey).

    Ligands with no matching SPL keep `label.found == False`, which is what the
    coverage indicator counts. Note this runs *after* dedup deliberately: running
    it per source record would issue the same label lookup several times for one
    drug and still produce one merged value.
    """
    named = [ligand for ligand in ligands if ligand.name and not ligand.name.startswith("CHEMBL")]
    if not named:
        return {"attempted": 0, "found": 0, "no_label": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        extracts = list(executor.map(dailymed_service.enrich, [l.name for l in named]))

    found = 0
    for ligand, extract in zip(named, extracts):
        ligand.label = extract
        if not extract.found:
            continue
        found += 1
        for key, value in extract.fields.items():
            # The SPL is the third black-box cross-check, not the authority —
            # §5 step 4 gives that to Open Targets.
            if key == "black_box_warning" and ligand.field_sources.get(key) == "open_targets":
                ligand.secondary_values.setdefault("black_box_warning_dailymed", value)
                continue
            ligand.fields[key] = value
            ligand.field_sources[key] = "dailymed"
    return {"attempted": len(named), "found": found, "no_label": len(named) - found}


# ── Step 7: exclusion flag ────────────────────────────────────────────────────
@observe(name="curatex_exclusion_setup")
def build_exclusion(disease: Optional[ResolvedDisease]) -> Dict[str, Any]:
    """
    §5 step 7 — the hard filter passed to Tool 2, never scored.

    Matching extends over the EFO descendant set, not the resolved ID alone: a
    drug already approved for a subtype of the disease is not a repurposing
    candidate for the parent indication, and exact-ID matching would surface it
    as a false candidate.
    """
    if disease is None or not disease.efo_id:
        return {
            "active": False,
            "count": 0,
            "efoIds": [],
            "drugs": {},
            "summary": "No disease resolved — the repurposing exclusion filter is inactive.",
        }

    family = disease.efo_family
    drugs = ligand_sources.open_targets_disease_known_drugs(family)
    return {
        "active": True,
        "count": len(drugs),
        "efoId": disease.efo_id,
        "efoIds": family,
        "descendantCount": len(family) - 1,
        "drugs": drugs,
        "summary": (
            f"{len(drugs)} drug(s) already indicated for {disease.efo_label or disease.query} "
            f"or one of its {max(0, len(family) - 1)} descendant term(s) will be removed "
            "before scoring."
        ),
    }


# ── Steps 1-8: the whole tool ─────────────────────────────────────────────────
@observe(name="curatex_tool1_profile")
def build_profile(
    target_name: Optional[str] = None,
    disease_name: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    skip_dailymed: bool = False,
    values: Optional[Dict[str, Any]] = None,
) -> DrugProfile:
    """
    Run Tool 1 end to end for a single target.

    Raises `ResolutionError` when neither a target nor a disease resolves; a
    disease that resolves without a target is handled by the caller (§5 step 1
    surfaces the upstream target shortlist for confirmation rather than picking
    one, since target choice changes every downstream profile and score).
    """
    warnings: List[str] = []

    resolved_disease: Optional[ResolvedDisease] = None
    if disease_name:
        try:
            resolved_disease = resolve_disease(disease_name)
            if not resolved_disease.efo_id:
                warnings.append(
                    f"'{disease_name}' resolved to MeSH {resolved_disease.mesh_id} but no EFO "
                    "term; the repurposing exclusion filter cannot run."
                )
        except ResolutionError as exc:
            warnings.append(str(exc))

    if not target_name:
        raise ResolutionError(
            "A target is required to build a drug profile. Confirm one of the "
            "candidate targets from target identification first."
        )

    resolved_target = resolve_target(target_name)
    if not resolved_target.chembl_target_id:
        warnings.append(
            f"UniProt {resolved_target.uniprot_accession} has no ChEMBL cross-reference; "
            "ChEMBL ligands and their physicochemical fields are unavailable for this target."
        )

    records, source_counts = retrieve_known_ligands(resolved_target)
    if not records:
        warnings.append(
            "No known ligands were found for this target in any of the six discovery sources; "
            "the profile has no baseline to normalize candidates against."
        )

    ligands = merge_records(records)
    enrich_open_targets_drug_fields(ligands)
    backfill_chembl_warnings(ligands)

    # Warnings surface next to the profile in the UI, so they name the criteria
    # rather than their numbers: "criteria 15-19" is internal numbering that means
    # nothing to a researcher reading it under the results.
    _LABEL_CRITERIA = "dosage form, dosing frequency, absorption, half-life and metabolism"

    if skip_dailymed:
        warnings.append(
            f"FDA label data was not retrieved, so {_LABEL_CRITERIA} carry no data."
        )
    else:
        stats = enrich_with_dailymed(ligands)
        if stats["no_label"]:
            warnings.append(
                f"{stats['no_label']} of {stats['attempted']} ligand(s) had no FDA label on "
                f"DailyMed, so {_LABEL_CRITERIA} are scored on the remainder only."
            )

    baselines = build_baselines([ligand.fields for ligand in ligands])
    # The researcher's edited criterion values override what the ligand set
    # taught us; without this the edits were collected and discarded.
    warnings.extend(apply_value_overrides(baselines, values))
    exclusion = build_exclusion(resolved_disease)

    active_weights = default_profile_weights()
    if weights:
        active_weights.update({k: float(v) for k, v in weights.items() if k in active_weights})

    langfuse_context.update_current_observation(
        output={"ligands": len(ligands), "excluded": exclusion.get("count", 0)}
    )
    return DrugProfile(
        target=resolved_target,
        disease=resolved_disease,
        ligands=ligands,
        baselines=baselines,
        weights=active_weights,
        exclusion=exclusion,
        source_counts=source_counts,
        warnings=warnings,
    )


def build_profiles(
    target_names: Sequence[str],
    disease_name: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    skip_dailymed: bool = False,
    values: Optional[Dict[str, Any]] = None,
) -> List[DrugProfile]:
    """§5 step 1 — several targets means one profile object per target, in parallel."""
    names = [n for n in (t.strip() for t in target_names) if n]
    if not names:
        raise ResolutionError("At least one target is required")
    if len(names) == 1:
        return [build_profile(names[0], disease_name, weights, skip_dailymed, values)]

    with ThreadPoolExecutor(max_workers=min(4, len(names))) as executor:
        futures = [
            executor.submit(build_profile, name, disease_name, weights, skip_dailymed, values)
            for name in names
        ]
        profiles: List[DrugProfile] = []
        for name, future in zip(names, futures):
            try:
                profiles.append(future.result())
            except Exception as exc:
                logger.error("Profile build failed for %s: %s", name, exc)
    if not profiles:
        raise ResolutionError("No profile could be built for any of the supplied targets")
    return profiles
