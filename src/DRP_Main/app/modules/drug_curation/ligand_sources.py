"""
CurateX §2 — the source clients.

Two distinct lists at two stages:

* §2.1 **known ligand discovery** (Tool 1 step 2) — six sources, wide net. These
  only define the reference ligand set; field cleanliness matters less than
  coverage, so a name-only hit is still useful (it is resolved to a structure by
  §3.3 before merging).
* §2.2 **criteria extraction** (Tool 1 steps 3-6, reused by Tool 2) — ChEMBL and
  Open Targets only, plus DailyMed in `dailymed_service`. DrugBank is excluded
  throughout: it requires a commercial licence.

Every client returns a list of plain dicts and swallows its own failures. A dead
source costs coverage, not the run.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.http_client import get_json, post_graphql

logger = get_logger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
OPEN_TARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
BINDINGDB_LIGANDS = "https://bindingdb.org/axis2/services/BDBService/getLigandsByUniprots"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
IUPHAR_BASE = "https://www.guidetopharmacology.org/services"
DGIDB_GRAPHQL = "https://dgidb.org/api/graphql"

# Potency measures worth treating as evidence of a real ligand relationship.
POTENCY_TYPES = ("IC50", "Ki", "EC50", "Kd")


# ── ChEMBL ────────────────────────────────────────────────────────────────────
def chembl_ligand_potency(chembl_target_id: str, limit: int = 300) -> Dict[str, Dict[str, Any]]:
    """
    §2.1 — the strongest recorded potency per molecule against the target.

    No criterion scores potency, so this is provenance rather than signal: it is
    what shows *why* a molecule is in the reference ligand set at all, next to
    the BindingDB affinities that already ride along on their own records.
    """
    if not chembl_target_id:
        return {}
    best: Dict[str, Dict[str, Any]] = {}
    offset, page = 0, 200
    while len(best) < limit:
        payload = get_json(
            f"{CHEMBL_BASE}/activity",
            {
                "target_chembl_id": chembl_target_id,
                "standard_type__in": ",".join(POTENCY_TYPES),
                "format": "json",
                "limit": page,
                "offset": offset,
            },
        )
        rows = (payload or {}).get("activities") or []
        if not rows:
            break
        for row in rows:
            molecule_id = row.get("molecule_chembl_id")
            if not molecule_id:
                continue
            value = _as_float(row.get("standard_value"))
            units = row.get("standard_units")
            entry = best.get(molecule_id)
            # Lower is more potent, but only nM values are comparable; a record
            # with no usable value still registers the molecule as a ligand.
            comparable = value if units == "nM" else None
            if entry is None:
                best[molecule_id] = {
                    "type": row.get("standard_type", ""),
                    "value_nm": comparable,
                    "source": "chembl",
                }
            elif comparable is not None and (
                entry.get("value_nm") is None or comparable < entry["value_nm"]
            ):
                entry.update({"type": row.get("standard_type", ""), "value_nm": comparable})
        if len(rows) < page:
            break
        offset += page
    return dict(list(best.items())[:limit])


def chembl_ligand_ids(chembl_target_id: str, limit: int = 300) -> List[str]:
    """The ligand ID list alone, for callers that do not need the potency values."""
    return list(chembl_ligand_potency(chembl_target_id, limit).keys())


def chembl_molecules(molecule_ids: List[str]) -> List[Dict[str, Any]]:
    """§2.2 — batch-fetch the ChEMBL criteria fields for a set of molecules."""
    records: List[Dict[str, Any]] = []
    batch = 20  # the `__in` filter is a URL parameter; keep it under the URI limit
    for start in range(0, len(molecule_ids), batch):
        chunk = molecule_ids[start : start + batch]
        payload = get_json(
            f"{CHEMBL_BASE}/molecule",
            {"molecule_chembl_id__in": ",".join(chunk), "format": "json", "limit": batch},
        )
        for molecule in (payload or {}).get("molecules") or []:
            parsed = parse_chembl_molecule(molecule)
            if parsed:
                records.append(parsed)
    return records


def chembl_candidate_universe(limit: int = 1000, min_phase: int = 1) -> List[Dict[str, Any]]:
    """
    §6 step 1 — the broader candidate pool: molecules with any populated max phase.

    Deliberately *not* approved-only. An investigational compound already carries
    a development data package, which is the head start repurposing exploits;
    maturity is scored by the Regulatory status category rather than filtered on
    twice.
    """
    records: List[Dict[str, Any]] = []
    offset, page = 0, 200
    while len(records) < limit:
        payload = get_json(
            f"{CHEMBL_BASE}/molecule",
            {
                "max_phase__gte": min_phase,
                # The pool is a slice, not the whole of ChEMBL. Ordering by max
                # phase makes that slice the most clinically mature compounds
                # rather than whichever molecules happen to hold the lowest
                # ChEMBL IDs, which is an alphabetical accident, not a criterion.
                "order_by": "-max_phase",
                "format": "json",
                "limit": min(page, limit - len(records)),
                "offset": offset,
            },
        )
        molecules = (payload or {}).get("molecules") or []
        if not molecules:
            break
        for molecule in molecules:
            parsed = parse_chembl_molecule(molecule)
            if parsed:
                records.append(parsed)
        if len(molecules) < page:
            break
        offset += page
    return records[:limit]


def parse_chembl_molecule(molecule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map one ChEMBL molecule record onto the criteria keys of §4.1."""
    chembl_id = molecule.get("molecule_chembl_id")
    if not chembl_id:
        return None
    props = molecule.get("molecule_properties") or {}
    structures = molecule.get("molecule_structures") or {}

    withdrawn = _as_bool(molecule.get("withdrawn_flag"))
    return {
        "source": "chembl",
        "record_id": chembl_id,
        "chembl_id": chembl_id,
        "name": molecule.get("pref_name") or chembl_id,
        "smiles": structures.get("canonical_smiles", "") or "",
        "inchikey": structures.get("standard_inchi_key", "") or "",
        "fields": _drop_none(
            {
                "molecular_weight": _as_float(props.get("full_mwt")),
                "logp": _as_float(props.get("alogp")),
                "tpsa": _as_float(props.get("psa")),
                "hbd": _as_float(props.get("hbd")),
                "hba": _as_float(props.get("hba")),
                "rotatable_bonds": _as_float(props.get("rtb")),
                "qed": _as_float(props.get("qed_weighted")),
                "ro5_violations": _as_float(props.get("num_ro5_violations")),
                "max_clinical_phase": _as_float(molecule.get("max_phase")),
                "orphan_designation": _orphan_flag(molecule.get("orphan")),
                "withdrawn_status": "withdrawn" if withdrawn else "not_withdrawn",
                "withdrawal_reason": (
                    _normalize_withdrawal_reason(molecule.get("withdrawn_reason"))
                    if withdrawn
                    else "not_withdrawn"
                ),
                "withdrawal_year": molecule.get("withdrawn_year"),
                "black_box_warning": _yes_no(molecule.get("black_box_warning")),
                "oral_administration": _yes_no(molecule.get("oral")),
            }
        ),
    }


def chembl_drug_warnings(molecule_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Withdrawal reason/year fallback.

    ChEMBL moved the withdrawal narrative out of the molecule record and into
    `drug_warning` in recent releases, so a molecule can report
    `withdrawn_flag: true` with no reason attached. Criterion 11b is worth 4
    points and the withdrawal year is a displayed field, so both are re-fetched
    here rather than reported as missing.
    """
    out: Dict[str, Dict[str, Any]] = {}
    batch = 20
    for start in range(0, len(molecule_ids), batch):
        chunk = molecule_ids[start : start + batch]
        payload = get_json(
            f"{CHEMBL_BASE}/drug_warning",
            {"molecule_chembl_id__in": ",".join(chunk), "format": "json", "limit": 100},
        )
        for warning in (payload or {}).get("drug_warnings") or []:
            molecule_id = warning.get("molecule_chembl_id")
            if not molecule_id or str(warning.get("warning_type", "")).lower() != "withdrawn":
                continue
            entry = out.setdefault(molecule_id, {})
            reason = _normalize_withdrawal_reason(
                warning.get("warning_class") or warning.get("efo_term")
            )
            if reason != "unknown" or "withdrawal_reason" not in entry:
                entry["withdrawal_reason"] = reason
            year = warning.get("warning_year")
            if isinstance(year, int):
                entry["withdrawal_year"] = year
    return out


# ── Open Targets ──────────────────────────────────────────────────────────────
# The Platform's `knownDrugs` field was replaced by `drugAndClinicalCandidates`,
# and clinical maturity is now a stage *string* rather than a numeric phase.
_OT_KNOWN_DRUGS = """
query KnownDrugs($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug { id name drugType maximumClinicalStage }
        diseases { disease { id name } }
      }
    }
  }
}
"""

_OT_DRUG = """
query DrugDetail($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    maximumClinicalStage
    drugWarnings { warningType toxicityClass description year country efoTerm }
    adverseEvents(page: {index: 0, size: 25}) {
      count
      criticalValue
      rows { name logLR count }
    }
    indications { count rows { maxClinicalStage disease { id name } } }
    mechanismsOfAction { rows { mechanismOfAction actionType } }
  }
}
"""

# `maxClinicalStage` values, mapped onto the 0-4 phase scale criterion 9 scores on.
CLINICAL_STAGE_TO_PHASE = {
    "APPROVAL": 4.0,
    "APPROVED": 4.0,
    "PHASE_4": 4.0,
    "PHASE_3": 3.0,
    "PHASE_2_3": 2.5,
    "PHASE_2": 2.0,
    "PHASE_1_2": 1.5,
    "PHASE_1": 1.0,
    "EARLY_PHASE_1": 0.5,
    "PRECLINICAL": 0.0,
}


def clinical_stage_to_phase(stage: Any) -> Optional[float]:
    """Map an Open Targets clinical stage string onto ChEMBL's numeric max phase."""
    if stage is None:
        return None
    if isinstance(stage, (int, float)):
        return float(stage)
    return CLINICAL_STAGE_TO_PHASE.get(str(stage).strip().upper())


def open_targets_known_drugs(ensembl_gene_id: str, size: int = 200) -> List[Dict[str, Any]]:
    """
    §2.1 + §5 step 7 — the drug/clinical-candidate rows for a target.

    Doubles as the exclusion source: each row carries the diseases the drug is
    already known for, which is what the repurposing filter matches against.
    """
    if not ensembl_gene_id:
        return []
    data = post_graphql(OPEN_TARGETS_GRAPHQL, _OT_KNOWN_DRUGS, {"ensemblId": ensembl_gene_id})
    target = (data or {}).get("target") or {}
    rows = ((target.get("drugAndClinicalCandidates") or {}).get("rows")) or []
    ligands: Dict[str, Dict[str, Any]] = {}
    for row in rows[:size]:
        drug = row.get("drug") or {}
        drug_id = drug.get("id")
        if not drug_id:
            continue
        entry = ligands.setdefault(
            drug_id,
            {
                "source": "open_targets",
                "record_id": drug_id,
                "chembl_id": drug_id,
                "name": drug.get("name") or drug_id,
                "smiles": "",
                "inchikey": "",
                "drug_type": drug.get("drugType") or "",
                "known_indications": [],
                "fields": {},
            },
        )
        for item in row.get("diseases") or []:
            disease = (item or {}).get("disease") or {}
            if not disease.get("id"):
                continue
            indication = {
                "efoId": disease["id"],
                "name": disease.get("name", ""),
                "phase": clinical_stage_to_phase(row.get("maxClinicalStage")),
            }
            if indication not in entry["known_indications"]:
                entry["known_indications"].append(indication)

        phase = clinical_stage_to_phase(
            drug.get("maximumClinicalStage") or row.get("maxClinicalStage")
        )
        if phase is not None:
            current = entry["fields"].get("max_clinical_phase")
            entry["fields"]["max_clinical_phase"] = max(phase, current or 0.0)
    return list(ligands.values())


_OT_DISEASE_KNOWN_DRUGS = """
query DiseaseKnownDrugs($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    drugAndClinicalCandidates {
      count
      rows { maxClinicalStage drug { id name } }
    }
  }
}
"""


def open_targets_disease_known_drugs(efo_ids: List[str], size: int = 500) -> Dict[str, Dict[str, Any]]:
    """
    §5 step 7 — every drug already known for the disease, keyed by ChEMBL drug ID.

    Called once per EFO term in the resolved disease's family (the term itself
    plus its descendants), because a drug approved for a subtype is not a
    repurposing candidate for the parent indication either.
    """
    excluded: Dict[str, Dict[str, Any]] = {}
    for efo_id in efo_ids:
        if not efo_id:
            continue
        data = post_graphql(OPEN_TARGETS_GRAPHQL, _OT_DISEASE_KNOWN_DRUGS, {"efoId": efo_id})
        disease = (data or {}).get("disease") or {}
        rows = ((disease.get("drugAndClinicalCandidates") or {}).get("rows")) or []
        for row in rows[:size]:
            drug = row.get("drug") or {}
            drug_id = drug.get("id")
            if not drug_id:
                continue
            entry = excluded.setdefault(
                drug_id,
                {
                    "drugId": drug_id,
                    "name": drug.get("name") or drug_id,
                    "matchedEfoIds": [],
                    "matchedDiseaseNames": [],
                    "maxPhase": None,
                },
            )
            if efo_id not in entry["matchedEfoIds"]:
                entry["matchedEfoIds"].append(efo_id)
                entry["matchedDiseaseNames"].append(disease.get("name", efo_id))
            phase = clinical_stage_to_phase(row.get("maxClinicalStage"))
            if phase is not None:
                entry["maxPhase"] = max(phase, entry["maxPhase"] or 0.0)
    return excluded


def open_targets_drug(chembl_id: str) -> Optional[Dict[str, Any]]:
    """§2.2 — per-drug safety/regulatory fields, including the FAERS AE signal."""
    if not chembl_id:
        return None
    data = post_graphql(OPEN_TARGETS_GRAPHQL, _OT_DRUG, {"chemblId": chembl_id})
    drug = (data or {}).get("drug")
    if not drug:
        return None

    ae_rows = ((drug.get("adverseEvents") or {}).get("rows")) or []
    # Criterion 14 is a *signal* strength: the strongest disproportionality
    # reported, not the mean, which any long tail of weak events would flatten.
    max_llr = max((_as_float(r.get("logLR")) or 0.0 for r in ae_rows), default=None)

    # Withdrawal and black-box status are no longer booleans on the drug record;
    # both are entries in `drugWarnings`, which also carries the reason
    # (`toxicityClass`) and the year that criteria 11b and the metadata row need.
    warnings = drug.get("drugWarnings") or []
    withdrawn_entries = [
        w for w in warnings if "withdraw" in str(w.get("warningType", "")).lower()
    ]
    black_box = any("black box" in str(w.get("warningType", "")).lower() for w in warnings)

    withdrawal_reason = None
    withdrawal_year = None
    if withdrawn_entries:
        for entry in withdrawn_entries:
            reason = _normalize_withdrawal_reason(
                entry.get("toxicityClass") or entry.get("efoTerm") or entry.get("description")
            )
            # Prefer a bucketed reason over "unknown" when several warnings exist.
            if withdrawal_reason in (None, "unknown"):
                withdrawal_reason = reason
            if entry.get("year") and withdrawal_year is None:
                withdrawal_year = entry["year"]

    indications = [
        {
            "efoId": (row.get("disease") or {}).get("id", ""),
            "name": (row.get("disease") or {}).get("name", ""),
            "phase": clinical_stage_to_phase(row.get("maxClinicalStage")),
        }
        for row in ((drug.get("indications") or {}).get("rows") or [])
        if (row.get("disease") or {}).get("id")
    ]
    mechanisms = [
        row.get("mechanismOfAction", "")
        for row in ((drug.get("mechanismsOfAction") or {}).get("rows") or [])
        if row.get("mechanismOfAction")
    ]

    return {
        "source": "open_targets",
        "record_id": drug.get("id", chembl_id),
        "chembl_id": chembl_id,
        "name": drug.get("name") or chembl_id,
        "known_indications": indications,
        "mechanism_of_action": "; ".join(mechanisms[:3]),
        "fields": _drop_none(
            {
                "max_clinical_phase": clinical_stage_to_phase(drug.get("maximumClinicalStage")),
                "withdrawn_status": "withdrawn" if withdrawn_entries else "not_withdrawn",
                "withdrawal_reason": withdrawal_reason
                if withdrawn_entries
                else "not_withdrawn",
                "withdrawal_year": withdrawal_year,
                "black_box_warning": "yes" if black_box else "no",
                "adverse_event_llr": max_llr,
            }
        ),
        "adverse_events": [
            {"name": r.get("name", ""), "logLR": _as_float(r.get("logLR"))} for r in ae_rows[:10]
        ],
    }


# ── BindingDB ─────────────────────────────────────────────────────────────────
def bindingdb_ligands(uniprot_accession: str, cutoff_nm: int = 10000) -> List[Dict[str, Any]]:
    """§2.1 supplementary binding affinity, queried by UniProt accession."""
    if not uniprot_accession:
        return []
    payload = get_json(
        BINDINGDB_LIGANDS,
        {
            "uniprot": uniprot_accession,
            "cutoff": cutoff_nm,
            "code": 0,
            "response": "application/json",
        },
    )
    hits = (((payload or {}).get("getLigandsByUniprotsResponse") or {}).get("affinities")) or []
    if isinstance(hits, dict):
        hits = [hits]
    records: List[Dict[str, Any]] = []
    for hit in hits:
        smiles = hit.get("smile") or hit.get("smiles") or ""
        monomer = hit.get("monomerid") or hit.get("monomerId")
        if not smiles and not monomer:
            continue
        records.append(
            {
                "source": "bindingdb",
                "record_id": str(monomer or smiles[:32]),
                "name": hit.get("query") or str(monomer or ""),
                "smiles": smiles,
                "inchikey": "",
                "bioactivity": {
                    "type": hit.get("affinity_type", ""),
                    "value_nm": _as_float(hit.get("affinity")),
                },
                "fields": {},
            }
        )
    return records


# ── PubChem BioAssay ──────────────────────────────────────────────────────────
def pubchem_bioassay_ligands(
    uniprot_accession: str, max_assays: int = 10, max_cids: int = 200
) -> List[Dict[str, Any]]:
    """
    §2.1 — target → AID → active CIDs, for targets thin in ChEMBL.

    Only *active* CIDs are pulled; the tested set includes every inactive
    compound in the screen, which would swamp the reference ligand set.
    """
    if not uniprot_accession:
        return []
    aid_payload = get_json(f"{PUBCHEM_BASE}/assay/target/accession/{uniprot_accession}/aids/JSON")
    aids = (((aid_payload or {}).get("IdentifierList") or {}).get("AID")) or []
    if not aids:
        return []

    cids: List[int] = []
    for aid in aids[:max_assays]:
        cid_payload = get_json(
            f"{PUBCHEM_BASE}/assay/aid/{aid}/cids/JSON", {"cids_type": "active"}
        )
        info = ((cid_payload or {}).get("InformationList") or {}).get("Information") or []
        for entry in info:
            for cid in entry.get("CID", [])[: max_cids - len(cids)]:
                if cid not in cids:
                    cids.append(cid)
        if len(cids) >= max_cids:
            break
    if not cids:
        return []

    records: List[Dict[str, Any]] = []
    batch = 100
    for start in range(0, len(cids), batch):
        chunk = ",".join(str(c) for c in cids[start : start + batch])
        props = get_json(
            f"{PUBCHEM_BASE}/compound/cid/{chunk}/property/CanonicalSMILES,InChIKey,Title/JSON"
        )
        for prop in (((props or {}).get("PropertyTable") or {}).get("Properties")) or []:
            records.append(
                {
                    "source": "pubchem_bioassay",
                    "record_id": str(prop.get("CID", "")),
                    "name": prop.get("Title") or str(prop.get("CID", "")),
                    "smiles": prop.get("CanonicalSMILES", "") or "",
                    "inchikey": prop.get("InChIKey", "") or "",
                    "fields": {},
                }
            )
    return records


# ── IUPHAR / Guide to Pharmacology ────────────────────────────────────────────
def iuphar_ligands(gene_symbol: str, max_ligands: int = 100) -> List[Dict[str, Any]]:
    """§2.1 — curated pharmacology, strongest for receptor and enzyme targets."""
    if not gene_symbol:
        return []
    targets = get_json(f"{IUPHAR_BASE}/targets", {"geneSymbol": gene_symbol})
    if not targets:
        return []
    target_id = (targets[0] or {}).get("targetId")
    if not target_id:
        return []

    interactions = get_json(f"{IUPHAR_BASE}/targets/{target_id}/interactions") or []
    ligand_ids: List[int] = []
    interaction_types: Dict[int, str] = {}
    for interaction in interactions:
        ligand_id = interaction.get("ligandId")
        if ligand_id and ligand_id not in ligand_ids:
            ligand_ids.append(ligand_id)
            interaction_types[ligand_id] = interaction.get("type", "") or ""

    records: List[Dict[str, Any]] = []
    for ligand_id in ligand_ids[:max_ligands]:
        ligand = get_json(f"{IUPHAR_BASE}/ligands/{ligand_id}")
        if not ligand:
            continue
        structure = get_json(f"{IUPHAR_BASE}/ligands/{ligand_id}/structure") or {}
        records.append(
            {
                "source": "iuphar",
                "record_id": str(ligand_id),
                "name": ligand.get("name", "") or str(ligand_id),
                # Often name-only; §3.3 resolves the structure before merging.
                "smiles": structure.get("smiles", "") or "",
                "inchikey": structure.get("inchiKey", "") or "",
                "interaction_type": interaction_types.get(ligand_id, ""),
                "fields": {},
            }
        )
    return records


# ── DGIdb ─────────────────────────────────────────────────────────────────────
_DGIDB_QUERY = """
query Interactions($names: [String!]!) {
  genes(names: $names) {
    nodes {
      name
      interactions {
        drug { name conceptId }
        interactionTypes { type directionality }
      }
    }
  }
}
"""


def dgidb_ligands(gene_symbol: str, max_drugs: int = 150) -> List[Dict[str, Any]]:
    """§2.1 — druggability and interaction type (agonist/antagonist/inhibitor)."""
    if not gene_symbol:
        return []
    data = post_graphql(DGIDB_GRAPHQL, _DGIDB_QUERY, {"names": [gene_symbol.upper()]})
    nodes = (((data or {}).get("genes") or {}).get("nodes")) or []
    records: List[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        for interaction in node.get("interactions") or []:
            drug = interaction.get("drug") or {}
            name = drug.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            types = [
                t.get("type", "") for t in (interaction.get("interactionTypes") or []) if t.get("type")
            ]
            records.append(
                {
                    "source": "dgidb",
                    "record_id": drug.get("conceptId", "") or name,
                    "name": name,          # name-only; §3.3 resolves the structure
                    "smiles": "",
                    "inchikey": "",
                    "interaction_type": ", ".join(types),
                    "fields": {},
                }
            )
            if len(records) >= max_drugs:
                return records
    return records


# ── helpers ───────────────────────────────────────────────────────────────────
def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "1"}


def _orphan_flag(value: Any) -> Optional[str]:
    """
    ChEMBL's `orphan` is tri-state: 1 designated, 0 not, -1 unknown.

    -1 must come back as None, not "yes" — it is truthy, and criterion 10 carries
    7 points, so treating "unknown" as a designation would inflate the regulatory
    score of every molecule ChEMBL has not assessed.
    """
    number = _as_float(value)
    if number is None or number < 0:
        return None
    return "yes" if number >= 1 else "no"


def _yes_no(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return "yes" if _as_bool(value) else "no"


_SAFETY_WORDS = (
    "toxic", "safety", "death", "cardio", "hepato", "nephro", "carcino",
    "teratogen", "adverse", "arrhythm", "hemato", "neuro", "immune", "misuse",
)
_COMMERCIAL_WORDS = ("commercial", "business", "marketing", "supply", "voluntar")
_EFFICACY_WORDS = ("efficacy", "lack of effect", "ineffective")


def _normalize_withdrawal_reason(raw: Any) -> str:
    """Bucket a free-text withdrawal reason into the §4.2 lookup keys."""
    if raw is None:
        return "unknown"
    text = str(raw).strip().lower()
    if not text or text in {"none", "unknown", "unspecified"}:
        return "unknown"
    if any(word in text for word in _COMMERCIAL_WORDS):
        return "commercial"
    if any(word in text for word in _EFFICACY_WORDS):
        return "efficacy"
    if any(word in text for word in _SAFETY_WORDS):
        return "safety"
    return "unknown"


def _drop_none(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Absent ≠ zero: a missing field must not reach the scorer as a value."""
    return {k: v for k, v in mapping.items() if v is not None}
