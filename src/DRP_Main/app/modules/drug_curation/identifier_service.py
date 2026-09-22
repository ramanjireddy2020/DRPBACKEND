"""
CurateX §3 — identifier resolution.

§3.1 target name / gene symbol → UniProt accession + Ensembl gene ID + ChEMBL target ID
§3.2 disease name → MeSH UI → EFO ID (+ the EFO descendant set used by §5 step 7)
§3.3 drug name / SMILES → standardized structure → InChIKey (the cross-source join key)
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.http_client import get_json, post_graphql

logger = get_logger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
MESH_LOOKUP_DESCRIPTOR = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
OPEN_TARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
CHEMBL_MOLECULE_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
PUBCHEM_NAME_PROPS = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/CanonicalSMILES,InChIKey/JSON"
)

_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)\\/%=#$:.*~{}]+$")


class ResolutionError(RuntimeError):
    """An identifier could not be resolved to anything usable downstream."""


# ── Result records ────────────────────────────────────────────────────────────
@dataclass
class ResolvedTarget:
    query: str
    uniprot_accession: str
    gene_symbol: str = ""
    protein_name: str = ""
    ensembl_gene_id: str = ""
    chembl_target_id: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "query": self.query,
            "uniprotAccession": self.uniprot_accession,
            "geneSymbol": self.gene_symbol,
            "proteinName": self.protein_name,
            "ensemblGeneId": self.ensembl_gene_id,
            "chemblTargetId": self.chembl_target_id,
        }


@dataclass
class ResolvedDisease:
    query: str
    mesh_id: str = ""
    mesh_label: str = ""
    efo_id: str = ""
    efo_label: str = ""
    # §5 step 7 / §6 step 2: the resolved term plus every child/subtype term.
    efo_descendants: List[str] = field(default_factory=list)
    resolution_path: str = ""

    @property
    def efo_family(self) -> List[str]:
        """The EFO IDs the exclusion filter matches against."""
        family = [self.efo_id] if self.efo_id else []
        family.extend(d for d in self.efo_descendants if d and d != self.efo_id)
        return family

    def as_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "meshId": self.mesh_id,
            "meshLabel": self.mesh_label,
            "efoId": self.efo_id,
            "efoLabel": self.efo_label,
            "efoDescendantCount": len(self.efo_descendants),
            "resolutionPath": self.resolution_path,
        }


@dataclass
class ResolvedStructure:
    name: str
    smiles: str = ""
    inchikey: str = ""
    standardized: bool = False
    source: str = ""


# ── §3.1 target ───────────────────────────────────────────────────────────────
_target_cache: Dict[str, ResolvedTarget] = {}
_cache_lock = threading.Lock()


def resolve_target(name_or_symbol: str) -> ResolvedTarget:
    """
    Resolve a gene symbol / protein name to all three downstream IDs in one call.

    Restricted to human, reviewed (Swiss-Prot) entries: TrEMBL hits are unreviewed
    and routinely return several accessions for one symbol, which would make the
    ChEMBL/Ensembl cross-references ambiguous. The UniProt record carries the
    Ensembl gene ID and the ChEMBL target ID as cross-references, so this single
    lookup supplies every ID the rest of the pipeline needs.
    """
    query = (name_or_symbol or "").strip()
    if not query:
        raise ResolutionError("A target name or gene symbol is required")

    key = query.lower()
    with _cache_lock:
        cached = _target_cache.get(key)
    if cached:
        return cached

    payload = get_json(
        UNIPROT_SEARCH,
        {
            # Gene-name match first; UniProt falls back to full-text within the
            # same query syntax when the symbol is really a protein name.
            "query": f'(gene:"{query}" OR protein_name:"{query}") '
                     f"AND organism_id:9606 AND reviewed:true",
            "fields": "accession,id,protein_name,gene_names,xref_chembl,xref_ensembl",
            "format": "json",
            "size": 5,
        },
    )
    results = (payload or {}).get("results") or []
    if not results:
        raise ResolutionError(
            f"No reviewed human UniProt entry matched '{query}'. "
            "Use the gene symbol (e.g. 'JAK2') or the canonical protein name."
        )

    entry = results[0]
    accession = entry.get("primaryAccession", "")
    genes = entry.get("genes") or []
    gene_symbol = ""
    if genes:
        gene_symbol = ((genes[0].get("geneName") or {}).get("value")) or ""
    protein_name = (
        ((entry.get("proteinDescription") or {}).get("recommendedName") or {})
        .get("fullName", {})
        .get("value", "")
    )

    ensembl_gene_id, chembl_target_id = "", ""
    for xref in entry.get("uniProtKBCrossReferences") or []:
        database = xref.get("database")
        if database == "Ensembl" and not ensembl_gene_id:
            # The Ensembl xref's id is a transcript; the gene ID is a property.
            for prop in xref.get("properties") or []:
                if prop.get("key") == "GeneId":
                    ensembl_gene_id = (prop.get("value") or "").split(".")[0]
                    break
        elif database == "ChEMBL" and not chembl_target_id:
            chembl_target_id = xref.get("id", "")

    resolved = ResolvedTarget(
        query=query,
        uniprot_accession=accession,
        gene_symbol=gene_symbol or query.upper(),
        protein_name=protein_name,
        ensembl_gene_id=ensembl_gene_id,
        chembl_target_id=chembl_target_id,
    )
    if not chembl_target_id:
        # Not fatal — BindingDB/PubChem/IUPHAR can still be queried by UniProt —
        # but ChEMBL is the primary criteria source, so say so loudly.
        logger.warning(
            "UniProt %s carries no ChEMBL cross-reference; ChEMBL ligands unavailable for %s",
            accession, query,
        )
    with _cache_lock:
        _target_cache[key] = resolved
    return resolved


# ── §3.2 disease ──────────────────────────────────────────────────────────────
_disease_cache: Dict[str, ResolvedDisease] = {}

_OT_DISEASE_SEARCH = """
query DiseaseSearch($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
    hits { id name entity }
  }
}
"""

_OT_DISEASE_DETAIL = """
query DiseaseDetail($efoId: String!) {
  disease(efoId: $efoId) { id name dbXRefs descendants }
}
"""


def lookup_mesh_id(disease_name: str) -> Tuple[str, str]:
    """MeSH Lookup API — fuzzy/synonym-aware match to a descriptor UI (e.g. D008545)."""
    payload = get_json(
        MESH_LOOKUP_DESCRIPTOR,
        {"label": disease_name, "match": "contains", "limit": 10},
    )
    if not payload:
        return "", ""
    hits = payload if isinstance(payload, list) else []
    if not hits:
        return "", ""
    lowered = disease_name.strip().lower()
    # Prefer an exact label match; the API's "contains" mode happily returns
    # broader descriptors that would anchor the query onto the wrong disease.
    best = next((h for h in hits if str(h.get("label", "")).lower() == lowered), hits[0])
    resource = str(best.get("resource", ""))
    mesh_id = resource.rsplit("/", 1)[-1] if resource else ""
    return mesh_id, str(best.get("label", ""))


def _ot_disease_detail(efo_id: str) -> Optional[Dict[str, object]]:
    data = post_graphql(OPEN_TARGETS_GRAPHQL, _OT_DISEASE_DETAIL, {"efoId": efo_id})
    return (data or {}).get("disease")


def _efo_from_mesh(mesh_id: str) -> Tuple[str, str, List[str]]:
    """
    Cross-walk a MeSH UI to EFO via Open Targets' own dbXRefs.

    The spec assumes an internal MeSH → EFO table built by the BioKG loader; no
    such table exists in this repo, and EFO's dbXrefs are exactly what that table
    would have been parsed from. Querying Open Targets for them is the same
    mapping without a build step, and it stays current.
    """
    if not mesh_id:
        return "", "", []
    data = post_graphql(
        OPEN_TARGETS_GRAPHQL,
        _OT_DISEASE_SEARCH,
        {"q": f"MESH:{mesh_id}"},
    )
    hits = (((data or {}).get("search") or {}).get("hits")) or []
    for hit in hits:
        detail = _ot_disease_detail(hit.get("id", ""))
        if not detail:
            continue
        xrefs = {str(x).upper() for x in (detail.get("dbXRefs") or [])}
        if f"MESH:{mesh_id}".upper() in xrefs or mesh_id.upper() in xrefs:
            return (
                str(detail.get("id", "")),
                str(detail.get("name", "")),
                [str(d) for d in (detail.get("descendants") or [])],
            )
    return "", "", []


def resolve_disease(disease_name: str) -> ResolvedDisease:
    """
    Resolve a disease name to a MeSH UI and an EFO ID, plus the EFO descendant set.

    Order per §3.2: MeSH lookup → MeSH-to-EFO cross-walk → free-text Open Targets
    disease search as the fallback when the MeSH term carries no EFO mapping.
    """
    query = (disease_name or "").strip()
    if not query:
        raise ResolutionError("A disease name is required")

    key = query.lower()
    with _cache_lock:
        cached = _disease_cache.get(key)
    if cached:
        return cached

    mesh_id, mesh_label = lookup_mesh_id(query)
    efo_id, efo_label, descendants = _efo_from_mesh(mesh_id)
    path = "mesh_lookup+efo_xref" if efo_id else ""

    if not efo_id:
        data = post_graphql(OPEN_TARGETS_GRAPHQL, _OT_DISEASE_SEARCH, {"q": query})
        hits = (((data or {}).get("search") or {}).get("hits")) or []
        if hits:
            detail = _ot_disease_detail(hits[0].get("id", ""))
            if detail:
                efo_id = str(detail.get("id", ""))
                efo_label = str(detail.get("name", ""))
                descendants = [str(d) for d in (detail.get("descendants") or [])]
                path = "open_targets_search"
                if not mesh_id:
                    for xref in detail.get("dbXRefs") or []:
                        if str(xref).upper().startswith("MESH:"):
                            mesh_id = str(xref).split(":", 1)[1]
                            mesh_label = efo_label
                            break

    if not efo_id and not mesh_id:
        raise ResolutionError(
            f"'{query}' matched neither a MeSH descriptor nor an EFO term. "
            "Try the disease name as it appears in MeSH (e.g. 'Diabetes Mellitus, Type 2')."
        )
    if not efo_id:
        # MeSH resolved but EFO did not: literature work can proceed, the
        # exclusion filter cannot. The caller reports this rather than silently
        # scoring drugs that are already indicated.
        logger.warning("Disease '%s' resolved to MeSH %s but no EFO term", query, mesh_id)
        path = path or "mesh_only"

    resolved = ResolvedDisease(
        query=query,
        mesh_id=mesh_id,
        mesh_label=mesh_label,
        efo_id=efo_id,
        efo_label=efo_label,
        efo_descendants=descendants,
        resolution_path=path,
    )
    with _cache_lock:
        _disease_cache[key] = resolved
    return resolved


# ── §3.3 molecule structure ───────────────────────────────────────────────────
_structure_cache: Dict[str, ResolvedStructure] = {}
_rdkit_unavailable = False


def _rdkit():
    """Lazy RDKit import — heavy, and the module must import without it."""
    global _rdkit_unavailable
    if _rdkit_unavailable:
        return None
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize

        RDLogger.DisableLog("rdApp.*")
        return Chem, rdMolStandardize
    except ImportError:
        _rdkit_unavailable = True
        logger.warning("RDKit unavailable — structures will not be standardized")
        return None


def looks_like_smiles(text: str) -> bool:
    """Cheap discriminator between a drug name and a SMILES string."""
    candidate = (text or "").strip()
    if not candidate or " " in candidate:
        return False
    if not _SMILES_CHARS.match(candidate):
        return False
    # A name like "aspirin" is all letters; a SMILES almost always carries a
    # ring closure, a branch, a bond symbol or an explicit atom block.
    return bool(re.search(r"[0-9@+\-\[\]\(\)=#]", candidate))


def standardize_smiles(smiles: str) -> Tuple[str, str]:
    """
    Strip salts and normalize tautomer/charge state, then compute the InChIKey.

    Returns (standardized_smiles, inchikey); either may be empty when RDKit is
    unavailable or the structure will not parse.
    """
    if not smiles:
        return "", ""
    loaded = _rdkit()
    if loaded is None:
        return smiles, ""
    Chem, rdMolStandardize = loaded
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, ""
        # Salt strip → charge neutralize → canonical tautomer, so that a salt
        # form and its parent collapse onto one InChIKey at merge time (§5 step 4).
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)
    except Exception as exc:  # RDKit raises bare Exceptions on odd valences
        logger.debug("Structure standardization failed for %s: %s", smiles[:60], exc)
        return smiles, ""


def _smiles_from_chembl(name: str) -> Tuple[str, str]:
    payload = get_json(CHEMBL_MOLECULE_SEARCH, {"q": name, "format": "json", "limit": 1})
    molecules = (payload or {}).get("molecules") or []
    if not molecules:
        return "", ""
    structures = molecules[0].get("molecule_structures") or {}
    return structures.get("canonical_smiles", "") or "", molecules[0].get(
        "molecule_chembl_id", ""
    )


def _smiles_from_pubchem(name: str) -> str:
    from urllib.parse import quote

    payload = get_json(PUBCHEM_NAME_PROPS.format(name=quote(name)))
    props = (((payload or {}).get("PropertyTable") or {}).get("Properties")) or []
    if not props:
        return ""
    return props[0].get("CanonicalSMILES", "") or ""


def resolve_structure(name_or_smiles: str) -> ResolvedStructure:
    """
    §3.3 — resolve a name-only record to a standardized structure and InChIKey.

    Used for ligands that arrive from DGIdb / IUPHAR without a structure; without
    this they cannot be merged onto the ChEMBL/Open Targets records in §5 step 4.
    """
    text = (name_or_smiles or "").strip()
    if not text:
        return ResolvedStructure(name="")

    key = text.lower()
    with _cache_lock:
        cached = _structure_cache.get(key)
    if cached:
        return cached

    if looks_like_smiles(text):
        std_smiles, inchikey = standardize_smiles(text)
        result = ResolvedStructure(
            name=text, smiles=std_smiles, inchikey=inchikey,
            standardized=bool(inchikey), source="input_smiles",
        )
    else:
        smiles, _chembl_id = _smiles_from_chembl(text)
        source = "chembl"
        if not smiles:
            smiles = _smiles_from_pubchem(text)
            source = "pubchem" if smiles else ""
        std_smiles, inchikey = standardize_smiles(smiles)
        result = ResolvedStructure(
            name=text, smiles=std_smiles or smiles, inchikey=inchikey,
            standardized=bool(inchikey), source=source,
        )
    with _cache_lock:
        _structure_cache[key] = result
    return result


def clear_caches() -> None:
    """Drop the in-process resolution caches (used by tests and /health tooling)."""
    with _cache_lock:
        _target_cache.clear()
        _disease_cache.clear()
        _structure_cache.clear()
