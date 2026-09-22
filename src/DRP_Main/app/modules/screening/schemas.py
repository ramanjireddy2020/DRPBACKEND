"""
Pydantic schemas for the Screening (ScreenSuite) module.

Two generations of contract live here on purpose:

* **Resolution + batch** (`ResolvedProtein`, `ResolvedDrug`, `ScreenBatchRequest`,
  `ScreenBatchResult`) — the ScreenSuite spec's shapes. A resolved record always
  carries the identifier *and* the source it came from, because structure
  retrieval branches on source (PDB vs. AlphaFold, PubChem vs. ZINC) and a
  source-less identifier cannot be acted on unambiguously.
* **Per-protein** (`ProteinBase`, `DrugBase`, `ProcessProteinResponse`) — the
  original one-protein-per-call contract, kept because `drp/runners.py` and the
  legacy `/screening/process-protein/` routes are built on it.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# ── Sources ───────────────────────────────────────────────────────────────────

class ProteinSource(str, Enum):
    """Where a protein structure is fetched from."""
    pdb = "pdb"
    alphafold = "alphafold"


class DrugSource(str, Enum):
    """Where a ligand structure is fetched from."""
    pubchem = "pubchem"
    zinc = "zinc"


# ── Resolution step ───────────────────────────────────────────────────────────

class ProteinQuery(BaseModel):
    """A protein as the agent received it: a name, an identifier, or both."""
    name: str
    identifier: Optional[str] = None
    source: Optional[ProteinSource] = None

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"name": "JAK2"}, {"name": "AcrB", "identifier": "6ABJ", "source": "pdb"}]}
    )


class DrugQuery(BaseModel):
    """A drug as the agent received it: a name, an identifier, or both."""
    name: str
    identifier: Optional[str] = None
    source: Optional[DrugSource] = None

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"name": "Chlorthalidone"}, {"name": "Aspirin", "identifier": "2244", "source": "pubchem"}]}
    )


class ResolvedProtein(BaseModel):
    """A protein resolved to one identifier, tagged with its source."""
    name: str
    identifier: str
    source: ProteinSource


class ResolvedDrug(BaseModel):
    """A drug resolved to one identifier, tagged with its source."""
    name: str
    identifier: str
    source: DrugSource


class CandidateMatch(BaseModel):
    """One entry in an ambiguous-resolution shortlist."""
    identifier: str
    source: str
    title: Optional[str] = None
    detail: Optional[str] = None


class AmbiguousEntity(BaseModel):
    """A name that matched several structures — held for confirmation."""
    name: str
    kind: str                                 # "protein" | "drug"
    candidates: List[CandidateMatch] = Field(default_factory=list)


class UnresolvedEntity(BaseModel):
    """A name that matched nothing. Reported, never silently dropped."""
    name: str
    kind: str                                 # "protein" | "drug"
    reason: str


class ResolutionResult(BaseModel):
    """
    The resolution step's whole output.

    `stage` is `awaiting_structure_confirmation` when anything is ambiguous —
    mirroring CurateX's `awaiting_target_confirmation` — and `resolved` when the
    batch is ready to screen.
    """
    stage: str
    proteins: List[ResolvedProtein] = Field(default_factory=list)
    drugs: List[ResolvedDrug] = Field(default_factory=list)
    ambiguous: List[AmbiguousEntity] = Field(default_factory=list)
    unresolved: List[UnresolvedEntity] = Field(default_factory=list)
    message: str = ""


class ResolutionRequest(BaseModel):
    """Input to the resolution step: names, identifiers, or a mix."""
    proteins: List[ProteinQuery] = Field(default_factory=list)
    drugs: List[DrugQuery] = Field(default_factory=list)


class ConfirmationSelection(BaseModel):
    """One choice made against an ambiguous-resolution shortlist."""
    name: str
    kind: str                                 # "protein" | "drug"
    identifier: str
    source: Optional[str] = None


class ConfirmResolutionRequest(BaseModel):
    """
    A confirmation round-trip: the shortlist choices plus the original queries
    they belong to, so the step can re-resolve the batch as a whole.
    """
    selections: List[ConfirmationSelection] = Field(default_factory=list)
    proteins: List[ProteinQuery] = Field(default_factory=list)
    drugs: List[DrugQuery] = Field(default_factory=list)


# ── Docking results ───────────────────────────────────────────────────────────

class AffinityRecord(BaseModel):
    Mode: int
    Affinity_kcal_per_mol: float
    protein_ligand: str = Field(
        validation_alias=AliasChoices("protein_ligand", "protien_ligand")
    )
    protein: str
    ligand: str
    out_pdbqt_file: str


class InteractionEntry(BaseModel):
    """One detected interaction between a ligand atom group and a residue."""
    residue: str
    interaction_type: str
    ligand_pose: int
    count: int = 1


class InteractionReport(BaseModel):
    """
    Interaction profiling for one pose, organised **by interaction type** with
    the actual rows under each — not a dataframe dump or bare column labels.
    """
    status: str
    protein: Optional[str] = None
    ligand: Optional[str] = None
    mode: Optional[int] = None
    interactions_by_type: Dict[str, List[InteractionEntry]] = Field(default_factory=dict)
    binding_site_residues: List[str] = Field(default_factory=list)
    interaction_count: int = 0
    report_file: Optional[str] = None
    error: Optional[str] = None


class EntityFailure(BaseModel):
    """
    A protein or drug that resolved fine but failed inside the pipeline —
    a download that 404'd, a preparation step that failed, a docking crash.
    Distinct from `UnresolvedEntity`, which failed a stage earlier.
    """
    name: str
    kind: str                                 # "protein" | "drug"
    stage: str                                # "download" | "preparation" | "docking" | "interaction"
    error: str


class ProteinScreenResult(BaseModel):
    """One protein's screening outcome. Ranked independently of every other."""
    protein_name: str
    identifier: Optional[str] = None
    source: Optional[str] = None
    status: str
    receptor_pdb: Optional[str] = None
    affinity_records: List[Dict[str, Any]] = Field(default_factory=list)
    top_affinity_records: List[Dict[str, Any]] = Field(default_factory=list)
    interaction_reports: List[InteractionReport] = Field(default_factory=list)
    files: Dict[str, Any] = Field(default_factory=dict)
    time_taken: Optional[str] = None


class ScreenBatchRequest(BaseModel):
    """
    The pipeline's input: the resolution step's output, whole.

    Every protein × every drug in one submission — not one protein per call.
    Nothing unresolved reaches here; a record without a usable identifier is a
    contract violation upstream, not something to recover from silently.
    """
    proteins: List[ResolvedProtein]
    drugs: List[ResolvedDrug]
    run_id: Optional[str] = None
    box_mode: Optional[str] = None            # "site" | "protein"; None uses settings
    exhaustiveness: Optional[int] = None
    num_modes: Optional[int] = None
    skip_interactions: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "proteins": [{"name": "AcrB", "identifier": "6ABJ", "source": "pdb"}],
                    "drugs": [{"name": "Chlorthalidone", "identifier": "2732", "source": "pubchem"}],
                }
            ]
        }
    )


class ScreenBatchResult(BaseModel):
    """One combined result for the whole batch."""
    run_id: str
    status: str
    proteins: List[ProteinScreenResult] = Field(default_factory=list)
    failures: List[EntityFailure] = Field(default_factory=list)
    files: Dict[str, Any] = Field(default_factory=dict)
    combinations: int = 0
    time_taken: Optional[str] = None
    recommendation: str = ""


# ── Legacy per-protein contract (drp/runners.py + /screening legacy routes) ───

class InteractionRequest(BaseModel):
    receptor_pdb: str
    vina_output_pdbqt: str
    mode: int = 1
    ligand: Optional[str] = None


class ProteinBase(BaseModel):
    protein_name: str
    pdb_file_path: str
    top_n_affinity_records: Optional[List[AffinityRecord]] = None
    status: Optional[str] = None
    time_taken: Optional[str] = None
    task_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "protein_name": "AcrB",
                    "pdb_file_path": "app/data/protein_structures/AcrB_6ABJ.pdb",
                }
            ]
        }
    )


class DrugBase(BaseModel):
    drug_name: str
    sdf_file_path: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "drug_name": "Chlorthalidone",
                    "sdf_file_path": "app/data/drug_structures/Chlorthalidone_2732.sdf",
                }
            ]
        }
    )


class ProcessProteinResponse(BaseModel):
    protein_name: str
    pdb_file_path: str
    top_n_affinity_records: List[AffinityRecord]
    status: str
    time_taken: str
    abort: Optional[bool] = False


class DownloadProteinDrugResponse(BaseModel):
    proteins: List[ProteinBase]
    drugs: List[DrugBase]
