"""
Pydantic models for the DRP `/v1` contract.

Field names are camelCase to match the published spec exactly — these are wire
models, not internal ones.
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Email is validated with a pattern rather than pydantic's EmailStr so the API
# does not require the optional `email-validator` dependency to boot.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

ModuleKey = Literal["TxKG", "LitMineX", "CurateX", "ScreenSuite", "NovSearch", "SaaS Pipeline"]
SessionStatus = Literal["Completed", "In Progress", "Saved"]
ProjectStatus = Literal["Active", "On Hold", "Review"]
JobStatus = Literal["queued", "running", "completed", "failed"]
ResultType = Literal["targets", "subgraph", "metapath", "litminex_results", "article"]
ExportFormat = Literal["csv", "json", "png", "svg"]
NodeType = Literal[
    "Disease Hub", "Protein", "Pathway", "Compound", "Genetic Disorder", "Comorbidity"
]


class _Wire(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


# ── Auth & Onboarding ────────────────────────────────────────────────────────
class User(_Wire):
    id: str
    name: str = ""
    role: str = ""
    avatarUrl: str = ""
    email: str = ""


class LoginRequest(_Wire):
    email: str = Field(..., pattern=EMAIL_PATTERN, examples=["priya@drp.app"])
    password: str


class LoginResponse(_Wire):
    accessToken: str
    refreshToken: str
    expiresIn: int = 3600
    user: User


class RefreshRequest(_Wire):
    refreshToken: str


class RefreshResponse(_Wire):
    accessToken: str
    expiresIn: int = 3600


class ForgotPasswordRequest(_Wire):
    email: Optional[str] = Field(None, pattern=EMAIL_PATTERN)


class SuccessResponse(_Wire):
    success: bool = True
    message: str = ""


class OnboardingStatus(_Wire):
    completed: bool
    currentStep: int


class ResearchFocusRequest(_Wire):
    therapeuticAreas: List[str] = Field(default_factory=list, examples=[["Oncology / Cancer", "Diabetes"]])


# ── Dashboard ────────────────────────────────────────────────────────────────
class DashboardSummary(_Wire):
    activeProjects: int
    targetsIdentified: int
    compoundsCurated: int
    patentsAnalysed: int


class QuickAction(_Wire):
    label: str
    module: str
    prefillQuery: str


class PipelineProgress(_Wire):
    target: str
    pipelineType: str
    progressPercent: int


class PipelineDetail(PipelineProgress):
    id: str
    status: str
    projectId: Optional[str] = None
    jobId: Optional[str] = None
    updatedAt: Optional[datetime] = None


# ── Modules ──────────────────────────────────────────────────────────────────
class Module(_Wire):
    key: str
    displayName: str
    icon: str = ""
    description: str = ""


# ── Sessions ─────────────────────────────────────────────────────────────────
class SessionMessage(_Wire):
    role: Literal["user", "agent"]
    agentName: str = ""
    content: str = ""
    stepId: Optional[str] = None


class SessionStep(_Wire):
    """One module run in a session's chain."""

    id: str
    stepIndex: int = 0
    module: str = ""
    status: str = "In Progress"
    summary: str = ""
    # The job fulfilling this step. Poll `/agents/jobs/{jobId}/status` with it —
    # this is how the UI follows progress after submitting a query.
    jobId: Optional[str] = None
    # What the researcher chose on this step, e.g. {"targetIds": ["P37231"]}.
    selections: Dict[str, Any] = Field(default_factory=dict)
    parentStepId: Optional[str] = None
    rerunOfStepId: Optional[str] = None
    branchName: str = ""
    updatedAt: Optional[datetime] = None


class Session(_Wire):
    id: str
    module: str = ""
    title: str = ""
    status: SessionStatus = "In Progress"
    summary: str = ""
    updatedAt: Optional[datetime] = None
    messages: List[SessionMessage] = Field(default_factory=list)
    # The chain, oldest first. `jobId` is the job of the newest step, lifted to the
    # top level so a client that has just created a session can start polling
    # without walking the steps.
    steps: List[SessionStep] = Field(default_factory=list)
    jobId: Optional[str] = None
    currentStepId: Optional[str] = None
    # The branch the session is currently on; "" is main. `jobId`,
    # `currentStepId`, `module` and `summary` all describe this branch, not the
    # newest step in the session — those differ once the chain forks. Each
    # step's own `branchName` says which branch it belongs to.
    activeBranch: str = ""


class SessionArtifact(_Wire):
    """One output a session produced — a completed step's result."""

    id: str
    stepId: str
    jobId: str
    module: str = ""
    title: str = ""
    resultType: ResultType = "targets"
    summary: str = ""
    createdAt: Optional[datetime] = None
    exportable: bool = True


class CreateStepRequest(_Wire):
    """Advance a session to its next module, carrying selections forward."""

    module: str = Field(..., examples=["LitMineX"])
    selections: Dict[str, Any] = Field(
        default_factory=dict, examples=[{"targetIds": ["P37231", "P27487"]}]
    )
    # Defaults to the session's latest step. Naming an earlier step forks the
    # chain there — that is what "Branch" does.
    fromStepId: Optional[str] = None
    branchName: str = ""
    query: str = ""


class RerunStepRequest(_Wire):
    """Re-run a step, optionally with adjusted parameters."""

    params: Dict[str, Any] = Field(default_factory=dict)


class SessionMessageRequest(_Wire):
    message: str = Field(..., examples=["Why is JAK2 ranked lower than DPP4?"])
    stepId: Optional[str] = None
    # Which branch the researcher is looking at; "" is main. Send it whenever the
    # branch switcher changes, so a question asked on main is answered from
    # main's results rather than from whichever branch ran most recently.
    # Omitting it leaves the session on the branch it was already on.
    branchName: Optional[str] = None


class SessionMessageResponse(_Wire):
    role: Literal["user", "agent"] = "agent"
    agentName: str = ""
    content: str = ""
    stepId: Optional[str] = None
    # Non-null when answering required starting fresh agent work; poll it.
    jobId: Optional[str] = None


class UpdateSessionRequest(_Wire):
    """Rename a session or mark it Saved / Completed."""

    title: Optional[str] = None
    status: Optional[SessionStatus] = None


class SessionPage(_Wire):
    items: List[Session]
    page: int
    totalPages: int


class CreateSessionRequest(_Wire):
    query: str
    module: Optional[str] = Field(None, examples=["TxKG"])
    projectId: Optional[str] = None


class DraftSessionRequest(_Wire):
    query: Optional[str] = None
    module: Optional[str] = None
    projectId: Optional[str] = None


class DraftSessionResponse(_Wire):
    sessionId: str
    query: str = ""
    module: Optional[str] = None
    projectId: Optional[str] = None
    savedAt: Optional[datetime] = None


# ── Jobs ─────────────────────────────────────────────────────────────────────
class JobAccepted(_Wire):
    jobId: str


class JobStatusResponse(_Wire):
    jobId: str
    status: JobStatus
    progressMessage: str = ""
    module: str = ""
    error: str = ""


# ── TxKG ─────────────────────────────────────────────────────────────────────
class TxkgQueryRequest(_Wire):
    query: str = Field(
        ..., examples=["Find protein targets associated with Type 2 Diabetes for drug repurposing"]
    )
    projectId: Optional[str] = None
    limit: int = Field(10, ge=1, le=500)
    maxHops: int = Field(3, ge=1, le=5)


class Target(_Wire):
    uniprotId: str = Field(..., examples=["P37231"])
    name: str = Field(..., examples=["PPARG receptor"])
    score: float = Field(..., examples=[89.0])
    geneName: Optional[str] = None
    fullName: Optional[str] = None
    hopDistance: Optional[int] = None
    # Type-level templates, e.g. "disease→gene/protein→biological_process".
    connectionTypes: List[str] = Field(default_factory=list)
    # The same paths with the entities actually traversed, e.g.
    # "Thrombocytosis —Associated With→ JAK2". Shortest/best-sourced first.
    connectionPaths: List[str] = Field(
        default_factory=list,
        examples=[["Thrombocytosis —Associated With→ JAK2"]],
    )
    fromKnowledgeGraph: bool = True
    customAdded: bool = False
    # TxKG spec labels (§4 category, §5 sourcing gate, §7 novelty axis). Optional so
    # other producers of this shape stay valid.
    category: Optional[Literal["Known/Direct", "Hidden/Novel"]] = None
    confirmed: Optional[bool] = None
    sourcingStatus: Optional[str] = None
    noveltyLabel: Optional[str] = None
    correctedScore: Optional[float] = None
    literatureHits: Optional[int] = None
    patentHits: Optional[int] = None
    supportingSources: List[str] = Field(default_factory=list)


class TargetDetail(_Wire):
    uniprotId: str
    name: str = ""
    geneName: Optional[str] = None
    fullName: Optional[str] = None
    organism: str = ""
    uniprotUrl: str = ""
    score: Optional[float] = None
    connectionTypes: List[str] = Field(default_factory=list)


class SourceLink(_Wire):
    """
    One citation in the Sources tab.

    `url` points at the **record** wherever an identifier exists — the MeSH entry
    for the disease, the UniProt entry for a target. `kind="database"` marks the
    curated sources behind an edge (CTD, DisGeNET, …): the knowledge graph records
    which database asserted a relationship but not that database's own record id,
    so those can only link to the database itself. The UI should render the two
    differently rather than implying every link is record-level.
    """

    name: str
    url: str = ""
    kind: Literal["record", "database", "method"] = "record"
    detail: str = ""


class InsightPanel(_Wire):
    tab: Literal["interpretation", "recommendations", "sources"]
    content: str = ""
    items: List[str] = Field(default_factory=list)
    # Added alongside `items` rather than replacing it, so existing clients keep
    # working while the UI moves to the linked form.
    links: List[SourceLink] = Field(default_factory=list)


# ── Knowledge graph ──────────────────────────────────────────────────────────
class GraphNode(_Wire):
    id: str
    label: str
    type: NodeType


class GraphEdge(_Wire):
    source: str
    target: str
    label: str = ""


class KnowledgeGraph(_Wire):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    legend: Dict[str, str] = Field(default_factory=dict)


class SubgraphRequest(_Wire):
    disease: str = Field(..., examples=["Type 2 Diabetes"])
    targetIds: List[str] = Field(default_factory=list)
    selectedTarget: Optional[str] = Field(None, examples=["JAK2"])
    maxNodes: int = Field(100, ge=10, le=500)


class GraphStats(_Wire):
    relationshipsFound: int = 0
    drugCandidates: int = 0
    pathwayConnections: int = 0


class ExploreRequest(_Wire):
    nodeId: str = Field(..., examples=["JAK2"])


class MetapathRequest(_Wire):
    jobId: str


class MetapathSummary(_Wire):
    jobId: str
    paths: int = 0
    targets: int = 0
    pathways: int = 0
    nodes: int = 0
    edges: int = 0
    clusters: int = 0


class TargetPredictionScore(_Wire):
    target: str
    uniprotId: str = ""
    score: float = 0.0
    contextScore: float = 0.0
    totalPaths: int = 0


class MetapathTraversal(_Wire):
    target: str
    uniprotId: str = ""
    totalPaths: int = 0
    pathsByHop: Dict[str, int] = Field(default_factory=dict)
    paths: List[Dict[str, Any]] = Field(default_factory=list)


# ── LitMinex ─────────────────────────────────────────────────────────────────
class CustomTargetRequest(_Wire):
    targetName: str = Field(..., examples=["EGFR"])


class LitminexQueryRequest(_Wire):
    targetIds: List[str] = Field(default_factory=list)
    query: str = Field(
        "", examples=["Mine literature for Type 2 Diabetes drug targets with confidence scoring"]
    )
    maxResults: int = Field(20, ge=1, le=100)


class ArticleSummary(_Wire):
    id: str
    title: str = ""
    year: Optional[int] = Field(None, examples=[2024])
    confidenceScore: float = Field(0.0, examples=[100])
    foundKeywords: List[str] = Field(default_factory=list)


class ArticlePage(_Wire):
    totalArticles: int
    page: int
    totalPages: int
    items: List[ArticleSummary] = Field(default_factory=list)


class ArticleDetail(_Wire):
    id: str
    title: str = ""
    authors: str = Field("", examples=["Chen, S. et al."])
    year: Optional[int] = Field(None, examples=[2024])
    abstract: str = ""
    keywords: List[str] = Field(default_factory=list)
    pmcLink: str = ""


class ArticlePreview(_Wire):
    id: str
    title: str = ""
    snippet: str = ""
    confidenceScore: float = 0.0
    foundKeywords: List[str] = Field(default_factory=list)


class ExternalLink(_Wire):
    url: str
    provider: str = "PubMed Central"


class ChatRequest(_Wire):
    message: str = Field(..., examples=["How does it modulate JAK2 signaling?"])
    # Which research session the question is asked from. Optional for backward
    # compatibility, but send it: without it the article's thread is shared across
    # every session, so reopening the same paper in new work shows the old
    # conversation. `GET .../chat/history?sessionId=` filters by the same value.
    sessionId: Optional[str] = None


class ChatMessage(_Wire):
    role: Literal["user", "agent"]
    content: str = ""
    citations: List[str] = Field(default_factory=list)


# ── CuraTeX ──────────────────────────────────────────────────────────────────
#: Edited criterion values submitted with a profile. A criterion takes a number
#: (the ideal value), a `{"min": x, "max": y}` range, or a string (the preferred
#: category). Scoring normalises each candidate against these instead of the
#: range learned from the target's known ligands — before this existed only
#: `weights` were sent, so an edited value changed nothing and the UI had to
#: label it "for reference, not sent".
CriterionValues = Optional[Dict[str, Any]]


class TargetProfileRequest(_Wire):
    target: str = Field(..., examples=["JAK2"])
    sourceSessionId: Optional[str] = None
    # Optional but load-bearing: without a disease the repurposing exclusion
    # filter has nothing to exclude against, so the profile reports it inactive.
    disease: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    values: CriterionValues = Field(
        None, examples=[{"molecular_weight": {"min": 200, "max": 500}, "qed": 0.6}]
    )


class CurateCompoundsRequest(_Wire):
    target: str
    compound: Optional[str] = None
    numResults: int = Field(20, ge=15, le=50)
    disease: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    values: CriterionValues = Field(
        None, examples=[{"molecular_weight": {"min": 200, "max": 500}, "qed": 0.6}]
    )


class CompoundPage(_Wire):
    """A page of ranked CurateX candidates, matching the results table footer."""

    totalCompounds: int = 0
    page: int = 1
    totalPages: int = 1
    target: str = ""
    # Rows are passed through as the curation service shapes them — the column set
    # is driven by the active scoring criteria, so it is not fixed here.
    items: List[Dict[str, Any]] = Field(default_factory=list)


class TargetProfile(_Wire):
    """
    The editable Ideal Candidate Profile behind the CurateX profile screen.

    The class name is kept so the generated OpenAPI schema name does not change
    under clients already generated from it; the description is what the UI and
    the docs show.
    """

    target: str = ""
    profile: Dict[str, Any] = Field(default_factory=dict)
    criteria: List[Dict[str, Any]] = Field(default_factory=list)
    ligandCount: int = 0
    editable: bool = True
    warnings: List[str] = Field(default_factory=list)


# ── ScreenSuite ──────────────────────────────────────────────────────────────
class ScreenRequest(_Wire):
    target: str = Field(..., examples=["EGFR"])
    compoundLibrary: Optional[str] = None
    pdbFilePath: Optional[str] = None
    compounds: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Optional explicit ligands: [{drug_name, sdf_file_path}]",
    )


class ScreeningHit(_Wire):
    mode: Optional[int] = None
    compound: str = ""
    protein: str = ""
    affinityKcalPerMol: Optional[float] = None
    outputFile: str = ""


# ── NovSearch ────────────────────────────────────────────────────────────────
class NoveltyAssessRequest(_Wire):
    """
    Either NovSearch input shape (spec §2).

    Case A — a ScreenSuite carry-over: `target` / `drug` / `disease` plus the
    `candidateId` / `dockingId` that tie the report back to the screening result.
    Case B — a fresh query: `query` alone.
    """

    target: Optional[str] = Field(None, examples=["JAK2"])
    disease: Optional[str] = Field(None, examples=["Thrombocytosis"])
    drug: Optional[str] = Field(None, examples=["Fedratinib"])
    query: Optional[str] = Field(
        None, examples=["Is there existing patent coverage for JAK inhibitors?"]
    )
    candidateId: Optional[str] = Field(None, examples=["cand_0091"])
    dockingId: Optional[str] = Field(None, examples=["dock_0142"])
    numResults: int = Field(5, ge=1, le=20)


class NoveltyReport(_Wire):
    jobId: str
    query: str = ""
    inputSource: str = "user_direct"
    # Present only for a ScreenSuite carry-over, so a consumer can branch on
    # whether this report is tied to a specific screening result.
    candidateId: Optional[str] = None
    dockingId: Optional[str] = None
    target: str = ""
    disease: str = ""
    assessment: str = ""
    recommendations: List[str] = Field(default_factory=list)
    patents: List[Dict[str, Any]] = Field(default_factory=list)
    patentsUsed: List[str] = Field(default_factory=list)
    totalPatents: int = 0
    totalChunks: int = 0
    modelUsed: Optional[str] = None


class NoveltyQARequest(_Wire):
    """Follow-up question; `patentIds` selects single / multiple / all scope."""

    question: str = Field(..., examples=["Who owns US10123456B2?"])
    patentIds: Optional[List[str]] = None
    topK: Optional[int] = Field(None, ge=1, le=50)


class NoveltyQAResponse(_Wire):
    answer: str = ""
    mode: str = "multi_patent"
    patentIdsUsed: List[str] = Field(default_factory=list)
    chunksUsed: int = 0


# ── Projects ─────────────────────────────────────────────────────────────────
class Project(_Wire):
    id: str
    name: str
    disease: str = ""
    module: str = ""
    status: ProjectStatus = "Active"
    updatedAt: Optional[datetime] = None


class ProjectPage(_Wire):
    items: List[Project]
    totalCount: int
    page: int
    totalPages: int


class CreateProjectRequest(_Wire):
    name: str = Field(..., examples=["Type 2 Diabetes Target Analysis"])
    disease: str = Field("", examples=["Type 2 Diabetes"])
    module: str = Field("", examples=["TxKG"])
    status: ProjectStatus = "Active"


class UpdateProjectRequest(_Wire):
    """Rename a project or change its status. Every field is optional."""

    name: Optional[str] = None
    disease: Optional[str] = None
    module: Optional[str] = None
    status: Optional[ProjectStatus] = None


class ProjectItem(_Wire):
    id: str
    resultType: ResultType
    sessionId: Optional[str] = None
    resultId: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    createdAt: Optional[datetime] = None


class SaveResultRequest(_Wire):
    sessionId: Optional[str] = None
    resultId: str
    resultType: ResultType


class SaveResultResponse(_Wire):
    success: bool = True
    projectName: str = ""
    itemId: str = ""


# ── Files & Export ───────────────────────────────────────────────────────────
class UploadResponse(_Wire):
    fileId: str
    fileName: str
    sizeBytes: int = 0


class ExportRequest(_Wire):
    resultId: str
    format: ExportFormat = "json"


class PdfExportRequest(_Wire):
    resultId: str


class ExportResponse(_Wire):
    downloadUrl: str
