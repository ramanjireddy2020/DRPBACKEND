"""
Static catalogues served by the DRP `/v1` API: research modules, therapeutic
areas and composer quick actions.
"""
from DRP_Main.app.drp.schemas import Module, QuickAction

THERAPEUTIC_AREAS: list[str] = [
    "Oncology / Cancer",
    "Diabetes",
    "Cardiovascular",
    "Neurology / CNS",
    "Immunology & Inflammation",
    "Infectious Disease",
    "Respiratory",
    "Rare / Orphan Disease",
    "Metabolic Disorders",
    "Dermatology",
    "Ophthalmology",
    "Nephrology",
]

#: The five user-facing agents, in pipeline order.
#:
#: "SaaS Pipeline" is deliberately absent. It is still a valid `ModuleKey` and
#: still has a runner, so sessions and jobs that already reference it keep
#: working — it is simply not offered. Listing it put a sixth "agent" in the
#: supervisor's menu that is an orchestration mode, not an agent, and choosing
#: it from that menu returned 422.
#:
#: `displayName` is the agreed user-facing agent name. It was previously the
#: internal key with drifted casing ("TxKG Query", "CuraTeX"), which is what the
#: UI rendered, so the platform called each module three different things.
MODULES: list[Module] = [
    Module(
        key="TxKG",
        displayName="Target Identification Agent",
        icon="network",
        description="Knowledge-graph target discovery — ranked protein targets, "
        "sub-graphs and meta-path reasoning for a disease.",
    ),
    Module(
        key="LitMineX",
        displayName="Literature Mining Agent",
        icon="book-open",
        description="Literature mining over PubMed with MeSH expansion and LLM "
        "relevance scoring per target.",
    ),
    Module(
        key="CurateX",
        displayName="Drug Curation Agent",
        icon="flask",
        description="Builds an Ideal Candidate Profile from known ligands, then "
        "ranks repurposing candidates against it.",
    ),
    Module(
        key="ScreenSuite",
        displayName="Virtual Screening Agent",
        icon="crosshair",
        description="Virtual / high-throughput screening — molecular docking of "
        "selected compounds against a selected protein structure.",
    ),
    Module(
        key="NovSearch",
        displayName="Novelty Search Agent",
        icon="shield-check",
        description="Novelty and freedom-to-operate assessment over patent and "
        "literature corpora.",
    ),
]

QUICK_ACTIONS: list[QuickAction] = [
    QuickAction(
        label="Find protein targets for T2D",
        module="TxKG",
        prefillQuery="Find protein targets associated with Type 2 Diabetes for drug repurposing",
    ),
    QuickAction(
        label="Screen compounds for EGFR",
        module="ScreenSuite",
        prefillQuery="Screen the repurposing compound library against EGFR",
    ),
    QuickAction(
        label="Mine literature for JAK2",
        module="LitMineX",
        prefillQuery="Mine literature for JAK2 drug targets with confidence scoring",
    ),
    QuickAction(
        label="Curate compounds for Aspirin",
        module="CurateX",
        prefillQuery="Curate compounds related to Aspirin for anti-inflammatory repurposing",
    ),
    QuickAction(
        label="Assess novelty of HER2 in breast cancer",
        module="NovSearch",
        prefillQuery="Assess patent novelty for HER2 in Breast Cancer",
    ),
]

# Accepts the spelling variants the spec uses across paths and enums.
_MODULE_ALIASES = {
    "txkg": "TxKG",
    "litminex": "LitMineX",
    "litminex_": "LitMineX",
    "litminx": "LitMineX",
    "curatex": "CurateX",
    "curatex_": "CurateX",
    "screensuite": "ScreenSuite",
    "novsearch": "NovSearch",
    "novelty": "NovSearch",
    "saas pipeline": "SaaS Pipeline",
    "saas": "SaaS Pipeline",
}


def canonical_module(name: str | None) -> str:
    """Normalise a module name ('litminex', 'LitMinex', 'CuraTeX') to its key."""
    if not name:
        return ""
    return _MODULE_ALIASES.get(name.strip().lower(), name.strip())


#: Message authorship, as it appears next to a chat bubble. Keyed by module key,
#: including "SaaS Pipeline" so historical messages still resolve an author.
AGENT_DISPLAY_NAMES = {
    "TxKG": "DRP TxKG Agent",
    "LitMineX": "DRP LitMineX Agent",
    "CurateX": "DRP CurateX Agent",
    "ScreenSuite": "DRP ScreenSuite Agent",
    "NovSearch": "DRP NovSearch Agent",
    "SaaS Pipeline": "DRP Pipeline Agent",
}
