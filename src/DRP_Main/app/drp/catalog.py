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

MODULES: list[Module] = [
    Module(
        key="TxKG",
        displayName="TxKG Query",
        icon="network",
        description="Knowledge-graph target discovery — ranked protein targets, "
        "sub-graphs and meta-path reasoning for a disease.",
    ),
    Module(
        key="LitMineX",
        displayName="LitMineX",
        icon="book-open",
        description="Literature mining over PubMed with MeSH expansion and LLM "
        "relevance scoring per target.",
    ),
    Module(
        key="CurateX",
        displayName="CuraTeX",
        icon="flask",
        description="Compound curation and target candidate profiling with "
        "confidence scoring and PubMed evidence.",
    ),
    Module(
        key="ScreenSuite",
        displayName="ScreenSuite",
        icon="crosshair",
        description="Virtual / high-throughput screening — molecular docking of "
        "compound libraries against a receptor.",
    ),
    Module(
        key="NovSearch",
        displayName="NovSearch",
        icon="shield-check",
        description="Novelty and freedom-to-operate assessment over patent and "
        "literature corpora.",
    ),
    Module(
        key="SaaS Pipeline",
        displayName="SaaS Pipeline",
        icon="workflow",
        description="End-to-end repurposing pipeline chaining all five modules.",
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


AGENT_DISPLAY_NAMES = {
    "TxKG": "DRP TxKG Agent",
    "LitMineX": "DRP LitMineX Agent",
    "CurateX": "DRP CuraTeX Agent",
    "ScreenSuite": "DRP ScreenSuite Agent",
    "NovSearch": "DRP NovSearch Agent",
    "SaaS Pipeline": "DRP Pipeline Agent",
}
