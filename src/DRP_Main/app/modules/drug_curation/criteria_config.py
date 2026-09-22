"""
CurateX §4.1 — the locked criteria table, default weights and categorical scoring maps.

Everything the profile scores on lives here so the weights can be spot-checked
against the spec in one place (the same reason LitMineX keeps `relation_config.py`).

19 weighted criteria across 4 categories summing to 100, plus one unweighted hard
exclusion filter (§5 step 7) and one unweighted display-only metadata field.

`CRITERIA` holds 20 rows, not 19: criterion 11 is scored as two independent rows
(11a withdrawn status, weight 10; 11b withdrawal reason, weight 4) because they
take different values from different lookup tables. The spec's count of 19 is a
count of criterion *numbers*, and the weights still sum to 100.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Criterion kinds ───────────────────────────────────────────────────────────
NUMERIC = "numeric"
CATEGORICAL = "categorical"
EXCLUSION = "exclusion"
METADATA = "metadata"

# Direction of desirability for numeric criteria. "lower" flips the normalized
# score (§4.2: `1 - normalized`).
HIGHER_IS_BETTER = "higher"
LOWER_IS_BETTER = "lower"

# Categories and their weight budgets (§4.1).
CATEGORY_BUDGETS: Dict[str, int] = {
    "Safety": 30,
    "Regulatory status": 25,
    "Physicochemical / drug-likeness": 30,
    "Dosage & PK/ADME": 15,
}


@dataclass(frozen=True)
class Criterion:
    """One row of the profile table."""

    key: str
    number: str                      # the spec's criterion number ("11a", "9", …)
    label: str
    category: str
    kind: str
    default_weight: int
    sources: Tuple[str, ...]
    direction: str = HIGHER_IS_BETTER
    unit: Optional[str] = None
    # Categorical criteria score off this map; the value is already 0-1.
    value_map: Dict[str, float] = field(default_factory=dict)
    # Fallback score when a categorical value is present but unmapped.
    unmapped_score: float = 0.5
    editable: bool = True
    note: str = ""


# ── §4.2 categorical lookup tables ────────────────────────────────────────────
# Keys are the normalized (lower-case, underscored) values the extractors emit.
_WITHDRAWN_STATUS = {"withdrawn": 0.0, "not_withdrawn": 1.0}

# The spec fixes this one explicitly: safety-related = 0, unknown = 0.5, commercial = 1.
_WITHDRAWAL_REASON = {
    "safety": 0.0,
    "efficacy": 0.25,
    "unknown": 0.5,
    "commercial": 1.0,
    # A drug that was never withdrawn has no reason to penalise.
    "not_withdrawn": 1.0,
}

_BLACK_BOX = {"yes": 0.0, "no": 1.0}

# Not fixed by the spec. An orphan designation is a regulatory asset, so its
# presence scores full; its absence is neutral rather than a defect, otherwise
# criterion 10's 7 points would penalise every non-orphan drug in the pool.
_ORPHAN = {"yes": 1.0, "no": 0.5}

_ORAL_ADMIN = {"yes": 1.0, "no": 0.0}

_DOSAGE_FORM = {
    "oral_solid": 1.0,      # tablet, capsule
    "oral_liquid": 0.85,    # solution, suspension, syrup
    "transdermal": 0.6,
    "topical": 0.55,
    "inhalation": 0.5,
    "injection": 0.4,       # parenteral of any route
    "implant": 0.3,
}

_DOSING_FREQUENCY = {
    "once_daily": 1.0,
    "twice_daily": 0.8,
    "three_times_daily": 0.6,
    "four_times_daily": 0.4,
    "as_needed": 0.5,
    "weekly_or_less": 0.9,
}

_METABOLISM = {
    "none": 1.0,                # not CYP-metabolised
    "single_non_3a4": 0.75,     # one CYP isoform, not 3A4
    "cyp3a4": 0.4,              # 3A4 involved — the busiest interaction hub
    "multiple": 0.5,            # several isoforms
    "unknown": 0.5,
}


# ── The 19 weighted criteria ──────────────────────────────────────────────────
CRITERIA: Tuple[Criterion, ...] = (
    # Safety — budget 30
    Criterion(
        key="withdrawn_status", number="11a", label="Withdrawn status",
        category="Safety", kind=CATEGORICAL, default_weight=10,
        sources=("chembl", "open_targets"), value_map=_WITHDRAWN_STATUS,
    ),
    Criterion(
        key="withdrawal_reason", number="11b", label="Withdrawal reason",
        category="Safety", kind=CATEGORICAL, default_weight=4,
        sources=("chembl", "open_targets"), value_map=_WITHDRAWAL_REASON,
    ),
    Criterion(
        key="black_box_warning", number="12", label="Black-box warning",
        category="Safety", kind=CATEGORICAL, default_weight=9,
        sources=("chembl", "open_targets", "dailymed"), value_map=_BLACK_BOX,
    ),
    Criterion(
        key="adverse_event_llr", number="14", label="Adverse event signal (LLR)",
        category="Safety", kind=NUMERIC, default_weight=7,
        sources=("open_targets",), direction=LOWER_IS_BETTER,
        note="Max FAERS log-likelihood ratio across reported adverse events.",
    ),
    # Regulatory status — budget 25
    Criterion(
        key="max_clinical_phase", number="9", label="Max clinical phase",
        category="Regulatory status", kind=NUMERIC, default_weight=18,
        sources=("chembl", "open_targets"), direction=HIGHER_IS_BETTER,
    ),
    Criterion(
        key="orphan_designation", number="10", label="Orphan designation",
        category="Regulatory status", kind=CATEGORICAL, default_weight=7,
        sources=("chembl",), value_map=_ORPHAN,
    ),
    # Physicochemical / drug-likeness — budget 30
    Criterion(
        key="qed", number="7", label="QED", category="Physicochemical / drug-likeness",
        kind=NUMERIC, default_weight=8, sources=("chembl",), direction=HIGHER_IS_BETTER,
    ),
    Criterion(
        key="ro5_violations", number="8", label="Ro5 violations",
        category="Physicochemical / drug-likeness", kind=NUMERIC, default_weight=5,
        sources=("chembl",), direction=LOWER_IS_BETTER,
    ),
    Criterion(
        key="molecular_weight", number="1", label="Molecular weight",
        category="Physicochemical / drug-likeness", kind=NUMERIC, default_weight=4,
        sources=("chembl",), unit="Da",
    ),
    Criterion(
        key="logp", number="2", label="LogP",
        category="Physicochemical / drug-likeness", kind=NUMERIC, default_weight=4,
        sources=("chembl",),
    ),
    Criterion(
        key="tpsa", number="3", label="TPSA",
        category="Physicochemical / drug-likeness", kind=NUMERIC, default_weight=3,
        sources=("chembl",), unit="Å²",
    ),
    Criterion(
        key="hbd", number="4", label="H-bond donors",
        category="Physicochemical / drug-likeness", kind=NUMERIC, default_weight=2,
        sources=("chembl",),
    ),
    Criterion(
        key="hba", number="5", label="H-bond acceptors",
        category="Physicochemical / drug-likeness", kind=NUMERIC, default_weight=2,
        sources=("chembl",),
    ),
    Criterion(
        key="rotatable_bonds", number="6", label="Rotatable bonds",
        category="Physicochemical / drug-likeness", kind=NUMERIC, default_weight=2,
        sources=("chembl",),
    ),
    # Dosage & PK/ADME — budget 15
    Criterion(
        key="oral_administration", number="13", label="Oral administration flag",
        category="Dosage & PK/ADME", kind=CATEGORICAL, default_weight=5,
        sources=("chembl",), value_map=_ORAL_ADMIN,
    ),
    Criterion(
        key="dosage_form", number="15", label="Dosage form",
        category="Dosage & PK/ADME", kind=CATEGORICAL, default_weight=2,
        sources=("dailymed",), value_map=_DOSAGE_FORM,
    ),
    Criterion(
        key="dosing_frequency", number="16", label="Dosing frequency",
        category="Dosage & PK/ADME", kind=CATEGORICAL, default_weight=2,
        sources=("dailymed",), value_map=_DOSING_FREQUENCY,
    ),
    Criterion(
        key="bioavailability", number="17", label="Absorption / bioavailability",
        category="Dosage & PK/ADME", kind=NUMERIC, default_weight=3,
        sources=("dailymed",), unit="%", direction=HIGHER_IS_BETTER,
    ),
    Criterion(
        key="half_life", number="18", label="Half-life",
        category="Dosage & PK/ADME", kind=NUMERIC, default_weight=2,
        sources=("dailymed",), unit="h",
    ),
    Criterion(
        key="metabolism_cyp", number="19", label="Metabolism (CYP pathway)",
        category="Dosage & PK/ADME", kind=CATEGORICAL, default_weight=1,
        sources=("dailymed",), value_map=_METABOLISM,
    ),
)

# ── Unweighted rows ───────────────────────────────────────────────────────────
EXCLUSION_CRITERION = Criterion(
    key="already_indicated", number="20", label="Already indicated for disease",
    category="Exclusion", kind=EXCLUSION, default_weight=0,
    sources=("open_targets",), editable=False,
    note="Hard filter applied in Tool 2 before scoring; never contributes a score.",
)

METADATA_CRITERION = Criterion(
    key="withdrawal_year", number="—", label="Withdrawal year",
    category="Metadata", kind=METADATA, default_weight=0,
    sources=("chembl", "open_targets"), editable=False,
    note="Display only.",
)

CRITERIA_BY_KEY: Dict[str, Criterion] = {c.key: c for c in CRITERIA}
ALL_ROWS: Tuple[Criterion, ...] = CRITERIA + (EXCLUSION_CRITERION, METADATA_CRITERION)

# Criteria whose values come from DailyMed label text — these are the ones that
# carry a coverage indicator (§4.3), because SPL extraction can legitimately miss.
DAILYMED_CRITERIA: Tuple[str, ...] = tuple(
    c.key for c in CRITERIA if c.sources == ("dailymed",)
)

DEFAULT_WEIGHTS: Dict[str, int] = {c.key: c.default_weight for c in CRITERIA}


def default_profile_weights() -> Dict[str, int]:
    """A fresh copy of the default weight map, safe for the caller to mutate."""
    return dict(DEFAULT_WEIGHTS)


def category_totals(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Sum the (possibly user-edited) weights per category."""
    active = weights or DEFAULT_WEIGHTS
    totals: Dict[str, float] = {name: 0.0 for name in CATEGORY_BUDGETS}
    for criterion in CRITERIA:
        totals[criterion.category] = totals.get(criterion.category, 0.0) + float(
            active.get(criterion.key, 0) or 0
        )
    return totals


def validate_weight_table(weights: Dict[str, float]) -> List[str]:
    """
    Check a user-edited weight table. Returns human-readable problems; an empty
    list means the table is usable.

    Weights are *not* forced back to 100 — §4.2's composite divides by the sum of
    the weights that actually had data, so any positive scale produces the same
    ranking. Only unknown keys and negatives are real errors.
    """
    problems: List[str] = []
    for key, value in weights.items():
        if key not in CRITERIA_BY_KEY:
            problems.append(f"Unknown criterion '{key}'")
            continue
        try:
            if float(value) < 0:
                problems.append(f"Weight for '{key}' is negative")
        except (TypeError, ValueError):
            problems.append(f"Weight for '{key}' is not a number")
    if weights and not any(
        float(v or 0) > 0 for k, v in weights.items() if k in CRITERIA_BY_KEY
    ):
        problems.append("At least one criterion must carry a positive weight")
    return problems
