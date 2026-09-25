"""
CurateX §4.2 / §4.3 — the scoring engine.

Numeric criteria are min-max normalized against the known-ligand profile's
baseline range and clipped to [0, 1]; inversely desirable criteria are flipped.
Categorical criteria score off the fixed lookup tables in `criteria_config`.

The composite divides by the sum of the weights of the criteria that actually had
data for that candidate. A criterion whose extraction failed is dropped from both
numerator and denominator — scoring it zero would punish a candidate for a gap in
DailyMed rather than for anything about the drug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from DRP_Main.app.modules.drug_curation.criteria_config import (
    CATEGORICAL,
    CRITERIA,
    CRITERIA_BY_KEY,
    DEFAULT_WEIGHTS,
    NUMERIC,
    LOWER_IS_BETTER,
    Criterion,
)


@dataclass
class CriterionBaseline:
    """Distribution of one criterion across the deduped, enriched ligand set (§5 step 6)."""

    key: str
    kind: str
    count: int = 0                       # ligands with a value
    total: int = 0                       # ligands considered
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    median_value: Optional[float] = None
    # Categorical criteria carry a value histogram instead of a range.
    distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return round(self.count / self.total, 3) if self.total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total": self.total,
            "coverage": self.coverage,
            "min": self.minimum,
            "max": self.maximum,
            "median": self.median_value,
            "distribution": dict(self.distribution) or None,
        }


def build_baselines(
    ligand_fields: Sequence[Dict[str, Any]],
) -> Dict[str, CriterionBaseline]:
    """§5 step 6 — per-criterion distribution stats over the ligand set."""
    total = len(ligand_fields)
    baselines: Dict[str, CriterionBaseline] = {}
    for criterion in CRITERIA:
        values = [
            row.get(criterion.key)
            for row in ligand_fields
            if row.get(criterion.key) is not None and row.get(criterion.key) != ""
        ]
        baseline = CriterionBaseline(key=criterion.key, kind=criterion.kind, total=total)
        if criterion.kind == NUMERIC:
            numbers = [_as_float(v) for v in values]
            numbers = [n for n in numbers if n is not None]
            baseline.count = len(numbers)
            if numbers:
                baseline.minimum = min(numbers)
                baseline.maximum = max(numbers)
                baseline.median_value = round(median(numbers), 4)
        else:
            texts = [str(v) for v in values]
            baseline.count = len(texts)
            for text in texts:
                baseline.distribution[text] = baseline.distribution.get(text, 0) + 1
        baselines[criterion.key] = baseline
    return baselines


def apply_value_overrides(
    baselines: Dict[str, CriterionBaseline],
    values: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Replace the learned "good" values for a criterion with the researcher's own.

    The profile screen let a user edit a criterion's value, but only weights were
    ever sent for scoring, so the UI had to label those edits "for reference, not
    sent" — the edit changed nothing. A baseline is what a candidate is scored
    against, so overriding it here is what makes an edited value take effect.

    Accepted per criterion:
      * a number                     — the ideal value
      * {"min": x, "max": y}         — the acceptable range
      * {"value": x} / {"target": x} — same as a bare number
      * a string                     — the preferred category

    Returns a note per applied override, for the profile's `warnings`, so a run
    scored against hand-entered values says so rather than looking learned.
    """
    notes: List[str] = []
    if not values:
        return notes

    for key, raw in values.items():
        baseline = baselines.get(key)
        if baseline is None or raw is None or raw == "":
            continue

        if isinstance(raw, dict):
            low = _as_float(raw.get("min"))
            high = _as_float(raw.get("max"))
            point = _as_float(raw.get("value", raw.get("target")))
        else:
            low = high = None
            point = _as_float(raw)

        if baseline.kind == NUMERIC:
            if low is None and high is None and point is None:
                continue
            if low is not None or high is not None:
                # A one-sided range keeps the observed bound on the other side,
                # so "at most 500 Da" does not silently also pin the minimum.
                baseline.minimum = low if low is not None else baseline.minimum
                baseline.maximum = high if high is not None else baseline.maximum
                mid = [v for v in (baseline.minimum, baseline.maximum) if v is not None]
                baseline.median_value = round(sum(mid) / len(mid), 4) if mid else None
                notes.append(f"{key}: scored against your range "
                             f"{baseline.minimum}-{baseline.maximum}.")
            else:
                baseline.minimum = baseline.maximum = baseline.median_value = point
                notes.append(f"{key}: scored against your value {point}.")
        else:
            text = str(raw.get("value", raw.get("target")) if isinstance(raw, dict) else raw)
            if not text:
                continue
            # A categorical baseline scores by how common a value is among the
            # ligands, so naming the preferred category means exactly that.
            baseline.distribution = {text: max(baseline.count, 1)}
            notes.append(f"{key}: scored against your preferred value '{text}'.")

    return notes


def normalize(criterion: Criterion, value: Any, baseline: Optional[CriterionBaseline]) -> Optional[float]:
    """
    Score one criterion for one candidate onto [0, 1], or None when unscoreable.

    None (not 0.0) is the "no data" signal — the composite must be able to tell a
    missing value from a bad one.
    """
    if value is None or value == "":
        return None

    if criterion.kind == CATEGORICAL:
        key = str(value).strip().lower().replace(" ", "_")
        if key in criterion.value_map:
            return criterion.value_map[key]
        return criterion.unmapped_score

    number = _as_float(value)
    if number is None:
        return None
    if baseline is None or baseline.minimum is None or baseline.maximum is None:
        return None

    span = baseline.maximum - baseline.minimum
    if span <= 0:
        # Every reference ligand shares one value: a candidate matching it is a
        # perfect match, anything else is not. Dividing by zero is not an option
        # and 0.5-for-everyone would waste the criterion's weight.
        return 1.0 if abs(number - baseline.minimum) < 1e-9 else 0.0

    normalized = (number - baseline.minimum) / span
    normalized = max(0.0, min(1.0, normalized))
    if criterion.direction == LOWER_IS_BETTER:
        normalized = 1.0 - normalized
    return normalized


@dataclass
class ScoredCriterion:
    key: str
    label: str
    category: str
    raw_value: Any
    normalized: Optional[float]
    weight: float
    weighted_score: Optional[float]
    source: str
    scored: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "value": self.raw_value,
            "normalized": None if self.normalized is None else round(self.normalized, 4),
            "weight": self.weight,
            "weightedScore": None if self.weighted_score is None else round(self.weighted_score, 4),
            "source": self.source,
            "scored": self.scored,
        }


@dataclass
class CandidateScore:
    composite: float
    breakdown: List[ScoredCriterion]
    weight_used: float
    weight_available: float
    missing: List[str]

    @property
    def data_completeness(self) -> float:
        """Share of the active weight that had data — §4.3's per-candidate figure."""
        return round(self.weight_used / self.weight_available, 3) if self.weight_available else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compositeScore": round(self.composite, 2),
            # `score` is the same number under the name every other module uses
            # (TxKG targets, LitMineX articles, ScreenSuite hits all carry `score`).
            # The results table was reading `score`, finding nothing and showing a
            # blank column while `compositeScore` sat beside it. Both are emitted:
            # nothing that already reads `compositeScore` has to change.
            "score": round(self.composite, 2),
            "breakdown": [row.as_dict() for row in self.breakdown],
            "weightUsed": round(self.weight_used, 2),
            "weightAvailable": round(self.weight_available, 2),
            "dataCompleteness": self.data_completeness,
            "missingCriteria": list(self.missing),
        }


def score_candidate(
    fields: Dict[str, Any],
    baselines: Dict[str, CriterionBaseline],
    weights: Optional[Dict[str, float]] = None,
    field_sources: Optional[Dict[str, str]] = None,
) -> CandidateScore:
    """
    §4.2 composite: Σ(normalized × weight) / Σ(weight of criteria with data) × 100.

    `weights` is the profile's current table — defaults on first load, whatever
    the user submitted afterwards. A criterion the user zeroed out drops out of
    both sums, which is how "exclude this criterion" is expressed.
    """
    active = weights or DEFAULT_WEIGHTS
    sources = field_sources or {}

    breakdown: List[ScoredCriterion] = []
    numerator = 0.0
    weight_used = 0.0
    weight_available = 0.0
    missing: List[str] = []

    for criterion in CRITERIA:
        weight = _as_float(active.get(criterion.key, 0)) or 0.0
        if weight > 0:
            weight_available += weight

        raw = fields.get(criterion.key)
        normalized = normalize(criterion, raw, baselines.get(criterion.key))
        scored = normalized is not None and weight > 0
        weighted = normalized * weight if scored else None

        if scored:
            numerator += weighted
            weight_used += weight
        elif weight > 0:
            missing.append(criterion.key)

        breakdown.append(
            ScoredCriterion(
                key=criterion.key,
                label=criterion.label,
                category=criterion.category,
                raw_value=raw,
                normalized=normalized,
                weight=weight,
                weighted_score=weighted,
                source=sources.get(criterion.key, ", ".join(criterion.sources)),
                scored=scored,
            )
        )

    composite = (numerator / weight_used * 100.0) if weight_used > 0 else 0.0
    breakdown.sort(key=lambda row: (row.weighted_score or -1.0), reverse=True)
    return CandidateScore(
        composite=composite,
        breakdown=breakdown,
        weight_used=weight_used,
        weight_available=weight_available,
        missing=missing,
    )


def score_drivers(score: CandidateScore, top_n: int = 3) -> List[str]:
    """The criteria that actually moved this candidate — used by §7 step 3."""
    drivers = [row for row in score.breakdown if row.scored and (row.weighted_score or 0) > 0]
    return [
        f"{row.label} ({_format_value(row.raw_value)})" for row in drivers[:top_n]
    ]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("_", " ")


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def criterion_for(key: str) -> Optional[Criterion]:
    return CRITERIA_BY_KEY.get(key)
