"""
CurateX §2.2 #3 / §5 step 5 — DailyMed SPL enrichment.

DailyMed publishes FDA Structured Product Labels as free REST v2. Unlike ChEMBL
and Open Targets it exposes no clean numeric fields and no InChIKey, so this is:

  1. a *name/UNII* match against the deduped ligand set (there is no structure to
     join on), and
  2. a *text-parsing* step over label sections addressed by LOINC section code.

Both facts drive the design: a ligand with no matching SPL is flagged
`no label found` rather than silently skipped, and every extracted value carries
the snippet it came from so a user can check the parse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from xml.etree import ElementTree as ET

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.drug_curation.http_client import get_json, get_text

logger = get_logger(__name__)

DAILYMED_SPLS = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_SPL_XML = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"

# LOINC section codes named by §5 step 5.
LOINC_DOSAGE_AND_ADMINISTRATION = "34068-7"
LOINC_CLINICAL_PHARMACOLOGY = "34090-1"
LOINC_BOXED_WARNING = "34066-1"

_HL7 = "{urn:hl7-org:v3}"


@dataclass
class LabelExtract:
    """What one SPL yielded for one ligand."""

    drug_name: str
    setid: str = ""
    title: str = ""
    found: bool = False
    fields: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "drugName": self.drug_name,
            "setid": self.setid,
            "title": self.title,
            "found": self.found,
            "fields": dict(self.fields),
            "evidence": dict(self.evidence),
            "note": self.note,
        }


# ── SPL lookup ────────────────────────────────────────────────────────────────
def find_spl(drug_name: str, unii: str = "") -> Tuple[str, str]:
    """
    Resolve a ligand to an SPL setid by UNII first, then by name.

    UNII is an exact substance identifier; the name index matches brand and
    generic strings and will happily return a combination product for a
    single-ingredient query, so it is only the fallback.
    """
    if unii:
        payload = get_json(DAILYMED_SPLS, {"unii": unii, "pagesize": 5})
        rows = (payload or {}).get("data") or []
        if rows:
            return rows[0].get("setid", ""), rows[0].get("title", "")

    name = (drug_name or "").strip()
    if not name:
        return "", ""
    payload = get_json(DAILYMED_SPLS, {"drug_name": name, "pagesize": 10})
    rows = (payload or {}).get("data") or []
    if not rows:
        return "", ""
    lowered = name.lower()
    # Prefer a label whose title names the drug outright over a combination
    # product that merely contains it.
    exact = [r for r in rows if lowered in str(r.get("title", "")).lower()]
    chosen = (exact or rows)[0]
    return chosen.get("setid", ""), chosen.get("title", "")


def fetch_sections(setid: str) -> Dict[str, str]:
    """Pull an SPL and return {loinc_code: section_text} for the sections we parse."""
    if not setid:
        return {}
    xml = get_text(DAILYMED_SPL_XML.format(setid=setid))
    if not xml:
        return {}
    try:
        root = ET.fromstring(xml.encode("utf-8"))
    except ET.ParseError as exc:
        logger.warning("SPL %s did not parse: %s", setid, exc)
        return {}

    wanted = {
        LOINC_DOSAGE_AND_ADMINISTRATION,
        LOINC_CLINICAL_PHARMACOLOGY,
        LOINC_BOXED_WARNING,
    }
    sections: Dict[str, str] = {}
    for section in root.iter(f"{_HL7}section"):
        code_el = section.find(f"{_HL7}code")
        code = code_el.get("code") if code_el is not None else None
        if code not in wanted:
            continue
        text = " ".join(t.strip() for t in section.itertext() if t and t.strip())
        if text:
            # A drug may carry several subsections under one code; keep them all.
            sections[code] = (sections.get(code, "") + " " + text).strip()
    return sections


# ── Text extraction ───────────────────────────────────────────────────────────
_DOSAGE_FORMS = (
    ("oral_solid", r"\b(tablet|capsule|caplet|orally disintegrating|chewable)\b"),
    ("oral_liquid", r"\b(oral solution|oral suspension|syrup|elixir|oral liquid)\b"),
    ("transdermal", r"\b(transdermal|patch)\b"),
    ("inhalation", r"\b(inhalation|inhaler|nebuli[sz]|aerosol)\b"),
    ("injection", r"\b(injection|intravenous|subcutaneous|intramuscular|infusion|parenteral)\b"),
    ("implant", r"\b(implant|depot)\b"),
    ("topical", r"\b(topical|cream|ointment|gel|lotion)\b"),
)

_FREQUENCIES = (
    ("four_times_daily", r"\b(four times (a |per )?da(y|ily)|q\.?i\.?d\.?|every 6 hours)\b"),
    ("three_times_daily", r"\b(three times (a |per )?da(y|ily)|t\.?i\.?d\.?|every 8 hours)\b"),
    ("twice_daily", r"\b(twice (a |per )?da(y|ily)|b\.?i\.?d\.?|every 12 hours)\b"),
    ("weekly_or_less", r"\b(once (a |per )?week|weekly|every (2|two|4|four) weeks|monthly)\b"),
    ("once_daily", r"\b(once (a |per )?da(y|ily)|daily|q\.?d\.?|every 24 hours)\b"),
    ("as_needed", r"\b(as needed|p\.?r\.?n\.?)\b"),
)

_BIOAVAILABILITY = re.compile(
    r"(?:absolute\s+)?(?:oral\s+)?bioavailability[^.]{0,80}?(\d{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_ABSORPTION_PCT = re.compile(
    r"(?:absorbed|absorption)[^.]{0,60}?(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE
)

_HALF_LIFE = re.compile(
    r"half[-\s]?life[^.]{0,90}?(\d+(?:\.\d+)?)\s*(?:to|-|–)?\s*(\d+(?:\.\d+)?)?\s*"
    r"(hour|hr|h\b|minute|min\b|day)",
    re.IGNORECASE,
)

_CYP = re.compile(r"CYP\s?([1-4][A-Z]\d{1,2})", re.IGNORECASE)
_NOT_METABOLIZED = re.compile(
    r"(not (?:significantly |appreciably |extensively )?metaboli[sz]ed|"
    r"undergoes (?:minimal|negligible) metabolism|excreted unchanged)",
    re.IGNORECASE,
)


def _match_bucket(text: str, patterns) -> Tuple[str, str]:
    for key, pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            return key, _snippet(text, found.start())
    return "", ""


def _snippet(text: str, index: int, width: int = 110) -> str:
    start = max(0, index - width // 3)
    return " ".join(text[start : start + width].split())


def parse_dosage_section(text: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Criteria 15-16 from the Dosage and Administration section."""
    fields: Dict[str, Any] = {}
    evidence: Dict[str, str] = {}
    if not text:
        return fields, evidence

    form, form_snippet = _match_bucket(text, _DOSAGE_FORMS)
    if form:
        fields["dosage_form"] = form
        evidence["dosage_form"] = form_snippet

    frequency, freq_snippet = _match_bucket(text, _FREQUENCIES)
    if frequency:
        fields["dosing_frequency"] = frequency
        evidence["dosing_frequency"] = freq_snippet
    return fields, evidence


def parse_clinical_pharmacology(text: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Criteria 17-19 from the Clinical Pharmacology section."""
    fields: Dict[str, Any] = {}
    evidence: Dict[str, str] = {}
    if not text:
        return fields, evidence

    bioavailability = _BIOAVAILABILITY.search(text) or _ABSORPTION_PCT.search(text)
    if bioavailability:
        value = float(bioavailability.group(1))
        if 0 < value <= 100:
            fields["bioavailability"] = value
            evidence["bioavailability"] = _snippet(text, bioavailability.start())

    half_life = _HALF_LIFE.search(text)
    if half_life:
        low = float(half_life.group(1))
        high = float(half_life.group(2)) if half_life.group(2) else low
        unit = half_life.group(3).lower()
        hours = (low + high) / 2.0
        if unit.startswith("min"):
            hours /= 60.0
        elif unit.startswith("day"):
            hours *= 24.0
        fields["half_life"] = round(hours, 3)
        evidence["half_life"] = _snippet(text, half_life.start())

    if _NOT_METABOLIZED.search(text):
        fields["metabolism_cyp"] = "none"
        evidence["metabolism_cyp"] = _snippet(text, _NOT_METABOLIZED.search(text).start())
    else:
        isoforms = {m.group(1).upper() for m in _CYP.finditer(text)}
        if isoforms:
            if "3A4" in isoforms:
                bucket = "cyp3a4"
            elif len(isoforms) > 1:
                bucket = "multiple"
            else:
                bucket = "single_non_3a4"
            fields["metabolism_cyp"] = bucket
            evidence["metabolism_cyp"] = ", ".join(sorted(f"CYP{i}" for i in isoforms))
    return fields, evidence


def enrich(drug_name: str, unii: str = "") -> LabelExtract:
    """
    Full §5 step 5 for one ligand: match a label, fetch it, parse the two sections.

    Never raises. When no SPL matches, `found` stays False and `note` says so —
    that flag is what the §4.3 coverage indicator counts, so the gap is visible
    instead of looking like a drug with no dosing information.
    """
    name = (drug_name or "").strip()
    if not name and not unii:
        return LabelExtract(drug_name=name, note="no drug name to match")

    setid, title = find_spl(name, unii)
    if not setid:
        return LabelExtract(drug_name=name, note="no label found")

    sections = fetch_sections(setid)
    if not sections:
        return LabelExtract(
            drug_name=name, setid=setid, title=title, note="label found but sections unreadable"
        )

    fields, evidence = parse_dosage_section(sections.get(LOINC_DOSAGE_AND_ADMINISTRATION, ""))
    pk_fields, pk_evidence = parse_clinical_pharmacology(
        sections.get(LOINC_CLINICAL_PHARMACOLOGY, "")
    )
    fields.update(pk_fields)
    evidence.update(pk_evidence)

    # §2.2: DailyMed is the third cross-check for the black-box warning.
    if LOINC_BOXED_WARNING in sections:
        fields["black_box_warning"] = "yes"
        evidence["black_box_warning"] = "Boxed warning section present in the SPL"

    return LabelExtract(
        drug_name=name,
        setid=setid,
        title=title,
        found=True,
        fields=fields,
        evidence=evidence,
        note="" if fields else "label found but no parsable dosage/PK values",
    )


def enrich_many(names: List[str], max_workers: int = 6) -> Dict[str, LabelExtract]:
    """Enrich a whole (deduped) ligand set. Keyed by the name passed in."""
    from concurrent.futures import ThreadPoolExecutor

    unique = [n for n in dict.fromkeys(n.strip() for n in names if n and n.strip())]
    if not unique:
        return {}
    results: Dict[str, LabelExtract] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for name, extract in zip(unique, executor.map(enrich, unique)):
            results[name] = extract
    return results
