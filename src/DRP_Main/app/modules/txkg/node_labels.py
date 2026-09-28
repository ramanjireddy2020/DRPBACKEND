"""
Human-readable labels for knowledge-graph nodes.

BioKG names proteins, drugs and diseases, but leaves pathway and ontology nodes
as accessions — so a graph drew "hsa04630" and a connecting path read
"Thrombopoietin -Protein Pathway Association-> hsa04630" where a researcher
expects "JAK-STAT signaling pathway".

Three sources, in cost order:

  * KEGG      bundled (`kegg_pathways.py`), 372 human pathways, no network
  * Reactome  ContentService, resolved once per accession and cached
  * GO        the GO API, same

Misses are cached alongside hits, so an unreachable service costs one attempt
per accession for the life of the process rather than one per render. Every
lookup degrades to the accession: a label is a nicety, never a dependency.

`resolve` is deliberately tolerant about what it is given. BioKG hands back a
node's "name" that is often the accession again, and annotation nodes arrive
pre-labelled as "GO BP: GO:0005102" — a prefix that is also sometimes wrong,
because the aspect comes from whichever BioKG edge type introduced the term
first. Both forms are treated as unresolved so the real label wins.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Dict, Optional

from DRP_Main.app.modules.txkg.kegg_pathways import KEGG_PATHWAY_NAMES

#: Accessions this module knows how to resolve.
_KEGG_RE = re.compile(r"^hsa\d{5}$")
_REACTOME_RE = re.compile(r"^R-[A-Z]{3}-\d+$")
_GO_RE = re.compile(r"^GO:\d{7}$")
#: OMIM entries arrive either bare ("187950") or prefixed ("OMIM:187950").
_OMIM_RE = re.compile(r"^(?:OMIM:)?(\d{6})$")

#: An annotation label the graph applied on top of an accession, e.g.
#: "GO BP: GO:0005102" or "Pathway: R-HSA-76009". The accession inside is what
#: can actually be resolved.
_PREFIXED_RE = re.compile(
    r"^(?:GO (?:BP|MF|CC)|Pathway|Complex|Tissue|Cell|MeSH group|OMIM)\s*:\s*(\S+)$"
)

_REACTOME_API = "https://reactome.org/ContentService/data/query/"
_GO_API = "https://api.geneontology.org/api/ontology/term/"
_NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_TIMEOUT = 5.0

#: Reactome answers 403 to urllib's default agent, so every lookup identifies
#: itself. Harmless elsewhere and required here.
_USER_AGENT = "InnoDD-API/1.0 (drug repurposing platform; node label lookup)"

_cache: Dict[str, str] = {}


def _fetch_json(url: str) -> Optional[dict]:
    try:
        request = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode())
    except Exception:  # noqa: BLE001 — a label is never worth failing a request over
        return None


def _lookup(accession: str) -> str:
    """Resolve one accession, or "" — cached either way."""
    if accession in _cache:
        return _cache[accession]

    name = ""
    omim = _OMIM_RE.match(accession)
    if _KEGG_RE.match(accession):
        name = KEGG_PATHWAY_NAMES.get(accession, "")
    elif _REACTOME_RE.match(accession):
        body = _fetch_json(_REACTOME_API + accession)
        name = (body or {}).get("displayName") or ""
    elif _GO_RE.match(accession):
        body = _fetch_json(_GO_API + accession.replace(":", "%3A"))
        name = (body or {}).get("label") or ""
    elif omim:
        name = _omim_title(omim.group(1))

    _cache[accession] = name
    return name


def _omim_title(number: str) -> str:
    """
    An OMIM entry's title, via NCBI esummary.

    OMIM's own API needs a registered key; esummary does not, and returns the
    same title. OMIM writes titles as "THROMBOCYTHEMIA 1; THCYT1" — the part
    after the semicolon is the gene/phenotype symbol, which is noise on a graph
    node — and in block capitals, which reads as shouting next to "Thrombopoietin
    receptor". Both are tidied.
    """
    body = _fetch_json(_NCBI_ESUMMARY + f"?db=omim&id={number}&retmode=json")
    result = (body or {}).get("result") or {}
    for key, entry in result.items():
        if key == "uids" or not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        title = title.split(";")[0].strip()
        # Left alone if it is already mixed case — only OMIM's all-caps needs it.
        return title.title() if title.isupper() else title
    return ""


def looks_unresolved(node_id: str, name: Optional[str]) -> bool:
    """Whether `name` is really a label or just the accession wearing one."""
    if not name:
        return True
    if name == node_id:
        return True
    return bool(_PREFIXED_RE.match(name))


def resolve(node_id: str, name: Optional[str] = None) -> str:
    """
    The best available label for a node.

    Returns `name` untouched when it is already a real one. Otherwise resolves
    the accession — either `node_id` itself, or the one carried inside a
    prefixed label — and falls back to whatever was passed in.
    """
    if not looks_unresolved(node_id, name):
        return str(name)

    resolved = _lookup(node_id)
    if resolved:
        return resolved

    # "GO BP: GO:0005102" — the accession inside may differ from the node id.
    match = _PREFIXED_RE.match(name or "")
    if match:
        inner = _lookup(match.group(1))
        if inner:
            return inner

    return str(name or node_id)
