"""Agents - TxKG — /agents/txkg/*."""
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.jobs import create_job, enqueue, require_completed
from DRP_Main.app.models.user import User

router = APIRouter()
TAG = "Agents - TxKG"


@router.post(
    "/agents/txkg/query",
    response_model=s.JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[TAG],
)
def txkg_query(
    body: s.TxkgQueryRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Run a TxKG natural-language target discovery query."""
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="query cannot be empty")
    job = create_job(
        db,
        user_id=user.id,
        kind="txkg.query",
        module="TxKG",
        params={"query": body.query, "limit": body.limit, "maxHops": body.maxHops},
        project_id=body.projectId,
    )
    enqueue(job)
    return s.JobAccepted(jobId=job.id)


@router.get("/agents/txkg/targets", response_model=list[s.Target], tags=[TAG])
def txkg_targets(
    jobId: str = Query(..., description="A completed TxKG job"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """List ranked targets from a completed TxKG job."""
    job = require_completed(db, jobId, user.id)
    return [s.Target(**t) for t in (job.result or {}).get("targets", [])]


@router.get("/agents/txkg/targets/{uniprotId}", response_model=s.TargetDetail, tags=[TAG])
async def txkg_target_detail(
    uniprotId: str,
    jobId: str | None = Query(None, description="Optional job to source the score from"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get detail for a single target by UniProt ID."""
    from DRP_Main.app.drp.jobs import get_job

    score = None
    connection_types: list[str] = []
    name = ""
    if jobId:
        job = get_job(db, jobId, user.id)
        if job is not None:
            for target in (job.result or {}).get("targets", []):
                if target.get("uniprotId") == uniprotId:
                    score = target.get("score")
                    connection_types = target.get("connectionTypes", []) or []
                    name = target.get("name", "")
                    break

    try:
        from DRP_Main.app.modules.txkg.service import fetch_uniprot_name_enhanced

        info = await fetch_uniprot_name_enhanced(uniprotId) or {}
    except Exception as exc:  # noqa: BLE001 — TxKG data/deps may be absent
        raise HTTPException(status_code=502, detail=f"UniProt lookup unavailable: {exc}")

    gene = info.get("gene")
    full_name = info.get("full_name")
    return s.TargetDetail(
        uniprotId=uniprotId,
        name=name or full_name or gene or uniprotId,
        geneName=gene,
        fullName=full_name,
        organism=info.get("organism", "") or "",
        uniprotUrl=f"https://www.uniprot.org/uniprotkb/{uniprotId}/entry",
        score=score,
        connectionTypes=connection_types,
    )


@router.get("/agents/txkg/insights/{jobId}", response_model=s.InsightPanel, tags=[TAG])
def txkg_insights(
    jobId: str,
    tab: str = Query("interpretation", pattern="^(interpretation|recommendations|sources)$"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get the AI-generated insights panel (Interpretation / Recommendations / Sources)."""
    job = require_completed(db, jobId, user.id)
    result = job.result or {}
    targets = result.get("targets", [])

    if tab == "interpretation":
        return s.InsightPanel(tab="interpretation", content=result.get("interpretation", ""))

    if tab == "recommendations":
        # TxKG's §10 recommendation names the specific candidate(s) and the factors that
        # drove the suggestion; prefer it over a generic per-target template.
        recommendation = result.get("recommendation") or {}
        if recommendation.get("text"):
            items = [recommendation["text"]]
            drivers = recommendation.get("drivers") or []
            if drivers:
                items.append("Driven by: " + ", ".join(drivers))
            if recommendation.get("nextModule") or recommendation.get("next_module"):
                items.append(
                    "Suggested next module: "
                    f"{recommendation.get('nextModule') or recommendation.get('next_module')}"
                )
            return s.InsightPanel(
                tab="recommendations",
                content="Recommended next step based on corrected score, sourcing and novelty.",
                items=items,
            )

        items = [
            f"Validate {t.get('name') or t.get('uniprotId')} "
            f"({t.get('uniprotId')}, score {t.get('score')}) with a literature review — "
            f"run LitMineX on this target."
            for t in targets[:5]
        ]
        if targets:
            items.append(
                "Generate a knowledge sub-graph for the top targets to inspect the "
                "supporting pathway evidence."
            )
        return s.InsightPanel(
            tab="recommendations",
            content="Suggested next steps based on the ranked targets.",
            items=items,
        )

    sources = ["BioKG knowledge graph", "UniProt (protein naming and organism filter)"]
    method = result.get("method") or {}
    if method.get("correction_method"):
        sources.append(f"Degree correction: {method['correction_method']}")
    # Every distinct curated source backing a surfaced candidate's path.
    path_sources = sorted(
        {
            src
            for target in targets
            for src in (target.get("supportingSources") or [])
        }
    )
    sources.extend(path_sources)
    if result.get("interpretation"):
        sources.append("LLM interpretation grounded in the sourced paths and retrieved literature")

    if method:
        content = (
            "Targets derived from full-graph random-walk-with-restart propagation "
            f"(restart {method.get('restart_probability')}, no hop cutoff) over "
            f"{method.get('graph_nodes', '?')} nodes, corrected for degree bias."
        )
    else:
        content = f"Targets derived from {result.get('maxHopsSearched', 3)}-hop context-graph traversal."
    return s.InsightPanel(
        tab="sources",
        content=content,
        items=sources,
        links=_source_links(result, targets, path_sources),
    )


#: Curated databases the knowledge graph names as the origin of an edge. The graph
#: stores which database asserted a relationship but not that database's own record
#: id, so these link to the database rather than to the specific assertion.
_DATABASE_HOMES = {
    "ctd": ("Comparative Toxicogenomics Database", "https://ctdbase.org/"),
    "disgenet": ("DisGeNET", "https://www.disgenet.org/"),
    "drugbank": ("DrugBank", "https://go.drugbank.com/"),
    "kegg": ("KEGG", "https://www.genome.jp/kegg/"),
    "reactome": ("Reactome", "https://reactome.org/"),
    "go": ("Gene Ontology", "https://geneontology.org/"),
    "string": ("STRING", "https://string-db.org/"),
    "omim": ("OMIM", "https://www.omim.org/"),
    "uniprot": ("UniProt", "https://www.uniprot.org/"),
    "hpo": ("Human Phenotype Ontology", "https://hpo.jax.org/"),
    "intact": ("IntAct", "https://www.ebi.ac.uk/intact/"),
}


def _source_links(
    result: Dict[str, Any], targets: List[Dict[str, Any]], path_sources: List[str]
) -> List[s.SourceLink]:
    """
    Turn the source names into links, pointing at records wherever an id exists.

    Previously this tab returned bare strings, so the UI could only guess a
    homepage. Two identifiers are already in the result and go straight to the
    underlying data: the disease's MeSH id and each target's UniProt accession.
    """
    links: List[s.SourceLink] = []

    disease_id = (result.get("diseaseId") or "").strip()
    disease = result.get("disease") or disease_id
    if disease_id.startswith("D") and disease_id[1:].isdigit():
        links.append(
            s.SourceLink(
                name=f"MeSH: {disease}",
                url=f"https://meshb.nlm.nih.gov/record/ui?ui={disease_id}",
                kind="record",
                detail="The disease node this run was anchored on.",
            )
        )

    # One link per surfaced target, straight to its curated protein entry.
    for target in targets[:10]:
        accession = (target.get("uniprotId") or "").strip()
        if not accession:
            continue
        links.append(
            s.SourceLink(
                name=f"UniProt: {target.get('geneName') or target.get('name') or accession}",
                url=f"https://www.uniprot.org/uniprotkb/{accession}/entry",
                kind="record",
                detail=target.get("fullName") or "",
            )
        )

    for source in path_sources:
        name, url = _DATABASE_HOMES.get(
            source.strip().lower(), (source, "")
        )
        links.append(
            s.SourceLink(
                name=name,
                url=url,
                kind="database",
                detail="Asserted one or more edges on a connecting path.",
            )
        )

    if (result.get("method") or {}).get("correction_method"):
        links.append(
            s.SourceLink(
                name="BioKG knowledge graph",
                url="https://github.com/dsi-bdi/biokg",
                kind="method",
                detail=result["method"]["correction_method"],
            )
        )
    return links
