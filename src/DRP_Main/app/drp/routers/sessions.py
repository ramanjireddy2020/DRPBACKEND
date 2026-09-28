"""
Sessions — /sessions, /sessions/{sessionId}, /sessions/draft, and the step chain.

A session is a chain of module runs, not a single one. The researcher submits a
query (step 1), reviews the results, ticks what matters, and advances to the next
module carrying those selections forward. Each step owns its own job, status,
summary and chat turns; the session's own `module`/`status`/`summary` are
denormalised headline fields kept in sync for the sessions list.
"""
import math
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session as OrmSession

from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.catalog import AGENT_DISPLAY_NAMES, MODULES, canonical_module
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.dispatch import JOB_KIND_BY_MODULE, infer_module, start_module_job
from DRP_Main.app.drp.models import (
    DrpJob,
    DrpSession,
    DrpSessionMessage,
    DrpSessionStep,
    utcnow,
)
from DRP_Main.app.models.user import User

logger = get_logger(__name__)
router = APIRouter()
TAG = "Sessions"
PAGE_SIZE = 20


# ── helpers ──────────────────────────────────────────────────────────────────
def _add_message(
    db: OrmSession,
    session_id: str,
    role: str,
    content: str,
    agent_name: str = "",
    step_id: Optional[str] = None,
) -> DrpSessionMessage:
    message = DrpSessionMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        step_id=step_id,
        role=role,
        agent_name=agent_name,
        content=content,
    )
    db.add(message)
    return message


def _create_step(
    db: OrmSession,
    session: DrpSession,
    *,
    module: str,
    query: str,
    selections: Optional[Dict[str, Any]] = None,
    parent_step_id: Optional[str] = None,
    rerun_of_step_id: Optional[str] = None,
    branch_name: str = "",
) -> DrpSessionStep:
    """Append a step to a session's chain. Does not start its job."""
    step = DrpSessionStep(
        id=f"stp_{uuid.uuid4().hex[:12]}",
        session_id=session.id,
        parent_step_id=parent_step_id,
        step_index=len(session.steps),
        module=module,
        status="In Progress",
        summary="",
        query=query,
        selections=selections or {},
        rerun_of_step_id=rerun_of_step_id,
        branch_name=branch_name,
    )
    db.add(step)
    return step


def _branch_of(step: Optional[DrpSessionStep]) -> str:
    """A step's branch. "" is main."""
    return (step.branch_name or "") if step is not None else ""


def _latest_step(
    session: DrpSession, branch: Optional[str] = None
) -> Optional[DrpSessionStep]:
    """
    The newest step on one branch — the session's active branch by default.

    `session.steps` is a single list across every branch, so returning its last
    entry made the newest step *anywhere* the current one. After branching, a
    follow-up on main was answered from the branch's results and a branch
    replayed main's; CurateX re-used main's profile on a branch for the same
    reason. Scoping by branch is what keeps the two apart.

    Falls back to the whole chain when the named branch has no steps yet, so a
    branch created but not yet run still answers from where it forked.
    """
    if not session.steps:
        return None
    wanted = session.active_branch or "" if branch is None else (branch or "")
    on_branch = [st for st in session.steps if _branch_of(st) == wanted]
    if on_branch:
        return on_branch[-1]
    # A branch with no steps of its own inherits the step it forked from.
    return session.steps[-1] if not wanted else _forked_from(session, wanted)


def _forked_from(session: DrpSession, branch: str) -> Optional[DrpSessionStep]:
    """The step a branch forked from, for a branch that has not run yet."""
    by_id = {st.id: st for st in session.steps}
    for step in session.steps:
        if _branch_of(step) == branch and step.parent_step_id:
            return by_id.get(step.parent_step_id)
    return None


def _sync_step(db: OrmSession, step: DrpSessionStep) -> DrpSessionStep:
    """Fold this step's finished job into its status, summary and chat thread."""
    if not step.job_id:
        return step
    job = db.query(DrpJob).filter(DrpJob.id == step.job_id).first()
    if job is None or job.status in ("queued", "running"):
        return step

    agent_name = AGENT_DISPLAY_NAMES.get(step.module, "DRP Agent")
    already_replied = any(m.role == "agent" and m.content for m in step.messages)

    if job.status == "completed":
        summary = (job.result or {}).get("summary") or "Analysis complete"
        step.status = "Completed"
        step.summary = summary
        if not already_replied:
            interpretation = (job.result or {}).get("interpretation") or summary
            _add_message(db, step.session_id, "agent", interpretation, agent_name, step.id)
    else:
        step.status = "Failed"
        step.summary = job.error or "Agent run failed"
        if not already_replied:
            _add_message(db, step.session_id, "agent", step.summary, agent_name, step.id)

    step.updated_at = utcnow()
    return step


def _sync_session(db: OrmSession, session: DrpSession) -> DrpSession:
    """
    Sync every step from its own job, then roll the headline fields up.

    Each step is synced from *its own* job. The previous implementation read only
    the newest job for the whole session, which meant a multi-step session
    reported one step's outcome as if it were the session's.
    """
    dirty = False
    for step in session.steps:
        before = (step.status, step.summary)
        _sync_step(db, step)
        dirty = dirty or before != (step.status, step.summary)

    latest = _latest_step(session)
    if latest is not None:
        session.module = latest.module or session.module
        session.summary = latest.summary or session.summary
        # "Saved" is a researcher's explicit choice and outranks anything derived.
        if session.status != "Saved":
            if all(st.status == "Completed" for st in session.steps):
                session.status = "Completed"
            elif latest.status == "Failed":
                # A failed run is finished, not ongoing. Reporting it as "In
                # Progress" left dead sessions sitting at the top of the list
                # looking like work still running.
                session.status = "Completed"
            else:
                session.status = "In Progress"
    elif not session.steps:
        # A session that never started a step — the researcher's input was not
        # understood and they were asked what they meant. Nothing is running, so
        # it should not claim to be.
        if session.status != "Saved":
            session.status = "Completed"

    if dirty:
        session.updated_at = utcnow()
        db.commit()
        db.refresh(session)
    return session


def _step_to_wire(step: DrpSessionStep) -> s.SessionStep:
    return s.SessionStep(
        id=step.id,
        stepIndex=step.step_index or 0,
        module=step.module or "",
        status=step.status or "In Progress",
        summary=step.summary or "",
        jobId=step.job_id,
        selections=step.selections or {},
        parentStepId=step.parent_step_id,
        rerunOfStepId=step.rerun_of_step_id,
        branchName=step.branch_name or "",
        updatedAt=step.updated_at,
    )


def _to_wire(session: DrpSession) -> s.Session:
    latest = _latest_step(session)
    return s.Session(
        id=session.id,
        module=session.module or "",
        title=session.title or "",
        status=session.status or "In Progress",
        summary=session.summary or "",
        updatedAt=session.updated_at,
        messages=[
            s.SessionMessage(
                role=m.role,
                agentName=m.agent_name or "",
                content=m.content or "",
                stepId=m.step_id,
            )
            for m in session.messages
        ],
        steps=[_step_to_wire(st) for st in session.steps],
        jobId=latest.job_id if latest else None,
        currentStepId=latest.id if latest else None,
        activeBranch=session.active_branch or "",
    )


def _owned_session(db: OrmSession, session_id: str, user: User) -> DrpSession:
    session = (
        db.query(DrpSession)
        .filter(DrpSession.id == session_id, DrpSession.user_id == user.id)
        .first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


# ── list / create ────────────────────────────────────────────────────────────
@router.get("/sessions", response_model=s.SessionPage, tags=[TAG])
def list_sessions(
    search: Optional[str] = Query(None, description="Search by target, disease, or module"),
    module: Optional[str] = Query(
        None, description="All | TxKG | LitMineX | ScreenSuite | CurateX | NovSearch"
    ),
    status_filter: Optional[str] = Query(
        None, alias="status", description="Completed | In Progress | Saved"
    ),
    page: int = Query(1, ge=1),
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """List / search recent research sessions."""
    query = db.query(DrpSession).filter(
        DrpSession.user_id == user.id, DrpSession.is_draft == False  # noqa: E712
    )
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                DrpSession.title.ilike(like),
                DrpSession.query.ilike(like),
                DrpSession.module.ilike(like),
                DrpSession.summary.ilike(like),
            )
        )
    if module and module.lower() != "all":
        # Match any step's module, not just the headline one: a session that ran
        # TxKG then LitMineX belongs under both filters.
        wanted = canonical_module(module)
        query = query.filter(
            or_(
                DrpSession.module == wanted,
                DrpSession.steps.any(DrpSessionStep.module == wanted),
            )
        )
    if status_filter:
        query = query.filter(DrpSession.status == status_filter)

    total = query.count()
    rows = (
        query.order_by(DrpSession.updated_at.desc())
        .offset((page - 1) * PAGE_SIZE)
        .limit(PAGE_SIZE)
        .all()
    )
    rows = [_sync_session(db, row) for row in rows]
    return s.SessionPage(
        items=[_to_wire(row) for row in rows],
        page=page,
        totalPages=max(1, math.ceil(total / PAGE_SIZE)),
    )


@router.post(
    "/sessions", response_model=s.Session, status_code=status.HTTP_201_CREATED, tags=[TAG]
)
def create_session(
    body: s.CreateSessionRequest,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """
    Create a new research session (submitting the composer query).

    Opens the chain's first step and starts its agent job. The response carries
    `jobId` and `currentStepId` — poll `/agents/jobs/{jobId}/status` for progress.
    """
    if not (body.query or "").strip():
        raise HTTPException(status_code=422, detail="query cannot be empty")

    module = infer_module(body.query, body.module)
    session = DrpSession(
        id=f"ses_{uuid.uuid4().hex[:16]}",
        user_id=user.id,
        module=module or "",
        title=(body.query or "").strip()[:120],
        status="In Progress",
        summary="",
        query=body.query,
        project_id=body.projectId,
        is_draft=False,
    )
    db.add(session)
    db.flush()  # session.id must exist before the step and message reference it

    if module is None:
        # Nothing in the query names a module or reads as one of their jobs.
        # Guessing here is worse than asking: the old fallback sent "open
        # report.pdf" to TxKG, which fuzzy-matched it to "Bone Resorption" and
        # propagated a knowledge graph for a disease the researcher never named.
        # The session is kept so the reply lands in a thread they can answer in.
        _add_message(db, session.id, "user", body.query)
        _add_message(db, session.id, "agent", _clarify_message(), "DRP Supervisor")
        session.status = "In Progress"
        db.commit()
        db.refresh(session)
        return _to_wire(session)

    step = _create_step(db, session, module=module, query=body.query)
    db.flush()
    _add_message(db, session.id, "user", body.query, step_id=step.id)
    db.commit()

    try:
        job = start_module_job(
            db,
            user_id=user.id,
            module=module,
            query=body.query,
            session_id=session.id,
            project_id=body.projectId,
            session_step_id=step.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    step.job_id = job.id
    db.commit()
    db.refresh(session)
    return _to_wire(session)


# ── the step chain ───────────────────────────────────────────────────────────
@router.post(
    "/sessions/{sessionId}/steps",
    response_model=s.SessionStep,
    status_code=status.HTTP_201_CREATED,
    tags=[TAG],
)
def create_step(
    sessionId: str,
    body: s.CreateStepRequest,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """
    Advance a session to its next module, carrying the researcher's selections.

    This is the hand-off behind "Continue" after target selection and behind
    CurateX's "View in ScreenSuite". Naming `fromStepId` as an earlier step forks
    the chain there instead of extending it — that is what "Branch" does.
    """
    session = _owned_session(db, sessionId, user)

    module = canonical_module(body.module)
    if module not in JOB_KIND_BY_MODULE:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown module '{body.module}'. Valid: {sorted(JOB_KIND_BY_MODULE)}",
        )

    parent = _latest_step(session)
    if body.fromStepId:
        parent = next((st for st in session.steps if st.id == body.fromStepId), None)
        if parent is None:
            raise HTTPException(
                status_code=404, detail=f"Step '{body.fromStepId}' not found in this session"
            )

    # An explicit branchName forks; otherwise the step continues whichever branch
    # its parent is on. Inheriting matters as much as forking: without it every
    # hand-off after a branch silently landed back on main.
    branch = body.branchName or _branch_of(parent)

    query = body.query or session.query or ""
    step = _create_step(
        db,
        session,
        module=module,
        query=query,
        selections=body.selections,
        parent_step_id=parent.id if parent else None,
        branch_name=branch,
    )
    db.flush()
    session.active_branch = branch

    try:
        job = start_module_job(
            db,
            user_id=user.id,
            module=module,
            query=query,
            session_id=session.id,
            project_id=session.project_id,
            session_step_id=step.id,
            selections=body.selections,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    step.job_id = job.id
    session.module = module
    session.status = "In Progress" if session.status != "Saved" else session.status
    session.updated_at = utcnow()
    db.commit()
    db.refresh(step)
    return _step_to_wire(step)


@router.post(
    "/sessions/{sessionId}/steps/{stepId}/rerun",
    response_model=s.SessionStep,
    status_code=status.HTTP_201_CREATED,
    tags=[TAG],
)
def rerun_step(
    sessionId: str,
    stepId: str,
    body: s.RerunStepRequest | None = None,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """
    Re-run a step, optionally with adjusted parameters (the "Rerun" button).

    The original step is kept — a re-run is a new step pointing back at it via
    `rerunOfStepId`, so the researcher can compare attempts rather than lose the
    first one.
    """
    session = _owned_session(db, sessionId, user)
    original = next((st for st in session.steps if st.id == stepId), None)
    if original is None:
        raise HTTPException(status_code=404, detail=f"Step '{stepId}' not found in this session")

    body = body or s.RerunStepRequest()
    step = _create_step(
        db,
        session,
        module=original.module,
        query=original.query or session.query or "",
        selections=original.selections or {},
        parent_step_id=original.parent_step_id,
        rerun_of_step_id=original.id,
        branch_name=original.branch_name or "",
    )
    db.flush()

    try:
        job = start_module_job(
            db,
            user_id=user.id,
            module=original.module,
            query=step.query,
            session_id=session.id,
            project_id=session.project_id,
            session_step_id=step.id,
            selections=original.selections or {},
            params_override=body.params or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    step.job_id = job.id
    session.updated_at = utcnow()
    db.commit()
    db.refresh(step)
    return _step_to_wire(step)


# ── chat ─────────────────────────────────────────────────────────────────────
@router.post(
    "/sessions/{sessionId}/messages",
    response_model=s.SessionMessageResponse,
    tags=[TAG],
)
def post_message(
    sessionId: str,
    body: s.SessionMessageRequest,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """
    Send a chat turn into a session and get the agent's reply.

    Routing deliberately differs from the composer's: a follow-up is answered in
    the context of the step the researcher is looking at, and only an explicit
    `@module` mention moves the conversation to a different agent. Re-inferring
    from keywords here would send "what about the patents on these?" to NovSearch
    in the middle of reading TxKG results.
    """
    if not (body.message or "").strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    session = _owned_session(db, sessionId, user)
    _sync_session(db, session)

    # `branchName` says which branch the researcher is looking at; `stepId` names
    # a step directly and implies its branch. Either one switches the session's
    # active branch, so switching back to main in the UI answers from main.
    if body.branchName is not None:
        session.active_branch = body.branchName or ""

    step = _latest_step(session)
    if body.stepId:
        step = next((st for st in session.steps if st.id == body.stepId), None)
        if step is None:
            raise HTTPException(status_code=404, detail=f"Step '{body.stepId}' not found")
        session.active_branch = _branch_of(step)

    _add_message(db, session.id, "user", body.message, step_id=step.id if step else None)
    db.commit()

    # An @mention moves the conversation to another agent; so does an unambiguous
    # instruction to run one ("create a drug profile for JAK2"), which previously
    # got answered out of the current module's results instead of starting CurateX.
    # `explicit_module_request` requires an action verb and that module's object,
    # so a question about the results on screen still stays with this agent.
    from DRP_Main.app.drp.dispatch import explicit_module_request, extract_module_mention

    mentioned = extract_module_mention(body.message) or explicit_module_request(body.message)
    # A request to run a module starts it even when that module is the current
    # step. The previous `mentioned != step.module` guard meant a second
    # "create a drug profile for JAK2" was answered out of the first run's
    # results — the one module you could never re-run was the one you were
    # looking at, and re-running with a different target is the normal way to
    # use CurateX. The earlier step and its results stay in the chain, so the
    # conversation keeps both.
    if mentioned:
        carried = _carry_forward(step, body.message)
        new_step = _create_step(
            db,
            session,
            module=mentioned,
            query=body.message,
            selections=carried,
            parent_step_id=step.id if step else None,
            # Stay on the branch the researcher is on — a run started from a
            # branch belongs to it, not to main.
            branch_name=_branch_of(step),
        )
        db.flush()
        job = start_module_job(
            db,
            user_id=user.id,
            module=mentioned,
            query=body.message,
            session_id=session.id,
            project_id=session.project_id,
            session_step_id=new_step.id,
            selections=carried,
        )
        new_step.job_id = job.id
        session.module = mentioned
        session.updated_at = utcnow()
        db.commit()
        return s.SessionMessageResponse(
            role="agent",
            agentName=AGENT_DISPLAY_NAMES.get(mentioned, "DRP Agent"),
            content=f"Starting a {mentioned} run for that.",
            stepId=new_step.id,
            jobId=job.id,
        )

    answer = _answer_follow_up(db, step, body.message)
    agent_name = AGENT_DISPLAY_NAMES.get(step.module if step else "", "DRP Agent")
    _add_message(db, session.id, "agent", answer, agent_name, step.id if step else None)
    session.updated_at = utcnow()
    db.commit()
    return s.SessionMessageResponse(
        role="agent",
        agentName=agent_name,
        content=answer,
        stepId=step.id if step else None,
        jobId=None,
    )


def _carry_forward(step: Optional[DrpSessionStep], message: str) -> Dict[str, Any]:
    """
    The previous step's selections, minus any target the new message overrides.

    Selections normally carry a researcher's ticked choice forward, and that is
    right for a hand-off. But a message that names its own target is a *new*
    instruction: "create a drug profile for STAT3" after a JAK2 run must profile
    STAT3. Left alone, the carried target would win and the module would re-run
    on JAK2 — re-running with a different target being the main reason to ask
    twice in the first place.
    """
    from DRP_Main.app.drp.dispatch import extract_gene_symbol

    carried = dict((step.selections or {}) if step else {})
    if extract_gene_symbol(message):
        for key in ("target", "targetIds", "targets", "selectedTargets"):
            carried.pop(key, None)
    return carried


def _clarify_message() -> str:
    """
    Asked when a query names no module and reads as none of their jobs.

    Names the modules rather than saying "I don't understand": a researcher who
    has just been told their input was not recognised needs to know what the
    platform can actually do, not to guess again.
    """
    lines = "\n".join(f"  • {m.key} — {m.description}" for m in MODULES)
    return (
        "I'm not sure what you'd like me to run. Here is what I can do:\n\n"
        f"{lines}\n\n"
        "Tell me the disease or target you're interested in — for example "
        '"find protein targets for thrombocytosis" — or start a module directly '
        "by typing @ and its name."
    )


def _module_menu() -> str:
    """
    The real module catalogue, for the follow-up prompt.

    Without it the model was told to "suggest which module would" answer the
    question while never being shown which modules exist, so it invented
    plausible ones — "Dictionary", "TargetFinder", "GeneWiki" — and researchers
    went looking for features that were never built. The list is derived from
    `MODULES` rather than written out here so a new module cannot go missing.
    """
    lines = "\n".join(f"  - {m.key}: {m.description}" for m in MODULES)
    return (
        "These are the ONLY modules that exist on this platform:\n"
        f"{lines}\n\n"
        "Never name a module outside this list — not even a plausible-sounding one, "
        "and never a database or website as though it were a module. If one of these "
        "would answer the question, name it exactly as written above and say it can "
        "be started by typing @<module> with the request. If none of them would, say "
        "so plainly and stop; do not invent a destination.\n\n"
        "Write to the person asking, as \"you\". They are reading this in a chat they "
        "are part of, so \"the researcher can...\" reads as if they were being "
        "discussed rather than answered."
    )


def _step_context(db: OrmSession, step: DrpSessionStep) -> Optional[Dict[str, Any]]:
    """One completed step's results, trimmed to what an answer can use."""
    if not step.job_id:
        return None
    job = db.query(DrpJob).filter(DrpJob.id == step.job_id).first()
    if job is None or job.status != "completed":
        return None

    result = job.result or {}
    context: Dict[str, Any] = {
        "module": step.module,
        "summary": result.get("summary", ""),
        "interpretation": result.get("interpretation", ""),
        "recommendation": result.get("recommendation", {}),
    }
    # Only the ranked head of a result set is worth sending; the full payload can
    # be hundreds of rows and the answer never needs the tail.
    for key in ("targets", "articles", "compounds", "hits", "patents"):
        if isinstance(result.get(key), list) and result[key]:
            context[key] = result[key][:10]
            break
    return context


def _answer_follow_up(db: OrmSession, step: Optional[DrpSessionStep], question: str) -> str:
    """
    Answer a follow-up from the branch's results rather than re-running anything.

    Every completed step on the current branch is in scope, not just the newest.
    "Why is JAK2 ranked higher?" was answered from the LitMineX step alone — "the
    results do not provide a specific explanation" — while the TxKG step two
    positions back held the ranking and the reason for it. A researcher asking
    about a session means the session, not whichever module happens to be on
    screen. Oldest first, so the newest results are nearest the question.

    Degrades to a plain message when no LLM is configured, so chat never
    hard-fails a session.
    """
    if step is None or not step.job_id:
        return "Ask me about the results once a module has finished running."

    job = db.query(DrpJob).filter(DrpJob.id == step.job_id).first()
    if job is None or job.status != "completed":
        return "That step is still running — I'll be able to answer once it finishes."

    session = step.session
    branch = _branch_of(step)
    earlier = [
        st for st in (session.steps if session is not None else [])
        if _branch_of(st) == branch and st.id != step.id
    ]
    history = [ctx for ctx in (_step_context(db, st) for st in earlier) if ctx]

    current = _step_context(db, step) or {"module": step.module}
    context: Dict[str, Any] = dict(current)
    if history:
        # Capped: a five-module session would otherwise send every result set.
        context["earlierSteps"] = history[-4:]

    try:
        import json

        from DRP_Main.app.core.llm import llm_client

        return llm_client.databricks(
            [
                {
                    "role": "system",
                    "content": (
                        f"You are the DRP {step.module} agent. Answer the question "
                        "from the results provided wherever they cover it, and say "
                        "which result you are drawing on. `earlierSteps` holds "
                        "results from earlier modules in this same session — use "
                        "them: a question about a ranking is usually answered by "
                        "the step that produced the ranking, not the newest one.\n\n"
                        "If the results do not cover it but the question is general "
                        "biomedical background — what a gene, protein, pathway or "
                        "disease is — answer it from your own knowledge in a sentence "
                        "or two, and begin that part with 'From general knowledge, not "
                        "these results:'. Refusing to say what a gene is, when the "
                        "researcher is looking at that gene, is not useful. Never invent "
                        "specific numbers, scores or citations this way — those must come "
                        "from the results or not at all. If the question asks for "
                        "something only another run could produce, say so plainly.\n\n"
                        + _module_menu()
                    ),
                },
                {
                    "role": "user",
                    "content": f"Results:\n{json.dumps(context, default=str)}\n\nQuestion: {question}",
                },
            ],
            max_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001 — chat must not break the session
        logger.warning("follow-up answer unavailable for step %s: %s", step.id, exc)
        return (
            context["summary"]
            or "I can't reach the language model right now — the results above are unchanged."
        )


# ── drafts ───────────────────────────────────────────────────────────────────
@router.post("/sessions/draft", response_model=s.DraftSessionResponse, tags=[TAG])
def save_draft(
    body: s.DraftSessionRequest,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """Cache an in-progress draft query (autosave while typing)."""
    draft = (
        db.query(DrpSession)
        .filter(DrpSession.user_id == user.id, DrpSession.is_draft == True)  # noqa: E712
        .order_by(DrpSession.updated_at.desc())
        .first()
    )
    if draft is None:
        draft = DrpSession(
            id=f"draft_{uuid.uuid4().hex[:12]}", user_id=user.id, is_draft=True, status="In Progress"
        )
        db.add(draft)
    draft.query = body.query or ""
    draft.title = (body.query or "").strip()[:120]
    if body.module:
        draft.module = canonical_module(body.module)
    if body.projectId:
        draft.project_id = body.projectId
    draft.updated_at = utcnow()
    db.commit()
    db.refresh(draft)
    return s.DraftSessionResponse(
        sessionId=draft.id,
        query=draft.query or "",
        module=draft.module or None,
        projectId=draft.project_id,
        savedAt=draft.updated_at,
    )


@router.patch("/sessions/draft", response_model=s.DraftSessionResponse, tags=[TAG])
def update_draft(
    body: s.DraftSessionRequest,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """Update the draft session with attached project/module tags."""
    draft = (
        db.query(DrpSession)
        .filter(DrpSession.user_id == user.id, DrpSession.is_draft == True)  # noqa: E712
        .order_by(DrpSession.updated_at.desc())
        .first()
    )
    if draft is None:
        raise HTTPException(status_code=404, detail="No draft session to update")
    if body.query is not None:
        draft.query = body.query
        draft.title = body.query.strip()[:120]
    if body.module is not None:
        draft.module = canonical_module(body.module)
    if body.projectId is not None:
        draft.project_id = body.projectId
    draft.updated_at = utcnow()
    db.commit()
    db.refresh(draft)
    return s.DraftSessionResponse(
        sessionId=draft.id,
        query=draft.query or "",
        module=draft.module or None,
        projectId=draft.project_id,
        savedAt=draft.updated_at,
    )


# ── single session ───────────────────────────────────────────────────────────
@router.get("/sessions/{sessionId}", response_model=s.Session, tags=[TAG])
def get_session(
    sessionId: str,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """Get a single session thread with its full step chain (resume research)."""
    session = _owned_session(db, sessionId, user)
    return _to_wire(_sync_session(db, session))


@router.get("/sessions/{sessionId}/artifacts", response_model=list[s.SessionArtifact], tags=[TAG])
def session_artifacts(
    sessionId: str,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """
    Everything this session produced, one entry per completed step.

    Backs the Artifacts panel, which lists a session's outputs rather than its
    conversation. `resultType` tells the UI which shaped endpoint to open for
    the detail view; `exportable` marks the ones `POST /v1/exports` accepts.
    """
    session = _owned_session(db, sessionId, user)
    _sync_session(db, session)

    # Which shaped result each module leaves behind, and what it is called.
    kinds = {
        "TxKG": ("targets", "Ranked targets"),
        "LitMineX": ("litminex_results", "Article shortlist"),
        "CurateX": ("targets", "Curated compounds"),
        "ScreenSuite": ("targets", "Docking hits"),
        "NovSearch": ("targets", "Novelty report"),
        "SaaS Pipeline": ("targets", "Pipeline run"),
    }

    artifacts = []
    for step in session.steps:
        if step.status != "Completed" or not step.job_id:
            continue
        result_type, label = kinds.get(step.module, ("targets", "Result"))
        artifacts.append(
            s.SessionArtifact(
                id=f"{step.id}:{step.job_id}",
                stepId=step.id,
                jobId=step.job_id,
                module=step.module or "",
                title=f"{label} — {step.module}",
                resultType=result_type,
                summary=step.summary or "",
                createdAt=step.updated_at,
                exportable=True,
            )
        )
    return artifacts


@router.patch("/sessions/{sessionId}", response_model=s.Session, tags=[TAG])
def update_session(
    sessionId: str,
    body: s.UpdateSessionRequest,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """Rename a session, or mark it Saved / Completed ("End Task")."""
    session = _owned_session(db, sessionId, user)
    if body.title is not None:
        session.title = body.title.strip()[:120]
    if body.status is not None:
        session.status = body.status
    session.updated_at = utcnow()
    db.commit()
    db.refresh(session)
    return _to_wire(session)


@router.delete("/sessions/{sessionId}", response_model=s.SuccessResponse, tags=[TAG])
def delete_session(
    sessionId: str,
    user: User = Depends(current_user),
    db: OrmSession = Depends(get_db),
):
    """Delete a session and its steps and messages. Jobs are left for audit."""
    session = _owned_session(db, sessionId, user)
    db.delete(session)
    db.commit()
    return s.SuccessResponse(success=True, message=f"Session '{sessionId}' deleted")
