"""Articles — /articles/{articleId}[/pmc-link|/save|/chat|/chat/history]."""
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from DRP_Main.app.core.config import settings
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.db.session import get_db
from DRP_Main.app.drp import schemas as s
from DRP_Main.app.drp.deps import current_user
from DRP_Main.app.drp.models import (
    DrpArticle,
    DrpArticleChatMessage,
    DrpProject,
    DrpProjectItem,
    DrpSavedArticle,
)
from DRP_Main.app.models.user import User

logger = get_logger(__name__)
router = APIRouter()
TAG = "Articles"


def _get_article(db: Session, article_id: str) -> DrpArticle:
    article = db.query(DrpArticle).filter(DrpArticle.id == article_id).first()
    if article is None:
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found")
    return article


@router.get("/articles/{articleId}", response_model=s.ArticleDetail, tags=[TAG])
def article_detail(
    articleId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Get full article detail."""
    article = _get_article(db, articleId)
    return s.ArticleDetail(
        id=article.id,
        title=article.title or "",
        authors=article.authors or "",
        year=article.year,
        abstract=article.abstract or article.preview or "",
        keywords=article.keywords or [],
        pmcLink=article.pmc_link or "",
    )


@router.get("/articles/{articleId}/pmc-link", response_model=s.ExternalLink, tags=[TAG])
def article_pmc_link(
    articleId: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Get the external PubMed Central link."""
    article = _get_article(db, articleId)
    url = article.pmc_link or article.pdf_url
    if not url and article.pmid:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"
    if not url:
        raise HTTPException(status_code=404, detail="No external link is available for this article")
    provider = "PubMed Central" if "pmc" in url.lower() else "PubMed"
    return s.ExternalLink(url=url, provider=provider)


@router.post("/articles/{articleId}/save", response_model=s.SuccessResponse, tags=[TAG])
def save_article(
    articleId: str,
    projectId: str | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Save an article to the library, optionally into a project."""
    article = _get_article(db, articleId)

    project = None
    if projectId:
        project = (
            db.query(DrpProject)
            .filter(DrpProject.id == projectId, DrpProject.user_id == user.id)
            .first()
        )
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project '{projectId}' not found")

    existing = (
        db.query(DrpSavedArticle)
        .filter(
            DrpSavedArticle.user_id == user.id,
            DrpSavedArticle.article_id == article.id,
            DrpSavedArticle.project_id == projectId,
        )
        .first()
    )
    if existing is None:
        db.add(
            DrpSavedArticle(
                id=str(uuid.uuid4()),
                user_id=user.id,
                article_id=article.id,
                project_id=projectId,
            )
        )
    if project is not None:
        db.add(
            DrpProjectItem(
                id=f"item_{uuid.uuid4().hex[:12]}",
                project_id=project.id,
                result_id=article.id,
                result_type="article",
                payload={"title": article.title, "pmcLink": article.pmc_link},
            )
        )
    db.commit()
    where = f" to '{project.name}'" if project is not None else " to your library"
    return s.SuccessResponse(success=True, message=f"Article saved{where}")


@router.post("/articles/{articleId}/chat", response_model=s.ChatMessage, tags=[TAG])
def chat_with_article(
    articleId: str,
    body: s.ChatRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Ask a question about the article ('Chat with Article')."""
    article = _get_article(db, articleId)
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    db.add(
        DrpArticleChatMessage(
            id=str(uuid.uuid4()),
            article_id=article.id,
            user_id=user.id,
            session_id=body.sessionId,
            role="user",
            content=body.message,
        )
    )
    db.commit()

    citations = [c for c in (article.pmc_link, article.pdf_url) if c]
    if article.pmid:
        citations.append(f"PMID:{article.pmid}")
    answer = _answer_from_article(article, body.message)

    db.add(
        DrpArticleChatMessage(
            id=str(uuid.uuid4()),
            article_id=article.id,
            user_id=user.id,
            session_id=body.sessionId,
            role="agent",
            content=answer,
            citations=citations,
        )
    )
    db.commit()
    return s.ChatMessage(role="agent", content=answer, citations=citations)


_PMCID = re.compile(r"(PMC\d+)", re.I)
#: Europe PMC serves open-access full text as JATS XML, keyed by PMCID. Only
#: open-access articles resolve; everything else 404s, which is the signal to
#: fall back to the abstract.
_EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
_FULL_TEXT_CHARS = 60_000


def _article_full_text(article: DrpArticle) -> tuple[str, str]:
    """
    The best available text for this article, and what it is.

    Chat was answering from the abstract even when the article's own full text
    was a request away, so "what does this article say about X" could only ever
    be answered from ~250 words — and for anything in the methods or the results
    the honest answer was always "the abstract does not cover that". The
    open-access body is fetched from Europe PMC when the article has a PMCID.

    Returns `(text, kind)` where kind is "full text" or "abstract", so the prompt
    and the answer can be honest about which one was read. Failure is never
    fatal: a timeout, a paywall or a missing PMCID all fall back to the abstract.
    """
    abstract = (article.abstract or article.preview or "").strip()

    match = _PMCID.search(f"{article.pmc_link or ''} {article.pdf_url or ''}")
    if not match:
        return abstract, "abstract"

    try:
        import httpx

        response = httpx.get(
            _EPMC_FULLTEXT.format(pmcid=match.group(1).upper()),
            timeout=12.0,
            headers={"User-Agent": "InnoDD-API/1.0 (drug repurposing platform)"},
        )
        if response.status_code != 200 or not response.text.strip():
            return abstract, "abstract"
        # Strip JATS tags rather than parsing: the body text is what matters and
        # a full XML parse would pull in references and float boxes as prose.
        text = re.sub(r"<(ref-list|back|front)\b.*?</\1>", " ", response.text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > len(abstract):
            return text[:_FULL_TEXT_CHARS], "full text"
    except Exception as exc:  # noqa: BLE001 — the abstract is always a usable source
        logger.warning("full-text fetch failed for %s, using abstract: %s", article.id, exc)
    return abstract, "abstract"


def _answer_from_article(article: DrpArticle, question: str) -> str:
    """
    Answer grounded in the article's full text where it is open access, and in
    its abstract otherwise.

    Uses the Databricks pay-per-token Foundation Model API when configured;
    otherwise falls back to returning the most relevant sentences so the
    endpoint stays useful offline.
    """
    context, source_kind = _article_full_text(article)
    if not context:
        return "This article has no text stored, so I cannot answer from its content."

    # No provider gate: `llm_client.chat` tries Databricks, then Groq, then
    # Gemini, and raises only when none is configured. This used to be gated on
    # DATABRICKS_TOKEN, which the deployed app does not set — so no LLM was ever
    # called and every question about an article was answered with the abstract's
    # opening sentences.
    try:
        from DRP_Main.app.core.llm import llm_client

        answer = llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a biomedical research assistant helping a researcher "
                        "read one paper. Answer their question about it.\n\n"
                        f"You have been given this paper's {source_kind}. Use it as "
                        "your source wherever it covers the question, and quote or "
                        "paraphrase what it actually says. Where it does not cover "
                        "the question, say so in a clause — not as the whole answer "
                        "— and then answer from general biomedical knowledge if you "
                        "reliably can, making clear that part is background rather "
                        "than from this paper. A plain question such as what a gene "
                        "or a term means deserves a plain answer, not a refusal.\n\n"
                        "Never invent findings, numbers or citations and attribute "
                        "them to this paper. Address the reader as \"you\"."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Title: {article.title}\n"
                    f"{source_kind.title()}: {context}\n\n"
                    f"Question: {question}",
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )
        if answer and answer.strip():
            return answer.strip()
    except Exception as exc:  # noqa: BLE001 — fall back rather than 500
        logger.warning("article chat LLM call failed, using extractive fallback: %s", exc)

    terms = {w.lower().strip("?.,") for w in question.split() if len(w) > 3}
    sentences = [snippet.strip() for snippet in context.split(". ") if snippet.strip()]
    ranked = sorted(
        sentences,
        key=lambda snippet: sum(1 for term in terms if term in snippet.lower()),
        reverse=True,
    )
    best = [snippet for snippet in ranked[:3] if any(t in snippet.lower() for t in terms)]
    if not best:
        return (
            "The abstract does not directly address that. Here is what it reports: "
            + ". ".join(sentences[:2])
            + "."
        )
    return ". ".join(best) + "."


@router.get("/articles/{articleId}/chat/history", response_model=list[s.ChatMessage], tags=[TAG])
def chat_history(
    articleId: str,
    sessionId: Optional[str] = Query(
        None,
        description="Scope the thread to one research session. Omit to get every "
                    "turn on this article, which is the pre-session behaviour.",
    ),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get the chat history for an article thread, within a session."""
    _get_article(db, articleId)
    query = db.query(DrpArticleChatMessage).filter(
        DrpArticleChatMessage.article_id == articleId,
        DrpArticleChatMessage.user_id == user.id,
    )
    if sessionId:
        # Only this session's turns. Rows written before the column existed have
        # no session and are correctly left out of a session-scoped thread.
        query = query.filter(DrpArticleChatMessage.session_id == sessionId)
    rows = query.order_by(DrpArticleChatMessage.created_at).all()
    return [
        s.ChatMessage(role=row.role, content=row.content or "", citations=row.citations or [])
        for row in rows
    ]
