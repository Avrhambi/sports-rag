"""FastAPI app: /api/ask, /api/health, and the pitch-lines UI served from static/."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src import gemini
from src.config import COMPETITION_NAME_HE, SPORT_NAME_HE, SPORTS
from src.generate import generate_answer
from src.models import AskRequest, AskResponse, SourceOut
from src.plan import QueryPlan, plan_query
from src.retrieve import retrieve

app = FastAPI(title="Sports Finals History RAG")


def off_tab_sport(requested: str | None, plan: QueryPlan) -> str | None:
    """The sport a question is about when that is not the selected tab.

    The tab is a scope the user chose, so it is not overruled: on the כדורסל
    tab this app answers out of the NBA Finals and nothing else. What it must
    not do is answer anyway. Passing a football question through the
    basketball filter used to retrieve four NBA chunks and hand them to the
    model, which then produced a refusal phrased as missing data -- the
    corpus looked incomplete when the tab was the whole story.

    Returning the mismatched sport lets `ask` say exactly that instead, and
    skip the generation call entirely.
    """
    if requested and plan.sport_filter and plan.sport_filter != requested:
        return plan.sport_filter
    return None


def off_tab_answer(tab: str, question_sport: str) -> str:
    """The refusal itself: what this tab covers, what was asked, what to do."""
    return (
        f"בלשונית {SPORT_NAME_HE[tab]} אני עונה רק מתוך {COMPETITION_NAME_HE[tab]}, "
        f"והשאלה הזאת היא על {SPORT_NAME_HE[question_sport]}. "
        f"כדי לקבל תשובה, עברו ללשונית {SPORT_NAME_HE[question_sport]} או ללשונית «הכל» ושאלו שוב."
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if request.sport is not None and request.sport not in SPORTS:
        raise HTTPException(status_code=400, detail=f"sport must be one of {SPORTS} or omitted.")

    # Planned once and handed to both stages: retrieval uses it to decide what
    # counts as evidence, generation to decide whether that evidence is a
    # complete set worth counting over.
    plan = plan_query(question)

    # Answer the tab mismatch instead of searching past it. This also costs
    # one Gemini call rather than two, which matters on a free-tier quota.
    tab = request.sport
    mismatch = off_tab_sport(tab, plan)
    if tab and mismatch:
        return AskResponse(
            answer=off_tab_answer(tab, mismatch),
            sources=[],
            degraded=plan.degraded,
            off_tab_sport=mismatch,
        )

    chunks = retrieve(question, sport=request.sport, plan=plan)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant match data found for this question.")

    try:
        answer = generate_answer(question, chunks, plan)
    except Exception as exc:  # re-raised unless it turns out to be the quota
        if not gemini.is_quota_error(exc):
            raise
        # src/gemini.py has already tried every key and waited out anything
        # short. Reaching here means the day's allowance is gone, which is
        # not "there is a problem with the line" -- the generic error the UI
        # would otherwise show sends the user to retry immediately, forever.
        raise HTTPException(
            status_code=503,
            detail="הגענו למכסת השאלות היומית מול שירות ה-AI. נסו שוב מאוחר יותר.",
        ) from exc
    sources = [
        SourceOut(
            source_title=c["source_title"],
            url=c["url"],
            sport=c["sport"],
            competition=c["competition"],
            season=c["season"],
            text=c["text"],
        )
        for c in chunks
    ]
    return AskResponse(answer=answer, sources=sources, degraded=plan.degraded)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
