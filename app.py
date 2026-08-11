"""FastAPI app: /api/ask, /api/health, and the pitch-lines UI served from static/."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.config import SPORTS
from src.generate import generate_answer
from src.models import AskRequest, AskResponse, SourceOut
from src.plan import plan_query
from src.retrieve import retrieve

app = FastAPI(title="Sports Finals History RAG")


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
    chunks = retrieve(question, sport=request.sport, plan=plan)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant match data found for this question.")

    answer = generate_answer(question, chunks, plan)
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
