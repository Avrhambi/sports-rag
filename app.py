"""FastAPI app: /api/ask, /api/health, and the pitch-lines UI served from static/."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.config import SPORTS
from src.generate import generate_answer
from src.models import AskRequest, AskResponse, SourceOut
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

    chunks = retrieve(question, sport=request.sport)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant match data found for this question.")

    answer = generate_answer(question, chunks)
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
    return AskResponse(answer=answer, sources=sources)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
