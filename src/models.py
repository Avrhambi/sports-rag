"""Pydantic request/response schemas for the /api/ask endpoint."""

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    sport: str | None = None  # "football" | "basketball" | None (all sports)


class SourceOut(BaseModel):
    source_title: str
    url: str
    sport: str
    competition: str
    season: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
