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
    # True when the query planner was unavailable and a frozen keyword
    # fallback stood in. That path treats every question as a single-fact
    # lookup, so counts, comparisons and "did X ever happen" answers are
    # drawn from a handful of chunks instead of the full set -- worth telling
    # the user rather than serving a quietly weaker answer as if it were normal.
    degraded: bool = False
    # The sport the question is actually about, when that contradicts the tab
    # the user had selected. Tabs are sticky and questions are not: a tab left
    # on כדורגל from an earlier search hard-filtered an NBA question down to
    # football chunks and returned a refusal, with nothing on screen to say
    # the tab had caused it. The question wins, and the UI says so.
    sport_override: str | None = None
