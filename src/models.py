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
    # The sport the question turned out to be about, when that is not the tab
    # the user had selected. The tab is a scope the user chose, so it stands
    # and the question goes unanswered -- but the answer says so in as many
    # words instead of searching the wrong sport and reporting the corpus as
    # incomplete. The UI turns this into a one-click switch to the right tab.
    off_tab_sport: str | None = None
