import pytest
from fastapi import HTTPException
from google.genai import errors as genai_errors

import app as app_module
from src.models import AskRequest
from src.plan import QueryPlan

CHUNK = {
    "id": "f-2022-info",
    "source_title": "2022 UEFA Champions League final",
    "url": "https://example.invalid",
    "sport": "football",
    "competition": "UEFA Champions League",
    "season": "2022",
    "text": "Liverpool 0-1 Real Madrid",
}


@pytest.fixture
def stub_pipeline(monkeypatch):
    monkeypatch.setattr(app_module, "plan_query", lambda q: QueryPlan(sport="football", years=[2022]))
    monkeypatch.setattr(app_module, "retrieve", lambda *a, **k: [CHUNK])


def _raise(exc):
    def boom(*args, **kwargs):
        raise exc

    return boom


def test_an_exhausted_quota_says_so_instead_of_looking_like_a_network_fault(stub_pipeline, monkeypatch):
    """src/gemini.py has already tried every key by this point, so the day is
    genuinely over. The generic error line tells the user to try again in a
    moment, which is advice that never comes good."""
    monkeypatch.setattr(
        app_module,
        "generate_answer",
        _raise(genai_errors.ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})),
    )

    with pytest.raises(HTTPException) as raised:
        app_module.ask(AskRequest(question="מי ניצח בגמר ליגת האלופות 2022?"))
    assert raised.value.status_code == 503
    assert "מכסת השאלות היומית" in raised.value.detail


def test_any_other_failure_is_not_dressed_up_as_a_quota_problem(stub_pipeline, monkeypatch):
    monkeypatch.setattr(app_module, "generate_answer", _raise(ValueError("something else broke")))

    with pytest.raises(ValueError):
        app_module.ask(AskRequest(question="מי ניצח בגמר ליגת האלופות 2022?"))
