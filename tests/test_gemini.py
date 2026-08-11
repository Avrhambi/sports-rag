import pytest
from google.genai import errors as genai_errors

from src import config, gemini


def test_keys_are_collected_in_order_and_deduplicated(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_1", " first ")
    monkeypatch.setenv("GEMINI_API_KEY_2", "second")
    monkeypatch.setenv("GEMINI_API_KEY_3", "")
    monkeypatch.setenv("GEMINI_API_KEY", "second")  # legacy duplicate
    assert config._gemini_keys() == ["first", "second"]


def test_no_keys_configured_reads_as_unconfigured(monkeypatch):
    for name in ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "")
    assert config._gemini_keys() == []


class _FakeModels:
    def __init__(self, fail_for: set[str], recorder: list[str]):
        self._fail_for = fail_for
        self._recorder = recorder

    def generate_content(self, model, contents, config=None):
        key = self._key
        self._recorder.append(key)
        if key in self._fail_for:
            raise genai_errors.ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
        return f"answered with {key}"


def _fake_client_factory(fail_for: set[str], recorder: list[str]):
    def factory(api_key: str):
        models = _FakeModels(fail_for, recorder)
        models._key = api_key
        return type("Client", (), {"models": models})()

    return factory


@pytest.fixture(autouse=True)
def _reset_rotation(monkeypatch):
    monkeypatch.setattr(gemini, "_current_key", 0)


def test_an_exhausted_key_rotates_to_the_next(monkeypatch):
    """A daily quota is per key, so a second key is the difference between a
    run finishing and a run silently dropping its hardest questions."""
    used: list[str] = []
    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1", "k2"])
    monkeypatch.setattr(gemini.genai, "Client", _fake_client_factory({"k1"}, used))

    assert gemini.generate_content("hi") == "answered with k2"
    assert used == ["k1", "k2"]


def test_rotation_sticks_so_a_dead_key_is_not_retried(monkeypatch):
    used: list[str] = []
    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1", "k2"])
    monkeypatch.setattr(gemini.genai, "Client", _fake_client_factory({"k1"}, used))

    gemini.generate_content("first")
    gemini.generate_content("second")
    # k1 is tried once, not once per call.
    assert used == ["k1", "k2", "k2"]


def test_every_key_exhausted_raises(monkeypatch):
    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1", "k2"])
    monkeypatch.setattr(gemini.genai, "Client", _fake_client_factory({"k1", "k2"}, []))

    with pytest.raises(genai_errors.ClientError):
        gemini.generate_content("hi")


def test_a_non_quota_error_is_not_retried_against_other_keys(monkeypatch):
    """Rotating on a real failure would burn every key on the same bug."""
    used: list[str] = []

    def factory(api_key: str):
        used.append(api_key)
        models = type("M", (), {
            "generate_content": lambda self, model, contents, config=None: (_ for _ in ()).throw(
                genai_errors.ClientError(400, {"error": {"message": "INVALID_ARGUMENT"}})
            )
        })()
        return type("Client", (), {"models": models})()

    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1", "k2"])
    monkeypatch.setattr(gemini.genai, "Client", factory)

    with pytest.raises(genai_errors.ClientError):
        gemini.generate_content("hi")
    assert used == ["k1"]
