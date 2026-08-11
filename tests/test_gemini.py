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


def test_sampling_is_off_unless_a_caller_overrides_it(monkeypatch):
    """Left at the API default, the same question over the same context came
    back both "Tatum won 2024" and "Tatum never won" on different runs, and
    single-run eval results were reporting that noise as signal."""
    seen: list[dict] = []

    def factory(api_key: str):
        models = type("M", (), {
            "generate_content": lambda self, model, contents, config=None: (
                seen.append(config) or "ok"
            )
        })()
        return type("Client", (), {"models": models})()

    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1"])
    monkeypatch.setattr(gemini.genai, "Client", factory)

    gemini.generate_content("q")
    assert seen[-1]["temperature"] == 0

    # A caller's own config survives, and can still override the default.
    gemini.generate_content("q", config={"response_mime_type": "application/json"})
    assert seen[-1] == {"temperature": 0, "response_mime_type": "application/json"}
    gemini.generate_content("q", config={"temperature": 0.7})
    assert seen[-1]["temperature"] == 0.7


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


def _rate_limited(retry_delay: str | None = "3s") -> genai_errors.ClientError:
    """A per-minute 429: same status code as the daily one, no "PerDay"."""
    detail = {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
    if retry_delay:
        detail["retryDelay"] = retry_delay
    return genai_errors.ClientError(429, {"error": {"message": f"RESOURCE_EXHAUSTED {detail}"}})


def _daily_exhausted() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429,
        {"error": {"message": "RESOURCE_EXHAUSTED {'quotaId': "
                              "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}"}},
    )


def test_the_two_kinds_of_429_are_told_apart():
    """Only the daily one is unrecoverable; waiting fixes the other."""
    assert gemini.is_daily_quota_error(_daily_exhausted())
    assert not gemini.is_daily_quota_error(_rate_limited())
    assert gemini.is_quota_error(_rate_limited())
    assert gemini.retry_after_seconds(_rate_limited("12s")) == 12
    assert gemini.retry_after_seconds(_daily_exhausted()) is None


def test_a_minute_limit_waits_and_retries_the_same_key(monkeypatch):
    """Rotating away from a good key -- or giving up with every key briefly
    rate limited -- turns a few seconds' wait into a degraded answer."""
    used: list[str] = []
    slept: list[float] = []
    monkeypatch.setattr(gemini.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1", "k2"])

    def factory(api_key: str):
        def generate_content(self, model, contents, config=None):
            used.append(api_key)
            if len(used) == 1:
                raise _rate_limited("3s")
            return f"answered with {api_key}"

        return type("Client", (), {"models": type("M", (), {"generate_content": generate_content})()})()

    monkeypatch.setattr(gemini.genai, "Client", factory)
    assert gemini.generate_content("hi") == "answered with k1"
    assert used == ["k1", "k1"]  # same key, not the next one
    assert slept == [3]


def test_a_long_or_unstated_wait_moves_on_rather_than_blocking(monkeypatch):
    """A user waiting on /api/ask should not be held for a minute."""
    used: list[str] = []
    slept: list[float] = []
    monkeypatch.setattr(gemini.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1", "k2"])

    def factory(api_key: str):
        def generate_content(self, model, contents, config=None):
            used.append(api_key)
            if api_key == "k1":
                raise _rate_limited("120s")
            return f"answered with {api_key}"

        return type("Client", (), {"models": type("M", (), {"generate_content": generate_content})()})()

    monkeypatch.setattr(gemini.genai, "Client", factory)
    assert gemini.generate_content("hi") == "answered with k2"
    assert used == ["k1", "k2"]
    assert slept == []


def test_a_daily_exhaustion_rotates_without_waiting(monkeypatch):
    used: list[str] = []
    slept: list[float] = []
    monkeypatch.setattr(gemini.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(gemini, "GEMINI_API_KEYS", ["k1", "k2"])

    def factory(api_key: str):
        def generate_content(self, model, contents, config=None):
            used.append(api_key)
            if api_key == "k1":
                raise _daily_exhausted()
            return f"answered with {api_key}"

        return type("Client", (), {"models": type("M", (), {"generate_content": generate_content})()})()

    monkeypatch.setattr(gemini.genai, "Client", factory)
    assert gemini.generate_content("hi") == "answered with k2"
    assert used == ["k1", "k2"]  # tried once, no wait
    assert slept == []


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
