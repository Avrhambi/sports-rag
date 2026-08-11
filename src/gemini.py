"""The one place Gemini gets called, so quota handling lives in one place.

Free-tier quota is counted per key per day (500 requests for flash-lite). A
full eval run is roughly three calls per question, so a single key runs out
after a handful of runs -- and it ran out mid-judging once, silently dropping
exactly the questions under investigation and reporting a clean score over
the ones that already passed. With more than one key configured, an exhausted
key rotates to the next instead of failing the run.
"""

from typing import Any

from google import genai
from google.genai import errors as genai_errors

from src.config import GEMINI_API_KEYS, GEMINI_MODEL_NAME

# Index of the key currently believed good. Rotation sticks: once a key is
# exhausted for the day, every later call starts from its successor rather
# than paying a failed request each time to rediscover it.
_current_key = 0


def has_key() -> bool:
    return bool(GEMINI_API_KEYS)


def key_count() -> int:
    return len(GEMINI_API_KEYS)


def active_key_label() -> str:
    """Which key is in use, for logs -- never the key itself."""
    return f"key {_current_key + 1} of {len(GEMINI_API_KEYS)}" if GEMINI_API_KEYS else "no key"


def is_quota_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def generate_content(contents: Any, config: dict | None = None, model: str = GEMINI_MODEL_NAME):
    """Call Gemini, moving to the next key when one is out of quota.

    Only quota errors rotate; anything else is a real failure and is raised
    unchanged rather than retried against every key in turn.
    """
    if not GEMINI_API_KEYS:
        raise RuntimeError(
            "No Gemini API key configured. Copy .env.example to .env and set GEMINI_API_KEY_1."
        )

    global _current_key
    last_error: Exception | None = None
    for offset in range(len(GEMINI_API_KEYS)):
        index = (_current_key + offset) % len(GEMINI_API_KEYS)
        client = genai.Client(api_key=GEMINI_API_KEYS[index])
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.ClientError as exc:
            if not is_quota_error(exc):
                raise
            last_error = exc
            print(f"[gemini] key {index + 1} of {len(GEMINI_API_KEYS)} is out of quota")
            continue
        if index != _current_key:
            print(f"[gemini] switched to key {index + 1} of {len(GEMINI_API_KEYS)}")
            _current_key = index
        return response

    raise last_error if last_error else RuntimeError("Gemini call failed with no error recorded")
