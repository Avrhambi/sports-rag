"""The one place Gemini gets called, so quota handling lives in one place.

Free-tier quota is counted per key per day (500 requests for flash-lite). A
full eval run is roughly three calls per question, so a single key runs out
after a handful of runs -- and it ran out mid-judging once, silently dropping
exactly the questions under investigation and reporting a clean score over
the ones that already passed. With more than one key configured, an exhausted
key rotates to the next instead of failing the run.
"""

import re
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from src.config import GEMINI_API_KEYS, GEMINI_MODEL_NAME

# Index of the key currently believed good. Rotation sticks: once a key is
# exhausted for the day, every later call starts from its successor rather
# than paying a failed request each time to rediscover it.
_current_key = 0

# A 429 is two different failures sharing one status code, and they want
# opposite handling. The daily free-tier allowance is per key and gone until
# tomorrow, so the only move is the next key. A per-minute rate limit clears
# in seconds -- rotating away from a perfectly good key, or (once every key
# has been tried) giving up and letting the planner fall back to its frozen
# heuristics, discards an answer that a short wait would have produced.
# Google names the quota in the error; only the daily one says "PerDay".
DAILY_QUOTA_MARKER = "PerDay"
# Long enough for the retryDelay the API actually returns on a minute limit,
# short enough that a user waiting on /api/ask is not left hanging.
MAX_RATE_LIMIT_WAIT_SECONDS = 20
_RETRY_DELAY = re.compile(r"'retryDelay': '(\d+(?:\.\d+)?)s'")


def has_key() -> bool:
    return bool(GEMINI_API_KEYS)


def key_count() -> int:
    return len(GEMINI_API_KEYS)


def active_key_label() -> str:
    """Which key is in use, for logs -- never the key itself."""
    return f"key {_current_key + 1} of {len(GEMINI_API_KEYS)}" if GEMINI_API_KEYS else "no key"


def is_quota_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def is_daily_quota_error(exc: Exception) -> bool:
    """True for the per-day allowance, which no amount of waiting restores."""
    return is_quota_error(exc) and DAILY_QUOTA_MARKER in str(exc)


def retry_after_seconds(exc: Exception) -> float | None:
    """The wait the API itself asks for, when it names one."""
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


def generate_content(contents: Any, config: dict | None = None, model: str = GEMINI_MODEL_NAME):
    """Call Gemini, moving to the next key when one is out of quota.

    Only quota errors rotate; anything else is a real failure and is raised
    unchanged rather than retried against every key in turn.

    Sampling is off by default. Nothing here wants a creative answer: the
    task is to read a fact out of a supplied context, and the eval judge is
    scoring whether that happened. Left at the API default, the same question
    over the same context returned "Jayson Tatum won 2024" and "Jayson Tatum
    did not win any title" on different runs -- so a one-run measurement was
    reporting sampling noise, and three rounds of chasing that as if it were
    a prompt problem produced two fixes that only appeared to work.
    """
    config = {"temperature": 0, **(config or {})}
    if not GEMINI_API_KEYS:
        raise RuntimeError(
            "No Gemini API key configured. Copy .env.example to .env and set GEMINI_API_KEY_1."
        )

    global _current_key
    last_error: Exception | None = None
    for offset in range(len(GEMINI_API_KEYS)):
        index = (_current_key + offset) % len(GEMINI_API_KEYS)
        label = f"key {index + 1} of {len(GEMINI_API_KEYS)}"
        client = genai.Client(api_key=GEMINI_API_KEYS[index])
        response = None
        # Two attempts at most, and only when the first failure was a minute
        # limit the API told us how long to wait out.
        for attempt in (1, 2):
            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                break
            except genai_errors.ClientError as exc:
                if not is_quota_error(exc):
                    raise
                last_error = exc
                if is_daily_quota_error(exc):
                    print(f"[gemini] {label} is out of quota for today")
                    break
                delay = retry_after_seconds(exc)
                if attempt == 2 or delay is None or delay > MAX_RATE_LIMIT_WAIT_SECONDS:
                    print(f"[gemini] {label} is rate limited")
                    break
                print(f"[gemini] {label} is rate limited; waiting {delay:.0f}s and retrying")
                time.sleep(delay)
        if response is None:
            continue
        if index != _current_key:
            print(f"[gemini] switched to {label}")
            _current_key = index
        return response

    raise last_error if last_error else RuntimeError("Gemini call failed with no error recorded")
