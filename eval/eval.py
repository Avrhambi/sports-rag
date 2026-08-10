"""Score the RAG pipeline against a seed Hebrew Q&A set.

Two tiers:
  - Retrieval checks (always run): does an unfiltered search surface the
    expected sport's chunks, and do expected keywords show up in the
    retrieved context? These need no API key, but with one set they measure
    the real system -- retrieval plans each question through Gemini, and
    without a key it silently falls back to `src.plan.heuristic_plan`.
  - Generation checks (only run if GEMINI_API_KEY is set): Gemini judges
    each generated Hebrew answer for faithfulness and relevance against
    the retrieved context, and for correctness against a reference answer,
    on a 0-1 scale.

Every question carries a `type`: `factoid` (one fact from one match report)
or one of the complex types -- `aggregate` (count/superlative/sum over
several finals), `comparison` (facts from several finals side by side), and
`multihop` (one final's fact selects which other final to look up). Metrics
are reported per type, because a system that answers factoids perfectly can
still score zero on everything else: a complex question typically needs
*every* relevant chunk in context, not just the top-scoring one.

For the same reason, `expected_keywords` on a complex question lists the
full evidence set (e.g. all five finals' results for a "which final had the
biggest margin" question), so keyword coverage doubles as a retrieval-recall
measure. `expected_sport` is null on cross-sport questions, which are
excluded from the top-result sport metric.

Run as: python -m eval.eval
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from src.generate import generate_answer
from src.plan import plan_query
from src.retrieve import retrieve

TESTSET_PATH = Path(__file__).resolve().parent / "qa_testset.json"

# Free-tier Gemini quota is a handful of requests/minute; pace calls to stay under it
# rather than let the whole run die on the first 429.
RATE_LIMIT_DELAY_SECONDS = 13

JUDGE_PROMPT = """You are grading a Hebrew answer to a sports-history question.
Score three things from 0 to 1 (a decimal number each):
- faithfulness: is every claim in the answer supported by the context?
- relevance: does the answer directly address the question?
- correctness: does the answer state the same facts as the reference answer?
  Judge this against the reference only, ignoring the context. An answer that
  declines to answer, or that covers only part of what the reference states,
  is not correct. Wording and phrasing may differ freely.

Question (Hebrew): {question}
Context: {context}
Reference answer (Hebrew): {reference}
Answer (Hebrew): {answer}

Respond with exactly three lines:
faithfulness: <0-1>
relevance: <0-1>
correctness: <0-1>
"""

# Reported in this order so factoids (the easy tier) come first.
TYPE_ORDER = ["factoid", "aggregate", "comparison", "multihop"]


def load_testset() -> list[dict]:
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


def score_retrieval(item: dict, top_k: int = 4) -> dict:
    chunks = retrieve(item["question"], sport=None, top_k=top_k)
    # Cross-sport questions have no single expected sport -- None here means
    # "not applicable", and main() drops those from the sport-match rate.
    top_sport_match = None
    if item["expected_sport"]:
        top_sport_match = bool(chunks) and chunks[0]["sport"] == item["expected_sport"]
    context = " ".join(c["text"] for c in chunks).lower()
    keyword_hits = [kw for kw in item["expected_keywords"] if kw.lower() in context]
    return {
        "top_sport_match": top_sport_match,
        "keyword_coverage": len(keyword_hits) / len(item["expected_keywords"]),
        "chunks": chunks,
    }


def score_generation(item: dict, chunks: list[dict]) -> dict | None:
    """Ask Gemini to judge one answer. Returns None when it can't be scored --
    no chunks, no API key, rate limit, or a network drop. A full run is ~13
    minutes of paced calls, so one flaky question must not discard the other
    nineteen; the counts printed at the end say how many actually landed."""
    if not GEMINI_API_KEY or not chunks:
        return None

    try:
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        # Cached from the retrieval pass, so this costs no extra call -- but it
        # keeps the eval on the same code path as app.py's /api/ask.
        answer = generate_answer(item["question"], chunks, plan_query(item["question"]))

        client = genai.Client(api_key=GEMINI_API_KEY)
        context = "\n".join(c["text"] for c in chunks)
        judge_prompt = JUDGE_PROMPT.format(
            question=item["question"],
            context=context,
            reference=item["reference_answer"],
            answer=answer,
        )
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=judge_prompt)
    except Exception as e:  # noqa: BLE001 - transport errors are as skippable as API ones
        print(f"  skipped {item['id']}: {type(e).__name__}: {e}")
        return None

    scores: dict = {"answer": answer}
    for line in response.text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            try:
                scores[key.strip()] = float(value.strip())
            except ValueError:
                pass
    return scores


def breakdown_by_type(scored: list[tuple[dict, float | bool | None]], fmt: str = ".0%") -> str:
    """Render `type: mean (count)` for each question type present, skipping
    items whose score is None (not applicable to that question)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for item, value in scored:
        if value is not None:
            groups[item["type"]].append(float(value))
    ordered = sorted(groups.items(), key=lambda kv: TYPE_ORDER.index(kv[0]))
    return "  ".join(f"{t} {sum(v) / len(v):{fmt}} ({len(v)})" for t, v in ordered)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    testset = load_testset()
    # With a key set, each score_retrieval makes a planner call, so it needs
    # the same free-tier pacing as the judging loop below.
    retrieval_results = []
    for i, item in enumerate(testset):
        if GEMINI_API_KEY and i:
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
        retrieval_results.append(score_retrieval(item))

    n = len(testset)
    sport_matches = [r["top_sport_match"] for r in retrieval_results if r["top_sport_match"] is not None]
    keyword_coverages = [r["keyword_coverage"] for r in retrieval_results]

    print(f"Retrieval - top result matches expected sport: {mean(sport_matches):.0%} ({len(sport_matches)} single-sport questions)")
    print(f"Retrieval - avg expected-keyword coverage: {mean(keyword_coverages):.0%} ({n} questions)")
    print(f"  by type: {breakdown_by_type(list(zip(testset, keyword_coverages)))}")

    if not GEMINI_API_KEY:
        print("Generation checks skipped - set GEMINI_API_KEY in .env to enable them.")
        return

    print(f"Judging {n} generated answers with Gemini (paced for free-tier rate limits, this takes a few minutes)...")
    generation_results = []
    for item, r in zip(testset, retrieval_results):
        generation_results.append(score_generation(item, r["chunks"]))

    judged = [g for g in generation_results if g]
    if judged:
        for metric in ("faithfulness", "relevance", "correctness"):
            values = [g.get(metric, 0) for g in judged]
            print(f"Generation - avg {metric}: {mean(values):.2f} ({len(judged)} answers judged)")
        correctness = [g.get("correctness", 0) if g else None for g in generation_results]
        print(f"  correctness by type: {breakdown_by_type(list(zip(testset, correctness)), fmt='.2f')}")

        # Name the questions that dragged a type's mean down -- without this
        # a regression shows up as a decimal with no way to chase it.
        weak = [(i, g) for i, g in zip(testset, generation_results) if g and g.get("correctness", 1) < 1]
        for item, scored in weak:
            print(f"  imperfect: {item['id']} ({scored.get('correctness')}) -> {scored['answer'][:140]}")


if __name__ == "__main__":
    main()
