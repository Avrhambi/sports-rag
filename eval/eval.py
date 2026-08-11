"""Score the RAG pipeline against a seed Hebrew Q&A set.

Two tiers:
  - Retrieval checks (always run): does an unfiltered search surface the
    expected sport's chunks, and do expected keywords show up in the
    retrieved context? These need no API key, but with one set they measure
    the real system -- retrieval plans each question through Gemini, and
    without a key it silently falls back to `src.plan.heuristic_plan`.
  - Generation checks (only run if a Gemini API key is set): Gemini judges
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

from src import gemini
from src.generate import generate_answer
from src.plan import plan_query
from src.retrieve import retrieve

TESTSET_PATH = Path(__file__).resolve().parent / "qa_testset.json"

# Free-tier Gemini quota is a handful of requests/minute; pace calls to stay under it
# rather than let the whole run die on the first 429.
RATE_LIMIT_DELAY_SECONDS = 13

JUDGE_PROMPT = """You are grading a Hebrew answer to a sports-history question.
Score four things:
- faithfulness: is every claim in the answer supported by the context?
- relevance: does the answer directly address the question?
- correctness: does the answer state the same facts as the reference answer?
  Judge this against the reference only, ignoring the context. An answer that
  declines to answer, or that covers only part of what the reference states,
  is not correct. Wording and phrasing may differ freely.
- abstained: 1 if the answer declines to answer -- says the sources lack the
  information, that it cannot tell, or similar -- and 0 if it commits to an
  answer. Score this on what the answer does, not on whether it is right. A
  confident wrong answer is 0. "No, that did not happen" is 0, not an
  abstention: it is an answer.

Question (Hebrew): {question}
Context: {context}
Reference answer (Hebrew): {reference}
Answer (Hebrew): {answer}

Respond with exactly four lines:
faithfulness: <0-1>
relevance: <0-1>
correctness: <0-1>
abstained: <0 or 1>
"""

# Reported in this order so factoids (the easy tier) come first. `existence`
# is the closed-world tier -- yes/no questions whose answer is often an
# absence. `uncovered` questions are unanswerable by design: the right
# behaviour there is to abstain and say what is covered instead.
TYPE_ORDER = ["factoid", "existence", "aggregate", "comparison", "multihop", "uncovered"]
ANSWERABLE_TYPES = [t for t in TYPE_ORDER if t != "uncovered"]


def load_testset(only: list[str] | None = None, as_typed: bool = False) -> list[dict]:
    """The whole set, or just the questions whose id contains one of `only`.

    A full run costs ~3 calls per question and the free tier allows 500 a
    day, so re-checking the handful that failed has to be possible without
    spending a run's worth of quota on the ones that already pass.

    `as_typed` swaps each question for its `as_typed` phrasing: the same
    question as a real user actually types it -- no question mark, no geresh
    in transliterated names ("גיילן" for "ג'יילן"), "ב5" for "ב-5", a nickname
    instead of a full club name, and often no competition named at all. The
    expected keywords and the reference answer are unchanged, so the numbers
    line up column-for-column against a normal run and the difference is
    purely what the phrasing cost. It is a separate run rather than extra
    questions because a full pass is already ~3 calls per question."""
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if as_typed:
        testset = [
            {**item, "question": item["as_typed"]} if item.get("as_typed") else item
            for item in testset
        ]
    if not only:
        return testset
    return [item for item in testset if any(fragment in item["id"] for fragment in only)]


def score_retrieval(item: dict, top_k: int = 4) -> dict:
    # Planned explicitly (and cached, so `retrieve` below costs no second
    # call) to find out whether this question was planned at all. On a quota
    # outage `plan_query` falls back to the frozen heuristics, which report
    # `factoid` for everything -- no guaranteed evidence, no completeness
    # promise -- so every aggregate collapses to top-4 and coverage craters.
    # A run that hit that wall once reported "as-typed 68% vs clean 91%" and
    # looked like a phrasing finding; it was measuring the fallback.
    plan = plan_query(item["question"])
    chunks = retrieve(item["question"], sport=None, top_k=top_k, plan=plan)
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
        "degraded": plan.degraded,
    }


def score_generation(item: dict, chunks: list[dict]) -> dict | None:
    """Ask Gemini to judge one answer. Returns None when it can't be scored --
    no chunks, no API key, rate limit, or a network drop. A full run is ~13
    minutes of paced calls, so one flaky question must not discard the other
    nineteen; the counts printed at the end say how many actually landed."""
    if not gemini.has_key() or not chunks:
        return None

    try:
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        # Cached from the retrieval pass, so this costs no extra call -- but it
        # keeps the eval on the same code path as app.py's /api/ask.
        answer = generate_answer(item["question"], chunks, plan_query(item["question"]))

        context = "\n".join(c["text"] for c in chunks)
        judge_prompt = JUDGE_PROMPT.format(
            question=item["question"],
            context=context,
            reference=item["reference_answer"],
            answer=answer,
        )
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        response = gemini.generate_content(judge_prompt)
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


def score_generation_repeated(item: dict, chunks: list[dict], repeats: int) -> tuple[dict | None, list[float]]:
    """Judge the same question `repeats` times and return the last scoring
    plus every correctness value.

    One run is not a result. The same question over a byte-identical prompt
    has returned "Tatum won the 2024 title" and "Tatum never won" on
    different runs -- at temperature 0, with the same plan and the same 32k
    of context. Answers are stable within a few minutes and differ across
    longer gaps, so a single sample reports whichever attractor the backend
    happened to be in, and three separate "fixes" were assessed against it.
    """
    scores: list[float] = []
    last: dict | None = None
    for _ in range(repeats):
        result = score_generation(item, chunks)
        if result:
            last = result
            scores.append(result.get("correctness", 0))
    return last, scores


def main(only: list[str] | None = None, repeats: int = 1, as_typed: bool = False) -> None:
    testset = load_testset(only, as_typed)
    if as_typed:
        print("Asking each question the way a user actually types it (--as-typed).")
    if only:
        print(f"Running {len(testset)} of the full set, filtered by {only}.")
    if repeats > 1:
        print(f"Judging each question {repeats} times to measure run-to-run variance.")
    # With a key set, each score_retrieval makes a planner call, so it needs
    # the same free-tier pacing as the judging loop below.
    retrieval_results = []
    for i, item in enumerate(testset):
        if gemini.has_key() and i:
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
        retrieval_results.append(score_retrieval(item))

    n = len(testset)
    sport_matches = [r["top_sport_match"] for r in retrieval_results if r["top_sport_match"] is not None]
    keyword_coverages = [r["keyword_coverage"] for r in retrieval_results]

    # Say this before any number, and loudly. The frozen fallback plans every
    # question as a single-fact lookup, so an outage turns the aggregate and
    # existence tiers into plain top-4 and the run reports a retrieval
    # regression that is really a spent quota.
    unplanned = [item["id"] for item, r in zip(testset, retrieval_results) if r["degraded"]]
    if unplanned:
        print(
            f"!! {len(unplanned)} of {n} questions were never planned - the planner was "
            f"unavailable and the frozen heuristics stood in. Every number below is measuring "
            f"that fallback, not the system. Re-run when the planner is back."
        )
        print(f"   unplanned: {', '.join(unplanned)}")

    print(f"Retrieval - top result matches expected sport: {mean(sport_matches):.0%} ({len(sport_matches)} single-sport questions)")
    print(f"Retrieval - avg expected-keyword coverage: {mean(keyword_coverages):.0%} ({n} questions)")
    print(f"  by type: {breakdown_by_type(list(zip(testset, keyword_coverages)))}")

    if not gemini.has_key():
        print("Generation checks skipped - set GEMINI_API_KEY_1 in .env to enable them.")
        return

    print(f"Judging {n} generated answers with Gemini (paced for free-tier rate limits, this takes a few minutes)...")
    generation_results = []
    repeated_scores: list[list[float]] = []
    for item, r in zip(testset, retrieval_results):
        result, scores = score_generation_repeated(item, r["chunks"], repeats)
        generation_results.append(result)
        repeated_scores.append(scores)

    judged = [g for g in generation_results if g]
    skipped = n - len(judged)
    if skipped:
        # Skipped questions leave the denominator, so a run that drops its
        # hard questions reports a better score than it earned. Say so.
        print(f"Generation - WARNING: {skipped} of {n} questions could not be judged and are excluded below.")

    if judged:
        for metric in ("faithfulness", "relevance", "correctness"):
            # Missing metric counts as 0 here and as unscored in the listing
            # below; keep both on the same default so a parse failure can't
            # depress the mean while hiding from the report that explains it.
            values = [g.get(metric, 0) for g in judged]
            print(f"Generation - avg {metric}: {mean(values):.2f} ({len(judged)} answers judged)")
        correctness = [g.get("correctness", 0) if g else None for g in generation_results]
        print(f"  correctness by type: {breakdown_by_type(list(zip(testset, correctness)), fmt='.2f')}")

        # The axis every reported failure lives on: refusing a question the
        # corpus can answer. Scored only over answerable types -- abstaining
        # on an `uncovered` question is the correct behaviour, not a miss.
        answerable = [
            (item, g.get("abstained", 0))
            for item, g in zip(testset, generation_results)
            if g and item["type"] in ANSWERABLE_TYPES
        ]
        if answerable:
            rate = mean([a for _, a in answerable])
            print(f"Generation - false-abstention rate: {rate:.0%} ({len(answerable)} answerable questions)")
            refused = [item["id"] for item, a in answerable if a]
            if refused:
                print(f"  refused: {', '.join(refused)}")

        # `uncovered` questions are NOT scored on abstention. The right answer
        # to "what was his series average" is not a refusal: it names the
        # coverage limit and then gives the closest fact held -- which the
        # reference does, so `correctness` already measures it. An earlier
        # version reported "correct abstention: 0%" for an answer that scored
        # correctness 1.00, which said more about the metric than the system.
        uncovered = [g.get("abstained", 0) for item, g in zip(testset, generation_results) if g and item["type"] == "uncovered"]
        if uncovered:
            print(
                f"Generation - uncovered questions answered with their coverage limit rather "
                f"than a flat refusal: {1 - mean(uncovered):.0%} ({len(uncovered)} questions)"
            )

        # A question that scores 1.00 on some runs and 0.00 on others is the
        # most misleading thing this eval can report, because either number
        # alone looks like a verdict. Name them.
        if repeats > 1:
            unstable = [
                (item["id"], scores)
                for item, scores in zip(testset, repeated_scores)
                if scores and len(set(scores)) > 1
            ]
            stable_pass = sum(1 for s in repeated_scores if s and set(s) == {1.0})
            print(f"Generation - questions correct on every one of {repeats} runs: {stable_pass}/{n}")
            for qid, scores in unstable:
                print(f"  UNSTABLE {qid}: {[f'{s:.1f}' for s in scores]}")

        # Name the questions that dragged a type's mean down -- without this
        # a regression shows up as a decimal with no way to chase it.
        weak = [(i, g) for i, g in zip(testset, generation_results) if g and g.get("correctness", 0) < 1]
        for item, scored in weak:
            print(f"  imperfect: {item['id']} ({scored.get('correctness')}) -> {scored['answer'][:140]}")


if __name__ == "__main__":
    import sys

    # e.g. `python -m eval.eval multihop existence` to re-check two types,
    # `python -m eval.eval --repeat 3 player-title` to measure variance, or
    # `python -m eval.eval --as-typed` to ask the same questions the way users
    # write them.
    args = sys.argv[1:]
    repeat_count = 1
    if "--repeat" in args:
        at = args.index("--repeat")
        repeat_count = int(args[at + 1])
        args = args[:at] + args[at + 2 :]
    typed = "--as-typed" in args
    args = [a for a in args if a != "--as-typed"]
    main(args or None, repeat_count, typed)
