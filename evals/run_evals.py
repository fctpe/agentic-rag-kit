"""End-to-end RAGAS evaluation against the live chat API.

For each non-out-of-scope golden question the script:
  1. POSTs /chat on the running backend (SSE) and captures the final answer
     plus the citations the agent actually produced.
  2. Retrieves contexts for the same query directly via hybrid_search — full
     chunk contents, not the 300-char citation snippets, so the RAGAS judges
     see the same text the agent's tools saw.
  3. Scores the (question, answer, contexts, reference) tuples with RAGAS
     0.4.x metrics: faithfulness, answer_relevancy, context_precision,
     context_recall.

RAGAS 0.4.x API used (verified against the installed package):
    from ragas.llms.base import llm_factory                # instructor-based judge LLM
    from ragas.embeddings.base import embedding_factory    # modern embeddings interface
    from ragas.metrics.collections import (
        AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness)
    result = await metric.ascore(...)                      # -> MetricResult(.value)

Note: ragas 0.4.3 hard-imports langchain_community.chat_models.vertexai, which
langchain-community 0.4.x (required by the backend's langchain 1.x stack)
removed. _shim_vertexai() below registers a stub module so `import ragas`
works; the stub class is only ever used by ragas in isinstance() checks.

Note: the judge's max_tokens is set explicitly. ragas defaults its instructor
judge to 1024 output tokens, and faithfulness' NLI step has to re-emit every
atomic statement of the answer verbatim, each with a free-text reason — output
that scales with the ANSWER, which the agent writes and nothing bounds. Long
answers blew that budget, and the failure landed on exactly the long
multi-hop questions, so what survived was the easy half. See score_one() for
the other half of that fix: a judge call that cannot be completed is recorded
and fails the run, never quietly dropped from a mean.

Prerequisites: backend running on --api-base (default http://localhost:8000)
with seeded users, Postgres on localhost:5434 with ingested chunks, and
OPENAI_API_KEY exported (judges + query embedding). Costs money — see README.

    cd backend && uv run --group evals python ../evals/run_evals.py --smoke
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import types
import warnings
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import gate_ragas  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_questions.yaml"
RESULTS_DIR = EVALS_DIR / "results"

DEFAULT_API_BASE = os.environ.get("EVAL_API_BASE", "http://localhost:8000")
EVAL_EMAIL = os.environ.get("EVAL_EMAIL", "analyst@example.com")
EVAL_PASSWORD = os.environ.get("EVAL_PASSWORD", "demo1234")
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
JUDGE_EMBEDDING_MODEL = os.environ.get("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
# ragas' own default is 1024 and its docstring recommends 4096+ for structured
# output. The worst golden answer measured here decomposes into 27 statements
# that the NLI step must re-emit with a reason each; that does not fit in 1024.
JUDGE_MAX_TOKENS = int(os.environ.get("RAGAS_JUDGE_MAX_TOKENS", "4096"))
JUDGE_ATTEMPTS = int(os.environ.get("RAGAS_JUDGE_ATTEMPTS", "3"))
JUDGE_RETRY_BASE_DELAY = float(os.environ.get("RAGAS_JUDGE_RETRY_DELAY", "2.0"))

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# The inline source marker. The frontend turns each [n] into the link between a
# sentence and the source panel entry it came from, so an answer that cites in
# prose instead ("(AI Act, Art. 50(1))") ships its sources unlinked. None of the
# RAGAS metrics can see that: faithfulness judges whether the claim is
# supported, not whether the answer said where from — which is how free-form
# answers drifted to prose with four green metrics above them.
#
# The marker is no longer a request to the model: app/agent/markers.py resolves
# merged brackets and strips indices that resolve to nothing before the text
# leaves the backend, and backend/tests/test_citation_markers.py holds it to
# that offline and free. This check stays as the end-to-end backstop, because a
# guarantee that is only ever asserted against fixtures is a guarantee about
# fixtures.
INLINE_CITATION = re.compile(r"\[\d+\]")


def _shim_vertexai() -> None:
    """Make `import ragas` survive the removed langchain_community vertexai module."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401

        return
    except Exception:
        pass
    shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # placeholder: ragas only uses this in isinstance() checks
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = shim


# --------------------------------------------------------------- chat API ---


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse a text/event-stream body into (event, data) pairs.

    Protocol per backend/app/api/chat.py: blocks of
    "event: <name>\\ndata: <json>\\n\\n" with events token, approval_required,
    citations, grounding, done, error.
    """
    events: list[tuple[str, dict]] = []
    event_name, data_lines = None, []
    for line in raw.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "" and event_name is not None:
            try:
                payload = json.loads("\n".join(data_lines)) if data_lines else {}
            except json.JSONDecodeError:
                payload = {"_raw": "\n".join(data_lines)}
            events.append((event_name, payload))
            event_name, data_lines = None, []
    return events


def digest_events(events: list[tuple[str, dict]]) -> dict:
    """Reduce the SSE stream to answer/citations/grounding.

    Answer preference: `done.content` (final persisted answer) >
    `approval_required.draft` (report route interrupted at the human gate) >
    concatenated `token` events (defensive fallback).
    """
    tokens: list[str] = []
    out = {
        "answer": "",
        "citations": [],
        "grounded": None,
        "grounding_issues": [],
        # Brackets the backend refused to turn into links, with the reason.
        # Separate from grounding_issues on the wire and separate here.
        "citation_issues": [],
        "approval_interrupted": False,
        "error": None,
    }
    for name, data in events:
        if name == "token":
            tokens.append(data.get("text", ""))
        elif name == "citations":
            out["citations"] = data.get("citations", [])
        elif name == "grounding":
            out["grounded"] = data.get("grounded")
            out["grounding_issues"] = data.get("issues", [])
        elif name == "done":
            out["answer"] = data.get("content", "")
            out["citation_issues"] = data.get("citation_issues", [])
        elif name == "approval_required":
            out["approval_interrupted"] = True
            if not out["answer"]:
                out["answer"] = data.get("draft", "")
                out["citation_issues"] = data.get("citation_issues", [])
        elif name == "error":
            out["error"] = data.get("message", "unknown SSE error")
    if not out["answer"]:
        out["answer"] = "".join(tokens)
    return out


async def login(client: httpx.AsyncClient, api_base: str) -> str:
    response = await client.post(
        f"{api_base}/auth/login", json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD}
    )
    response.raise_for_status()
    return response.json()["token"]


async def ask_chat(client: httpx.AsyncClient, api_base: str, token: str, message: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    async with client.stream(
        "POST",
        f"{api_base}/chat",
        json={"message": message},
        headers=headers,
        timeout=httpx.Timeout(10.0, read=300.0),
    ) as response:
        response.raise_for_status()
        raw = (await response.aread()).decode("utf-8", errors="replace")
    return digest_events(parse_sse(raw))


# ------------------------------------------------------------------ ragas ---


def build_judges():
    _shim_vertexai()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI()
    judge_llm = llm_factory(JUDGE_MODEL, client=client, max_tokens=JUDGE_MAX_TOKENS)
    judge_embeddings = embedding_factory(
        "openai", model=JUDGE_EMBEDDING_MODEL, client=client, interface="modern"
    )
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
    }


def transient_error_types() -> tuple[type[BaseException], ...]:
    """Judge failures worth a retry: the provider was busy, not the sample.

    Everything else is deterministic for a given sample at a given max_tokens —
    a truncated structured output or a schema violation will reproduce exactly,
    so retrying it only spends money to arrive at the same failure. Imported
    lazily to keep this module importable (and testable) without touching the
    ragas import chain.
    """
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    return (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def is_retryable(err: BaseException) -> bool:
    return isinstance(err, transient_error_types())


async def score_one(metric, sample_id: str, call: dict) -> float:
    """One metric for one sample, or an exception. Never a substitute value.

    The point of raising is that the caller has to decide what to do about it
    in the open. Returning 0.0 would drag the mean down for a provider hiccup;
    returning None used to drop the question out of this metric's denominator
    while it stayed in the other three, so the run reported a full population
    and published a mean over whatever happened to survive.
    """
    result = None
    for attempt in range(1, JUDGE_ATTEMPTS + 1):
        try:
            result = await metric.ascore(**call)
            break
        except Exception as err:
            if attempt == JUDGE_ATTEMPTS or not is_retryable(err):
                raise
            delay = JUDGE_RETRY_BASE_DELAY * 2 ** (attempt - 1)
            print(
                f"  [{sample_id}] transient judge error ({type(err).__name__}), "
                f"retrying in {delay:.0f}s ({attempt}/{JUDGE_ATTEMPTS - 1})",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
    value = float(result.value)
    if math.isnan(value):
        # ragas returns nan when a metric could not be computed at all (e.g. the
        # answer decomposed into zero statements). nan is not None, so it used
        # to pass the aggregator's filter and poison the mean.
        raise ValueError("judge returned nan — the metric could not be computed")
    return value


async def score_sample(metrics: dict, sample: dict) -> tuple[dict[str, float | None], list[dict]]:
    """(scores, failures). A metric that could not be scored appears in BOTH.

    The per-sample null stays in the artifact so the table still lines up, but
    the failure is also returned as a record, because a null nothing counts is
    how a lost judge call gets mistaken for a good score.
    """
    scores: dict[str, float | None] = {}
    failures: list[dict] = []
    calls = {
        "faithfulness": dict(
            user_input=sample["question"],
            response=sample["answer"],
            retrieved_contexts=sample["contexts"],
        ),
        "answer_relevancy": dict(user_input=sample["question"], response=sample["answer"]),
        "context_precision": dict(
            user_input=sample["question"],
            reference=sample["reference"],
            retrieved_contexts=sample["contexts"],
        ),
        "context_recall": dict(
            user_input=sample["question"],
            retrieved_contexts=sample["contexts"],
            reference=sample["reference"],
        ),
    }
    for name in METRIC_NAMES:
        try:
            scores[name] = await score_one(metrics[name], sample["id"], calls[name])
        except Exception as err:
            print(f"  [{sample['id']}] {name} failed: {type(err).__name__}: {err}", file=sys.stderr)
            scores[name] = None
            failures.append(
                {
                    "id": sample["id"],
                    "metric": name,
                    "error_type": type(err).__name__,
                    "error": str(err)[:500],
                }
            )
    return scores, failures


# ------------------------------------------------------------------- main ---


def load_questions(smoke: bool, only_ids: list[str] | None) -> list[dict]:
    data = yaml.safe_load(GOLDEN_PATH.read_text())
    questions = [q for q in data["questions"] if q["query_type"] != "out_of_scope"]
    if only_ids:
        wanted = set(only_ids)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            raise SystemExit(f"Unknown/out-of-scope question ids: {sorted(missing)}")
    if smoke:
        questions = questions[:3]
    return questions


async def collect_sample(
    question: dict,
    client: httpx.AsyncClient,
    api_base: str,
    token: str,
    session_factory,
    k: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    from app.retrieval.hybrid import hybrid_search

    async with semaphore:
        chat = await ask_chat(client, api_base, token, question["question"])
        async with session_factory() as session:
            chunks = await hybrid_search(session, question["question"], final_k=k)
        contexts = [f"[{chunk.regulation} {chunk.article_ref}] {chunk.content}" for chunk in chunks]
        print(
            f"  [{question['id']}] answer: {len(chat['answer'])} chars, "
            f"citations: {len(chat['citations'])}, contexts: {len(contexts)}"
        )
        return {
            "id": question["id"],
            "question": question["question"],
            "answer": chat["answer"],
            "contexts": contexts,
            "reference": question["reference_answer"],
            "citations": chat["citations"],
            "grounded": chat["grounded"],
            "citation_issues": chat["citation_issues"],
            "approval_interrupted": chat["approval_interrupted"],
            "chat_error": chat["error"],
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="first 3 questions only")
    parser.add_argument("--questions", nargs="+", default=None, help="run only these ids")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--k", type=int, default=6, help="contexts per question (default 6)")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (RAGAS judges + query embedding).")

    questions = load_questions(args.smoke, args.questions)
    print(f"Evaluating {len(questions)} questions against {args.api_base} (judge: {JUDGE_MODEL})")

    from app.db import dispose_engine, get_session_factory

    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        try:
            token = await login(client, args.api_base)
        except (httpx.HTTPError, KeyError) as err:
            raise SystemExit(
                f"Login failed against {args.api_base} as {EVAL_EMAIL}: {err}. "
                "Is the backend running and seeded (uv run python -m app.seed)?"
            ) from err
        samples = await asyncio.gather(
            *(
                collect_sample(
                    q, client, args.api_base, token, get_session_factory(), args.k, semaphore
                )
                for q in questions
            )
        )
    await dispose_engine()

    failed_chats = [s for s in samples if s["chat_error"] or not s["answer"]]
    for sample in failed_chats:
        print(
            f"  [{sample['id']}] chat failed, excluded from scoring: "
            f"{sample['chat_error'] or 'empty answer'}",
            file=sys.stderr,
        )
    scorable = [s for s in samples if s not in failed_chats]

    metrics = build_judges()
    judge_semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded_score(sample: dict) -> tuple[dict, list[dict]]:
        async with judge_semaphore:
            return await score_sample(metrics, sample)

    judge_failures: list[dict] = []
    all_scores = await asyncio.gather(*(guarded_score(s) for s in scorable))
    for sample, (scores, failures) in zip(scorable, all_scores, strict=True):
        sample["scores"] = scores
        judge_failures.extend(failures)

    # aggregate
    def contributing(name: str) -> list[float]:
        return [s["scores"][name] for s in scorable if s["scores"].get(name) is not None]

    def mean(name: str) -> float | None:
        values = contributing(name)
        return round(sum(values) / len(values), 4) if values else None

    summary = {name: mean(name) for name in METRIC_NAMES}
    # Every mean states its own denominator. They are not all len(scorable): a
    # metric whose judge call died loses that question from its mean alone, and
    # a run-level count cannot see it happen.
    n_contributing = {name: len(contributing(name)) for name in METRIC_NAMES}

    print("\n## RAGAS results\n")
    header = "| id | " + " | ".join(METRIC_NAMES) + " |"
    print(header)
    print("|" + "----|" * (len(METRIC_NAMES) + 1))
    for sample in scorable:
        cells = [
            f"{sample['scores'][name]:.3f}" if sample["scores"][name] is not None else "err"
            for name in METRIC_NAMES
        ]
        print(f"| {sample['id']} | " + " | ".join(cells) + " |")
    mean_cells = [
        f"{summary[name]:.3f}" if summary[name] is not None else "-" for name in METRIC_NAMES
    ]
    print("| **mean** | " + " | ".join(f"**{c}**" for c in mean_cells) + " |")
    print(
        "| **scored** | "
        + " | ".join(f"{n_contributing[name]}/{len(scorable)}" for name in METRIC_NAMES)
        + " |"
    )

    if judge_failures:
        print(
            f"\n{len(judge_failures)} judge call(s) returned no score — the means above are "
            f"NOT over all {len(scorable)} questions:",
            file=sys.stderr,
        )
        for failure in judge_failures:
            print(
                f"  [{failure['id']}] {failure['metric']}: "
                f"{failure['error_type']}: {failure['error']}",
                file=sys.stderr,
            )

    # An answer the grounding verifier rejected is not an answer — it is the
    # refusal that replaced one, and it has no sources to mark because it makes
    # no claims. Counting it as a missing citation would make the fail-closed
    # path look like a formatting defect, and the cheapest way to turn that
    # gate green would be to weaken the verifier. That is the one incentive
    # this suite must never create.
    #
    # Reported separately rather than dropped: an answer count that quietly
    # shrinks is the defect this same file was just fixed for.
    citable = [s for s in scorable if s.get("grounded") is not False]
    ungrounded = [s["id"] for s in scorable if s.get("grounded") is False]
    unmarked = [s["id"] for s in citable if not INLINE_CITATION.search(s["answer"])]
    print(
        f"\ninline citations: {len(citable) - len(unmarked)}/{len(citable)} answers carry an "
        "[n] marker" + (f" — missing: {', '.join(unmarked)}" if unmarked else "")
    )
    if ungrounded:
        print(
            f"  ({len(ungrounded)} answer(s) not counted — the grounding verifier "
            f"refused them, so they cite nothing: {', '.join(ungrounded)})"
        )

    # Per BRACKET, not per answer. The line above is satisfied by one good
    # marker anywhere, so an answer could ship ten unlinkable `[2(a)]…[2(j)]`
    # brackets and still count as marked — G09 in run 075206 did exactly that.
    # The measured denominator was answers; the thing that breaks is brackets.
    #
    # Additive: `answers_without_inline_citation` keeps its meaning and its
    # gate, so no committed number changes shape. This is the finer-grained
    # observation the gate did not have.
    from app.agent.markers import unlinkable_brackets

    unlinked = [
        {"id": s["id"], "bracket": bracket}
        for s in citable
        for bracket in unlinkable_brackets(
            s["answer"], {c["index"]: c for c in s.get("citations") or []}
        )
    ]
    print(
        f"unlinkable brackets: {len(unlinked)}"
        + (f" — {', '.join(f'{u["id"]} {u["bracket"]}' for u in unlinked)}" if unlinked else "")
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"ragas_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = bool(args.smoke or args.questions)
    payload = {
        "summary": summary,
        "judge_model": JUDGE_MODEL,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "api_base": args.api_base,
        "n_scored": len(scorable),
        "n_contributing": n_contributing,
        "n_chat_failures": len(failed_chats),
        "n_judge_failures": len(judge_failures),
        "judge_failures": judge_failures,
        "answers_without_inline_citation": unmarked,
        # Every citation-shaped bracket the frontend will not turn into a link,
        # named individually. Observed, not gated: the guarantee is enforced in
        # app/agent/markers.py and tested offline, and adding a gate key here
        # would mean editing thresholds.yaml against a baseline that has not
        # been re-measured.
        "unlinkable_citation_brackets": unlinked,
        # Denominator for the line above, and the ids it excluded. Without both,
        # "0 answers missing a marker" cannot be told apart from "0 answers".
        "n_citable": len(citable),
        "answers_refused_as_ungrounded": ungrounded,
        "timestamp": timestamp,
        "partial": partial,
        "samples": samples,
    }
    # allow_nan=False: bare NaN is not JSON, and an artifact that no strict
    # parser will read is not a result. Better to blow up here than to commit
    # a file whose numbers depend on who parses it.
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"\nwrote {out_path}")

    gate = gate_ragas(payload, partial=partial)
    gate.report()
    if partial:
        # A subset run is informational: it cannot be judged against a full-run
        # baseline, so it must not gate — but it must not claim to have gated
        # either. Thresholds are the only thing a subset run gets to skip: a
        # question that was asked and not scored is broken at any sample size.
        print("\n(subset run — thresholds not applied)")
        return 1 if failed_chats or judge_failures else 0
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
