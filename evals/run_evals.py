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

Prerequisites: backend running on --api-base (default http://localhost:8000)
with seeded users, Postgres on localhost:5434 with ingested chunks, and
OPENAI_API_KEY exported (judges + query embedding). Costs money — see README.

    cd backend && uv run --group evals python ../evals/run_evals.py --smoke
"""

import argparse
import asyncio
import json
import os
import sys
import types
import warnings
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_questions.yaml"
RESULTS_DIR = EVALS_DIR / "results"

DEFAULT_API_BASE = os.environ.get("EVAL_API_BASE", "http://localhost:8000")
EVAL_EMAIL = os.environ.get("EVAL_EMAIL", "analyst@example.com")
EVAL_PASSWORD = os.environ.get("EVAL_PASSWORD", "demo1234")
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
JUDGE_EMBEDDING_MODEL = os.environ.get("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


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
        elif name == "approval_required":
            out["approval_interrupted"] = True
            if not out["answer"]:
                out["answer"] = data.get("draft", "")
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
    judge_llm = llm_factory(JUDGE_MODEL, client=client)
    judge_embeddings = embedding_factory(
        "openai", model=JUDGE_EMBEDDING_MODEL, client=client, interface="modern"
    )
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
    }


async def score_sample(metrics: dict, sample: dict) -> dict:
    scores: dict[str, float | None] = {}
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
            result = await metrics[name].ascore(**calls[name])
            scores[name] = float(result.value)
        except Exception as err:
            print(f"  [{sample['id']}] {name} failed: {err}", file=sys.stderr)
            scores[name] = None
    return scores


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

    async def guarded_score(sample: dict) -> dict:
        async with judge_semaphore:
            return await score_sample(metrics, sample)

    all_scores = await asyncio.gather(*(guarded_score(s) for s in scorable))
    for sample, scores in zip(scorable, all_scores):
        sample["scores"] = scores

    # aggregate
    def mean(name: str) -> float | None:
        values = [s["scores"][name] for s in scorable if s["scores"].get(name) is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary = {name: mean(name) for name in METRIC_NAMES}

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

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"ragas_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "judge_model": JUDGE_MODEL,
                "api_base": args.api_base,
                "n_scored": len(scorable),
                "n_chat_failures": len(failed_chats),
                "timestamp": timestamp,
                "samples": samples,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")
    return 1 if failed_chats else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
