# ADR 0001: Explicit LangGraph StateGraph, not a prebuilt agent

**Status:** accepted · 2026-07-12

## Context

The agent needs governance features that are graph-shaped: input guarding before anything reaches the model, a human approval gate for report-type output, and a grounding check before anything reaches the user. LangGraph 1.x deprecated `langgraph.prebuilt`/`create_react_agent` in favor of `create_agent` in `langchain.agents`; alternatives (OpenAI Agents SDK, Pydantic-AI, CrewAI) have weaker durable-persistence stories.

## Decision

Hand-build the `StateGraph` (`app/agent/graph.py`): guard_input → router → agent⇄tools loop → approval_gate (`interrupt()`) → verify. Compile with `AsyncPostgresSaver` so interrupts are durable. The tool-execution node is ~20 lines of our own code rather than a prebuilt — tool errors are surfaced back to the model as tool messages for self-correction.

## Consequences

- The graph is visible, testable per-node, and each governance control is a node you can point at in a review — the graph *is* the compliance narrative (AI Act Art. 14 human-oversight framing).
- A pending approval survives backend restarts; verified by killing uvicorn mid-interrupt and resuming after restart.
- We own ~40 lines a prebuilt would provide; the trade is control over the loop-termination policy (MAX_TOOL_ROUNDS) and message flow.

## Addendum, 2026-08-04: "before anything reaches the user" was two different promises

The Context above says the graph gives *a grounding check before anything reaches the user*. Node order delivered that; the SSE transport did not. `/chat` streamed tokens straight from the agent node, so both the approval gate and the grounding check ran **after** the text was already on screen — the frontend even deleted the streamed bubble once `approval_required` arrived, which is a draft rendered and then withdrawn, not a draft withheld. Graph order is not delivery order once a transport streams.

Splitting the promise in two is what made it fixable, because the two halves have different right answers:

- **Reports are withheld.** The router settles `task_type` before the agent produces a token, so `/chat` suppresses streaming for a report and emits `drafting` instead. The draft reaches the user in the `approval_required` payload or not at all. This matters more now that an admin can decide another user's approval — while author and reviewer were necessarily the same person, showing someone their own draft early was harmless.
- **QA answers stream, and say they are unverified.** Buffering every answer until `verify` returns would trade the product's core interaction for a wording fix. The honest claim is the one now in `chat.py`'s docstring: shown unverified, then verified before the run completes — and the UI renders that intermediate state ("Checking sources…", then "Unverified" if no verdict ever lands) rather than letting an unchecked answer look identical to a checked one.

`backend/tests/test_stream_ordering.py` pins the report half against the real SSE assembly, including a negative control that fails if qa stops streaming — otherwise "no tokens before approval" is satisfied by a stream that emits nothing at all.

## Addendum: two node behaviours that came out of eval failures, not design

Both were found by running the golden set, and both are decisions rather than fixes, so they are recorded here rather than narrated in the README.

- **`read_article` exists because top-k search returns *fragments*.** "List all prohibited practices" was systematically incomplete: the retriever returns the parts of a long article that match, and an enumeration needs the whole thing. The agent now calls `read_article` on the controlling article before enumerating. The before/after RAGAS runs that motivated it predate the committed results and no artifact survives them, so no delta is quoted for it.
- **The grounding judge audits full chunk content, not the citation snippets.** It originally judged against the 300-character excerpts the citation panel shows, and flagged *correct* enumerations as unsupported whenever the list ran past the cutoff. It now sees full chunk text and judges factual support. The related observation is why the check is worth its latency at all: with only part of the corpus ingested, the model padded enumerations from parametric memory and the audit caught exactly that.
