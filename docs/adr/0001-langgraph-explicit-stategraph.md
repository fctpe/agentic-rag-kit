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
