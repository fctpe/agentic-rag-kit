# ADR 0004: SQLAlchemy 2.0 over SQLModel; SSE over WebSockets; regex redaction with a Presidio swap point

**Status:** accepted · 2026-07-12

## SQLAlchemy 2.0 (async) instead of SQLModel

The schema needs pgvector columns, a generated `tsvector` column, an HNSW index with custom ops, and Alembic autogeneration over all of it. SQLModel's abstraction adds nothing here and its release cadence lags SQLAlchemy; typed `mapped_column` declarations give the same ergonomics with none of the impedance.

## SSE instead of WebSockets

Chat streaming is strictly server→client: tokens, citations, approval events. SSE works through proxies and load balancers, reconnects trivially, and needs no connection-state management. Bidirectionality — the only WebSocket argument — is not needed: approval decisions are ordinary POSTs to `/chat/{thread}/resume`.

## Regex-based PII redaction by default, Presidio behind the same seam

Structural identifiers (emails, phones, IBANs, cards) cover the PII that actually appears in compliance questions, with zero heavy dependencies — Presidio pulls spaCy plus a model download, which would wreck the <5-minute quickstart. `redact_pii()` (`app/security/redaction.py`) is the single seam: swapping in a `PresidioRedactor` changes one function, and docs/security.md documents the trade-off honestly.

**Where it runs was wrong, and the fix is the interesting part.** This ADR originally said redaction runs in `guard_input`, BEFORE the text can reach the model, the checkpointer, or a Langfuse trace. The first node of a checkpointed graph is not the first thing that writes: LangGraph persists the input super-step before any node runs, so `graph.astream({"messages": [HumanMessage(body.message)]})` put the raw message in `checkpoints` at steps −1 and 0, and the redacted version only appeared at step 1. The model and the trace never saw raw text; the durable store did, which is the one that keeps it.

Redaction therefore belongs at the **API boundary** — `/chat` redacts once and passes the clean text plus the labels it stripped into the graph. `guard_input` still calls `redact_pii` and unions the result, so the node remains a real boundary for any caller that bypasses the route, but it is no longer the only thing standing between raw input and a durable write.

The general lesson, worth more than the fix: *"redaction runs before the first node"* and *"redaction runs before the first write"* are different claims, and a framework decides which one you actually get. The test that now pins it (`backend/tests/test_redaction_boundary.py`) asserts against checkpoint contents rather than node logic, and includes the negative control that fails if the input super-step ever stops being persisted — because an invariant asserted only in the passing direction is how this survived review the first time.
