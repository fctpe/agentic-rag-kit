# ADR 0004: SQLAlchemy 2.0 over SQLModel; SSE over WebSockets; regex redaction with a Presidio swap point

**Status:** accepted · 2026-07-12

## SQLAlchemy 2.0 (async) instead of SQLModel

The schema needs pgvector columns, a generated `tsvector` column, an HNSW index with custom ops, and Alembic autogeneration over all of it. SQLModel's abstraction adds nothing here and its release cadence lags SQLAlchemy; typed `mapped_column` declarations give the same ergonomics with none of the impedance.

## SSE instead of WebSockets

Chat streaming is strictly server→client: tokens, citations, approval events. SSE works through proxies and load balancers, reconnects trivially, and needs no connection-state management. Bidirectionality — the only WebSocket argument — is not needed: approval decisions are ordinary POSTs to `/chat/{thread}/resume`.

## Regex-based PII redaction by default, Presidio behind the same seam

Structural identifiers (emails, phones, IBANs, cards) cover the PII that actually appears in compliance questions, with zero heavy dependencies — Presidio pulls spaCy plus a model download, which would wreck the <5-minute quickstart. `redact_pii()` (`app/security/redaction.py`) is the single seam: swapping in a `PresidioRedactor` changes one function, and docs/security.md documents the trade-off honestly. Redaction runs in `guard_input`, BEFORE the text can reach the model, the checkpointer, or a Langfuse trace.
