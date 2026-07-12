# ADR 0003: Structural chunking + contextual prefixes; late chunking rejected

**Status:** accepted · 2026-07-12

## Context

Fixed-size chunking on statutory text produces chunks that straddle article boundaries — and a citation that mixes Article 5 with Article 6 is legally wrong, not just imprecise. Both corpus regulations have a clean Chapter → Article → Paragraph hierarchy exposed in EUR-Lex markup.

## Decision

Chunk on article boundaries, packing whole numbered paragraphs up to ~700 tokens (`app/ingestion/chunker.py`); store `article_ref`/`heading` per chunk for exact citations. Every chunk gets a deterministic context prefix ("Regulation …, Art. 9 — Risk management system.") embedded with the content; an optional LLM pass (Anthropic-style contextual retrieval) adds one situating sentence per chunk at ingest, behind `--contextual` because it costs one model call per chunk.

Late chunking (long-context embedding models) was considered and rejected: it constrains the embedding model choice and adds pipeline complexity, while statutory structure plus contextual prefixes already carry the disambiguation late chunking is meant to recover.

## Consequences

- Citations resolve to exact articles with EUR-Lex deep links, for free.
- Chunks never cross articles; an oversized single paragraph splits on sentence boundaries only within its article.
- Recitals and annexes are not ingested in v1 (limitation — Annex III matters for high-risk classification questions and is called out in the README).
