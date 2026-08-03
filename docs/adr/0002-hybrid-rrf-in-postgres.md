# ADR 0002: Hybrid retrieval fused with RRF inside Postgres

**Status:** accepted · 2026-07-12

## Context

Legal queries come in two shapes: lexical ("Article 6(1)(f)") and semantic ("processing without consent"). Vector-only retrieval misses the first; keyword-only misses the second. Adding a dedicated search engine (Elasticsearch/OpenSearch) would double the infrastructure for a kit meant to be piloted quickly.

## Decision

One SQL statement (`app/retrieval/hybrid.py`): a pgvector cosine arm and a `tsvector` full-text arm, each over-fetching top-20, fused with Reciprocal Rank Fusion (k=60) — no score calibration needed. `SET hnsw.iterative_scan = relaxed_order` guards the vector arm against pgvector's filtered-query overfiltering failure (fixed in 0.8.0). One database holds app state, vectors, FTS, LangGraph checkpoints, and the audit log.

Reranking (Cohere v4 / local cross-encoder) is a documented flag, off by default: on a two-regulation corpus the RRF fusion already lands the right articles, and the eval suite is the place to justify the extra 200–400 ms, not a hunch.

## Consequences

- `docker compose up` is the entire retrieval infrastructure.
- RRF constants and arm sizes are config, and the retrieval eval (`evals/run_retrieval_eval.py`) measures hybrid vs single-arm ablations to keep the choice honest.
- Postgres FTS is English-configured; a German corpus needs a second `tsvector` column (known limitation).
- **The ablation does not yet justify the decision.** Measured on the 38 scored golden questions, hybrid ties vector-only on hit@6 and recall and trails it by 0.013 MRR, the whole difference being three questions. The premise above — that lexical queries exist and vector search misses them — is untested, because exactly one golden question cites an article number. The decision stands on the argument, not on the numbers, until the golden set carries citation-shaped questions.
