# agentic-rag-kit

[![CI](https://github.com/fctpe/agentic-rag-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/fctpe/agentic-rag-kit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

**An enterprise-shaped agentic RAG kit you can actually pilot**: agentic retrieval over the EU AI Act and GDPR with article-level citations, durable human-in-the-loop approvals, grounding verification, eval suites, and an append-only audit trail — one Postgres, one compose file.

Most RAG demos answer questions. Regulated teams need the parts demos skip: *who approved this output, is every claim actually in the source, what happened when, and how do we know quality didn't regress after the last prompt change?* This kit makes those first-class: the LangGraph graph **is** the governance story, and the corpus (EU AI Act + GDPR) doubles as the compliance framework it's built to satisfy.

|  |  |
|---|---|
| ![Cited answer with the EUR-Lex citation panel](docs/ui-chat-citations.png) | ![Human-in-the-loop approval banner on a report request](docs/ui-approval-banner.png) |
| Every answer is grounded in article-level sources, shown in a live citation panel with EUR-Lex links. | Report-type output stops at a durable approval gate — approve or reject with a comment for the audit trail. |

```mermaid
flowchart LR
    Q[Question] --> G[guard_input<br/>PII redaction · injection checks]
    G --> A[Agent ⇄ tools<br/>search · read_article · compare]
    A --> R[(Postgres<br/>pgvector + FTS + RRF)]
    A --> P{report?}
    P -->|yes| H[Human approval<br/>durable interrupt]
    P -->|no| V[Grounding audit]
    H --> V
    V --> C[Answer with Art.-level<br/>EUR-Lex citations]
```

## What makes it production-shaped

- **Durable human-in-the-loop** — report-type output stops at a LangGraph `interrupt()` checkpointed in Postgres. Kill the backend mid-approval, restart, approve — the run resumes. (Verified exactly that way; try it.)
- **Grounding verification** — after every answer, a judge audits each claim against the retrieved article text and flags unsupported ones *to the user* instead of shipping them silently.
- **Hybrid retrieval in one SQL statement** — pgvector cosine + Postgres FTS fused with RRF (k=60), with `hnsw.iterative_scan` guarding against the classic filtered-query overfiltering failure. No second search engine.
- **Structure-aware ingestion** — chunks never cross article boundaries; every chunk carries `Art. N` metadata, so citations deep-link into EUR-Lex. Anthropic-style contextual prefixes are one flag away.
- **Security you can point at** — OWASP LLM Top 10 (2025) mapping ([docs/security.md](docs/security.md)), PII redaction before model/trace/checkpoint, JWT RBAC, and a hash-chained audit log framed against AI Act Art. 12/14: `GET /audit/verify` walks the chain and names the first entry an edit — or a deletion between two others — broke. It is tamper-*evidence*, not tamper-proofing; [docs/security.md](docs/security.md#audit-trail-eu-ai-act-art-12-framing) states exactly what it does not catch.
- **Telemetry that survives the request** — OpenTelemetry spans on the GenAI semantic conventions around every graph node, tool call, retrieval and the grounding verifier, with token and cost accounting from `usage_metadata`, plus JSON logs correlated by request and trace id. Exports OTLP to any collector — Langfuse included, which is why it stays. Unset endpoint means no exporter at all ([ADR 0006](docs/adr/0006-otel-genai-spans-and-structured-logs.md)).
- **Evals built to gate merges** — golden dataset with expected articles, retrieval ablations (vector-only vs hybrid), RAGAS metrics, and a red-team suite for injection/PII/refusal behavior. CI validates the dataset and compiles the suites on every push; wiring the scored runs in as a blocking gate needs a provider key and is a one-line CI change (documented in [docs/deployment.md](docs/deployment.md)).

## Quickstart

```bash
git clone https://github.com/fctpe/agentic-rag-kit && cd agentic-rag-kit
cp .env.example .env          # set OPENAI_API_KEY, JWT_SECRET
make db migrate seed
make ingest-smoke             # or `make ingest` for the full corpus + contextual prefixes
make api                      # :8000
make dev                      # :3000 — login as analyst@example.com / demo1234
```

Ask *"What obligations apply to providers of high-risk AI systems?"* — cited answer. Ask for *"a compliance report on prohibited practices"* — the approval banner appears; approve or reject with a comment.

## Results

![demo: backend test suite, retrieval ablation, RAGAS, and red-team results](docs/demo.gif)

All numbers below are from live runs against the full corpus (283 chunks, 212 articles), `gpt-4o-mini` as agent and judge. Raw outputs are committed under [`evals/results/`](evals/results), stamped with the run that produced them.

**RAGAS over the 38-question golden set** (0 chat failures, 2026-08-03):

| faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|
| 0.932 | 0.881 | 0.849 | 0.961 |

An earlier single run on 2026-07-12 scored faithfulness **0.964**. Re-running it twice on 2026-08-03 — same corpus, same `ragas` 0.4.3, retrieval byte-identical — gave **0.918** and **0.932**. On 38 questions that gap is one or two flipping, and with one sample in July against two in August there is no way to tell a lucky draw from a small real shift. So the honest figure is a range, not the best draw: **faithfulness 0.918–0.932 over two runs**.

That is the same lesson [`voice-desk-agent`](https://github.com/fctpe/voice-desk-agent) already encodes with `--runs 5` — an LLM-judged number from a single run is not an estimate. It had simply never been applied here. The thresholds in [`evals/thresholds.yaml`](evals/thresholds.yaml) are sized for that spread, which is why the gate passed on both runs rather than crying wolf.

**Retrieval ablation** (hit-rate@6 / MRR / article recall, 38 questions):

| mode | hit@6 | MRR | recall |
|---|---|---|---|
| hybrid (RRF) — production | **1.000** | 0.891 | 0.897 |
| vector only | 1.000 | 0.904 | 0.897 |
| text only (AND semantics) | 0.026 | 0.026 | 0.026 |

The text arm is nearly silent on natural-language questions **by design**: it exists for lexical/citation-style queries and fires with high precision there. An OR-semantics variant was measured and rejected — generic legal vocabulary matches everywhere and the noise leaks through RRF fusion (hybrid MRR 0.891 → 0.772). Negative results are results.

**Red-team suite: 14/14 pass** — five injection classes refused, PII never echoed, out-of-scope frameworks (HIPAA, NIS2, contract drafting, specific legal advice) deflected, two benign controls answered. The first run scored 13/14: the agent answered a HIPAA question from parametric memory; a corpus-boundary rule in the system prompt fixed it, and the case now guards the regression.

The eval suites live in [`evals/`](evals): `make eval` (RAGAS over the golden dataset), `uv run python evals/run_retrieval_eval.py` (retrieval ablations), `make redteam` (adversarial suite).

### What eval-driven iteration caught during development

1. **Fragments break enumerations.** Top-k search returns *parts* of long articles, so "list all prohibited practices" was systematically incomplete. Fix: a `read_article` tool the agent calls on the controlling article before enumerating. Together with the grounding-window fix below this lifted RAGAS faithfulness from 0.886 to 0.964.
2. **The grounding judge needs full text.** It originally audited against 300-char snippets and flagged **correct** enumerations (list items past the cutoff) as unsupported; it now sees full chunk content and judges factual support, not citation-number exactness.
3. **Partial corpus exposed parametric padding.** With only part of the corpus ingested, the model padded enumerations from memory — and the grounding audit caught exactly that. The flag is a feature.
4. **Refusal boundaries are probabilistic.** An adversarial-review pass plus a red-team re-run each surfaced a *different* out-of-scope question (HIPAA once, specific legal advice once) that the model answered instead of deflecting. Both are now firm prompt rules with regression cases, not hopeful phrasing.

## Regulatory accuracy

Docs and corpus reflect the **2026 Digital Omnibus** (adopted June 2026): Annex III high-risk obligations apply from 2 Dec 2027; Art. 50 transparency largely from Aug 2026. Recitals and annexes are not ingested in v1 (see limitations).

## Design decisions

Short ADRs in [`docs/adr/`](docs/adr): explicit StateGraph over prebuilt agents, hybrid RRF inside Postgres, structural chunking (+ why late chunking was rejected), SQLAlchemy over SQLModel, SSE over WebSockets, regex redaction with a Presidio seam, per-request budgets and the audit hash chain, OTel GenAI spans against conventions that are not stable yet. Architecture walkthrough in [docs/architecture.md](docs/architecture.md), deployment in [docs/deployment.md](docs/deployment.md) — compose, a kustomize base + local overlay under [`deploy/`](deploy), and a Terraform module for the managed Postgres, secrets and DNS. Security posture (RBAC and object-level authz, redaction scope, what the audit chain does not prove, request budgets) is summarised in [SECURITY.md](SECURITY.md) and covered in full in [docs/security.md](docs/security.md).

## Limitations

- **Ingestion from live EUR-Lex is currently blocked upstream.** As of 2026-08-03 EUR-Lex answers the document URLs with `HTTP 202` and an empty body — bot protection or a queued render, not an error status. `make ingest` therefore fails with a `FetchError` naming the cause. The parser is unchanged and the committed corpus and eval results were produced on 2026-07-12 when the endpoint still served HTML. Working around the block is not something this repo does; retry later, or point `eurlex.py` at a local copy.
- Corpus is articles-only (no recitals/annexes) and English-only; Annex III questions answer from Art. 6 references, not the annex text itself.
- The router's report-detection is heuristic-first; unusual phrasings can miss the approval gate. The gate is policy, not a security boundary — RBAC is.
- PII redaction is structural (emails, phones, IBANs, cards); free-text names need the documented Presidio swap.
- Grounding audit adds one model call of latency to every answer, and rejected reports end the run — there is no revision loop yet.
- Single-tenant by design; per-document ACLs and SSO are out of scope for the kit.
- The audit chain is unkeyed sha256 over columns the database can rewrite, so it detects edits made *around* the application, not an attacker who recomputes the chain — and deleting the newest entries leaves the rest walking clean (ADR 0005).
- The GenAI semantic conventions are still Development status, so the span attribute names can move; the OTel packages are pinned exactly and a test holds the names to the generated constants (ADR 0006). Span cost is whatever you configure — the repo ships no price table.

## Roadmap

- Style-matching RAG over previously **approved** reports, so drafts converge on the reviewing team's voice.
- Revision loop on rejection (reviewer comment feeds a redraft pass).
- German corpus variant (second `tsvector` configuration).

---

AI-assisted scaffolding; architecture, retrieval design, prompts, evals, and the governance model are hand-designed — the ADRs record the reasoning.

[MIT](LICENSE)
