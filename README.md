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

- **Durable human-in-the-loop** — report-type output stops at a LangGraph `interrupt()` checkpointed in Postgres. Kill the backend mid-approval, restart, approve — the run resumes. (Verified exactly that way; try it.) A report is never streamed: the draft reaches the user in the approval payload or not at all, and an admin can decide another user's approval, so reviewer and author need not be the same person ([ADR 0001 addendum](docs/adr/0001-langgraph-explicit-stategraph.md)).
- **Grounding verification** — after every answer, a judge checks each claim against the retrieved article text and reports the ones **no** source supports *to the user* instead of shipping them silently. It verifies support, not attribution: a claim that cites the wrong article passes as long as some retrieved source backs the claim itself. That is deliberate — checking the numbers made it flag correct enumerations — and [docs/security.md](docs/security.md) states the boundary. A qa answer streams before the verdict exists, and the UI shows that window as unverified rather than rendering an unchecked answer identically to a checked one.
- **Hybrid retrieval in one SQL statement** — pgvector cosine + Postgres FTS fused with RRF (k=60), with `hnsw.iterative_scan` guarding against the classic filtered-query overfiltering failure. No second search engine.
- **Structure-aware ingestion** — chunks never cross the boundary of the unit they came from, and every chunk carries that unit's ref (`Art. 6`, `Annex III`), so citations deep-link into the right place in EUR-Lex (`#art_6`, `#anx_III`). Annexes are ingested and chunked as units in their own right, not folded into the last article: the AI Act's high-risk list is Annex III, and Art. 6(2) only points at it. Anthropic-style contextual prefixes are one flag away.
- **Security you can point at** — OWASP LLM Top 10 (2025) mapping ([docs/security.md](docs/security.md)), PII redaction at the API boundary — before the model, a trace, *and* the checkpointer, which is the one that took a test against real checkpoint contents to get right ([ADR 0004](docs/adr/0004-sqlalchemy-sse-redaction-choices.md)) — JWT RBAC, and a hash-chained audit log framed against AI Act Art. 12/14: `GET /audit/verify` walks the chain and names the first entry an edit — or a deletion between two others — broke. It is tamper-*evidence*, not tamper-proofing; [docs/security.md](docs/security.md#audit-trail-eu-ai-act-art-12-framing) states exactly what it does not catch.
- **Telemetry that survives the request** — OpenTelemetry spans on the GenAI semantic conventions around every graph node, tool call, retrieval and the grounding verifier, with token and cost accounting from `usage_metadata`, plus JSON logs correlated by request and trace id. Exports OTLP to any collector — Langfuse included, which is why it stays. Unset endpoint means no exporter at all ([ADR 0006](docs/adr/0006-otel-genai-spans-and-structured-logs.md)).
- **Evals built to gate merges** — golden dataset with expected articles, retrieval ablations (vector-only vs hybrid), RAGAS metrics, and a red-team suite for injection/PII/refusal behavior. CI validates the dataset and compiles the suites on every push; wiring the scored runs in as a blocking gate needs a provider key and is a one-line CI change (documented in [docs/deployment.md](docs/deployment.md)).

## Quickstart

```bash
git clone https://github.com/fctpe/agentic-rag-kit && cd agentic-rag-kit
cp .env.example .env          # set OPENAI_API_KEY, JWT_SECRET
make db migrate seed
make ingest-smoke             # 10 units per regulation, straight from the committed corpus
make api                      # :8000
make dev                      # :3000 — login as analyst@example.com / demo1234
```

The corpus is committed under [`data/fixtures/`](data/fixtures) — the EU AI Act and GDPR as this repo's parser produced them — so a clean clone ingests without depending on EUR-Lex being in a good mood. It holds **212 articles and 13 annexes → 280 chunks**: articles (`Art. 6`) and annexes (`Annex III`) are separate citable units, because Art. 6(2) makes an AI system high-risk by pointing at the list in Annex III and the list exists nowhere else. `make ingest-fixture` loads all of it; `make ingest` re-fetches the same units from live EUR-Lex. The two fixture targets need **no provider key at all** — see the two paragraphs below; `make ingest` does, because it generates prefixes and embeddings as it goes.

**The contextual prefixes are committed too**, in [`data/fixtures/context_prefixes.json`](data/fixtures) — one LLM-written sentence per chunk, each keyed by a hash of the chunk it describes. They used to be written at ingest time, and `temperature=0` is not determinism: there is no seed and no provider guarantees one, so two ingests of the *same* committed fixture produced different prefixes, different embeddings and different retrieval numbers (hybrid MRR 0.888 on one ingest, 0.914 on the next — same fixture, same code, same SQL). That is wider than the floors in `evals/thresholds.yaml`, which are set on the premise that retrieval is deterministic, and it broke the promise the committed corpus exists for: a reviewer could not reproduce a committed number, because they could not reproduce the corpus. `make ingest-fixture` now *reads* those sentences and **fails closed** if the cache does not cover the corpus, naming the missing chunks and the command to fix it — it does not call the model on a miss (that is the non-determinism being removed) and does not degrade to the deterministic prefix (that is an unlabelled third corpus). `make ingest` still generates them, because a fresh EUR-Lex parse has no cache by definition and is not what a committed number describes. Regenerating is one model call per chunk, so it is its own target — `make prefix-cache` — and deliberately not part of the free `make refresh-fixtures`; `backend/tests/test_prefix_cache.py` fails offline when the two drift apart.

**And the embedding vectors are committed too**, in [`data/fixtures/chunk_embeddings.json`](data/fixtures) — because committing the prefixes did not actually make the corpus reproduce, and the claim that it did was checked against the wrong column. A SHA-256 over `chunks.context_prefix` was identical across ingests, so the text was pinned; `chunks.embedding` — the column the vector arm ranks on — still differed in **99 to 141 of 284 rows on every pair of ingests**, because the embedding endpoint does not return bit-identical vectors for identical input. Re-measured while fixing it: embedding the same 284 strings again and comparing at float32 gives 141 of 284 rows different, largest single-component delta 3.2e-03. Small, and not the point — a corpus that changes in half its rows is not a corpus a committed number describes. The vectors are now generated once, keyed by a hash over the **embedding model, the dimension count and the exact embedded string** (so a model change invalidates the cache rather than pairing this text with vectors from another embedding space), stored as base64 float32 — the width pgvector keeps — at 2.2 MiB, and read at fixture-ingest time with the same fail-closed rule as the prefixes. `make embedding-cache` regenerates them, and like `make prefix-cache` it costs money and is nobody's accident.

Two consequences worth stating. A fixture ingest now calls **no provider at all**, so a reviewer with no API key can reproduce the corpus exactly — `make corpus-digest` prints a SHA-256 of the chunk text *and* one of the vector column, and three consecutive `make ingest-fixture` runs print the same pair. And the query side is pinned the same way for the eval: the endpoint also returns slightly different vectors for the same *question* (measured: 2–4 of 38 questions differ between passes, max cosine distance 6.2e-07), so `evals/run_retrieval_eval.py` reads the committed [`evals/query_embeddings.json`](evals/query_embeddings.json) by default and `--query-embeddings live` runs the production path as the control. The application always embeds live; a user's question arrives as text.

Ask *"What obligations apply to providers of high-risk AI systems?"* — cited answer. Ask for *"a compliance report on prohibited practices"* — the approval banner appears; approve or reject with a comment.

## Results

![demo: backend test suite, retrieval ablation, RAGAS, and red-team results](docs/demo.gif)

All numbers below are from live runs against the full corpus (283 chunks, 212 articles), `gpt-4o-mini` as agent and judge. Raw outputs are committed under [`evals/results/`](evals/results), stamped with the run that produced them and with the commit that promoted them — a stamp `backend/tests/test_eval_gate.py` now checks is still reachable from `HEAD`, after one of them turned out to name a commit a rebase had orphaned.

> **These figures predate the current corpus and have not been re-measured.** They were run on a 283-chunk corpus in which annexes and the OJ trailer were swept into whichever article came last; the corpus is now 280 chunks over 212 articles and 13 separately addressable annexes. Nothing below has been rerun against it, and the retrieval numbers in particular will move — annex chunks that used to rank as "Art. 113" now rank as `Annex III`. They are left as measured rather than adjusted, because a number nobody re-ran is not a number.

**Read the RAGAS numbers with that agent-is-also-judge caveat in front of you.** A judge scoring output from its own model family grades leniently on exactly the failure it shares — phrasing that sounds supported. The retrieval ablation below does not have this problem — no judge, and deterministic now that the corpus vectors, the query vectors and the SQL ordering are all pinned (it was not before: the same fixture re-ingested moved hybrid MRR by 0.026, and an unordered `ORDER BY f.score DESC` moved it by a further 0.013 with the data unchanged). That is why the thresholds in [`evals/thresholds.yaml`](evals/thresholds.yaml) give it tight floors and give RAGAS loose ones. Treating these as an independent audit rather than as the author's own instrumentation would be a mistake; a cross-family judge is on the roadmap.

**RAGAS over the 38-question golden set** (0 chat failures, 2026-08-03):

| faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|
| 0.932 | 0.881 | 0.849 | 0.961 |

An earlier single run on 2026-07-12 scored faithfulness **0.964**. Re-running it twice on 2026-08-03 — same corpus, same `ragas` 0.4.3, retrieval byte-identical — gave **0.918** and **0.932**. On 38 questions that gap is one or two flipping, and with one sample in July against two in August there is no way to tell a lucky draw from a small real shift. So the honest figure is a range, not the best draw: **faithfulness 0.918–0.932 over two runs**.

That is the same lesson [`voice-desk-agent`](https://github.com/fctpe/voice-desk-agent) already encodes with `--runs 5` — an LLM-judged number from a single run is not an estimate. It had simply never been applied here. The thresholds in [`evals/thresholds.yaml`](evals/thresholds.yaml) are sized for that spread, which is why the gate passed on both runs rather than crying wolf.

**Retrieval ablation** (hit-rate@6 / MRR / article recall, 38 questions):

| mode | hit@6 | MRR | recall |
|---|---|---|---|
| hybrid (RRF) — production | 1.000 | 0.891 | 0.897 |
| vector only | 1.000 | **0.904** | 0.897 |
| text only (AND semantics) | 0.026 | 0.026 | 0.026 |

**On this question distribution hybrid is not better than vector-only.** It ties on hit@6 and on recall, and loses 0.013 MRR. The entire gap is three questions: hybrid ranks the expected article first on A01 where vector-only ranks it second, and second on A07 and A10 where vector-only ranks it first. Three of 38 is not evidence in either direction — and it is certainly not grounds for bolding hybrid's numbers, which an earlier version of this table did.

Hybrid stays in production because the golden set cannot measure the thing the text arm exists for. Exactly one of the 45 golden questions cites an article number (G06, *"…principles in Article 5 GDPR?"*), so lexical and citation-shaped queries — the shape dense retrieval handles worst — are effectively absent from the distribution these numbers describe. What the run does show is the arm staying silent: it returns no rows at all for 35 of the 38 scored questions, and its one hit is A01, not G06. That is silence on natural-language questions, not precision on citation ones. The precision claim is untested here, so it is not made here. Adding citation-shaped questions and re-running is what would settle it (roadmap below).

An OR-semantics variant was measured and rejected — generic legal vocabulary matches everywhere and the noise leaks through RRF fusion (hybrid MRR 0.891 → 0.772). Negative results are results.

The three-question delta comes straight out of the committed runs:

```bash
jq -n --slurpfile h evals/results/retrieval_hybrid.json --slurpfile v evals/results/retrieval_vector.json \
  '($v[0].questions | map({(.id): .mrr}) | add) as $vm
   | $h[0].questions | map(select(.mrr != $vm[.id]) | {id, hybrid: .mrr, vector: $vm[.id]})'
```

**Red-team suite: 14/14 pass** — five injection classes refused, PII never echoed, out-of-scope frameworks (HIPAA, NIS2, contract drafting, specific legal advice) deflected, two benign controls answered. The first run scored 13/14: the agent answered a HIPAA question from parametric memory; a corpus-boundary rule in the system prompt fixed it, and the case now guards the regression.

The eval suites live in [`evals/`](evals): `make eval` (RAGAS over the golden dataset), `make eval-retrieval` (the three ablation arms above), `make redteam` (adversarial suite).

### What eval-driven iteration caught during development

1. **Fragments break enumerations.** Top-k search returns *parts* of long articles, so "list all prohibited practices" was systematically incomplete. Fix: a `read_article` tool the agent calls on the controlling article before enumerating. Together with the grounding-window fix below this lifted RAGAS faithfulness from 0.886 to 0.964.
2. **The grounding judge needs full text.** It originally audited against 300-char snippets and flagged **correct** enumerations (list items past the cutoff) as unsupported; it now sees full chunk content and judges factual support, not citation-number exactness.
3. **Partial corpus exposed parametric padding.** With only part of the corpus ingested, the model padded enumerations from memory — and the grounding audit caught exactly that. The flag is a feature.
4. **Refusal boundaries are probabilistic.** An adversarial-review pass plus a red-team re-run each surfaced a *different* out-of-scope question (HIPAA once, specific legal advice once) that the model answered instead of deflecting. Both are now firm prompt rules with regression cases, not hopeful phrasing.

## Regulatory accuracy

Docs and corpus reflect the **2026 Digital Omnibus** (adopted June 2026): Annex III high-risk obligations apply from 2 Dec 2027; Art. 50 transparency largely from Aug 2026. Articles and annexes are both ingested; recitals are not (see limitations).

## Design decisions

Short ADRs in [`docs/adr/`](docs/adr): explicit StateGraph over prebuilt agents, hybrid RRF inside Postgres, structural chunking (+ why late chunking was rejected), SQLAlchemy over SQLModel, SSE over WebSockets, regex redaction with a Presidio seam, per-request budgets and the audit hash chain, OTel GenAI spans against conventions that are not stable yet. Architecture walkthrough in [docs/architecture.md](docs/architecture.md), deployment in [docs/deployment.md](docs/deployment.md) — compose, a kustomize base + local overlay under [`deploy/`](deploy), and a Terraform module for the managed Postgres, secrets and DNS. Security posture (RBAC and object-level authz, redaction scope, what the audit chain does not prove, request budgets) is summarised in [SECURITY.md](SECURITY.md) and covered in full in [docs/security.md](docs/security.md).

## Limitations

- **Live EUR-Lex is intermittent.** It sometimes answers the document URLs with `HTTP 202` and an empty body — bot protection or a queued render, not an error status, so `raise_for_status()` waves it through. `fetch_html` rejects any body under 20 KB and `make ingest` fails with a `FetchError` naming the cause, rather than handing an empty string to the parser and blaming the markup. On 2026-08-03 the endpoint returned 202/0 bytes twice and then served the full 1.26 MB document on the next twelve requests. That flakiness is why the corpus is committed: `make ingest-smoke` and `make ingest-fixture` never touch the network. The two sources agree — checked on 2026-08-04, live EUR-Lex and `data/fixtures/` both yield 113 AI Act articles + 13 annexes → 163 chunks, and 99 GDPR articles + 0 annexes → 117 chunks.
- Corpus is articles and annexes, no recitals, and English-only. Both kinds are addressed and cited under their own ref, which is a deliberate reversal of the previous behaviour: annexes carry no article marker, so a paragraph-level parser swept all thirteen of them plus the OJ trailer into the last article, and AI Act "Art. 113" was a 19-chunk blob of mostly Annex III served to users as *Entry into force*. Nothing outside the enacting terms attaches itself to an article now, and content that belongs to no container makes the parser refuse rather than guess. The GDPR has no annexes at all; zero is an ordinary result there, not a failure.
- The router's report-detection is heuristic-first; unusual phrasings can miss the approval gate. The gate is policy, not a security boundary — RBAC is.
- PII redaction is structural (emails, phones, IBANs, cards); free-text names need the documented Presidio swap.
- Grounding audit adds one model call of latency to every answer, and rejected reports end the run — there is no revision loop yet. It also does not check *which* source a claim cites, so a misattributed claim passes: "All AI systems must process personal data in accordance with the GDPR (AI Act, Art. 50(3))" was judged grounded, though Art. 50(3) is about emotion-recognition disclosure. Reading a citation as proof that *that* article says it is the mistake this check does not protect you from.
- Single-tenant by design; per-document ACLs and SSO are out of scope for the kit.
- The audit chain is unkeyed sha256 over columns the database can rewrite, so it detects edits made *around* the application, not an attacker who recomputes the chain — and deleting the newest entries leaves the rest walking clean (ADR 0005).
- The GenAI semantic conventions are still Development status, so the span attribute names can move; the OTel packages are pinned exactly and a test holds the names to the generated constants (ADR 0006). Span cost is whatever you configure — the repo ships no price table.

## Roadmap

- Citation- and lexical-shaped golden questions, so the text arm's precision is measured instead of asserted and the hybrid-vs-vector choice rests on something.
- Re-ingest and a re-run of every suite against the re-cut corpus, so the Results tables describe the articles-plus-annexes corpus instead of the one that predates it. Golden questions whose answer is an annex (high-risk classification) should name it in `expected_articles` once that run exists — the harness already scores `AI Act Annex III` as an expected unit.
- Style-matching RAG over previously **approved** reports, so drafts converge on the reviewing team's voice.
- Revision loop on rejection (reviewer comment feeds a redraft pass).
- German corpus variant (second `tsvector` configuration).

---

AI-assisted scaffolding; architecture, retrieval design, prompts, evals, and the governance model are hand-designed — the ADRs record the reasoning.

[MIT](LICENSE)
