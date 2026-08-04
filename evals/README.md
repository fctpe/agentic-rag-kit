# Evals

Three suites over the golden set in `golden_questions.yaml` (45 hand-written
Q/A pairs: ~15 AI Act, ~15 GDPR, 8 cross-regulation, 7 out-of-scope refusal
probes) plus the adversarial cases in `redteam/cases.yaml`.

All commands run from `backend/` so the `app` package and the `evals`
dependency group (ragas, deepeval, pyyaml) resolve:

```bash
cd backend
uv run --group evals python ../evals/<script>.py [flags]
```

Results land in `evals/results/` as JSON; each script also prints a markdown
table.

## 1. Retrieval eval — `run_retrieval_eval.py`

Retrieval-only metrics, no LLM judges. For every non-out-of-scope golden
question it retrieves top-k chunks and matches them against
`expected_articles` (a hit = retrieved chunk with the same regulation and
unit). A label may name either shape the corpus stores — `AI Act Art. 6` or
`AI Act Annex III` — since the high-risk list Art. 6(2) classifies against
lives in the annex. The two never cross-match: an expected `Art. III` is not
satisfied by `Annex III`.

```bash
uv run --group evals python ../evals/run_retrieval_eval.py                  # hybrid, k=6
uv run --group evals python ../evals/run_retrieval_eval.py --mode text_only # FTS arm only
uv run --group evals python ../evals/run_retrieval_eval.py --mode vector_only
uv run --group evals python ../evals/run_retrieval_eval.py --regulation-filter off
uv run --group evals python ../evals/run_retrieval_eval.py --smoke          # first 5 questions
uv run --group evals python ../evals/run_retrieval_eval.py --query-embeddings live
```

Flags: `--k` (default 6), `--mode hybrid|vector_only|text_only`,
`--regulation-filter on|off`, `--query-embeddings committed|live`,
`--json <path>`, `--smoke`, `--questions <ids>`.

Every run writes `results/retrieval_<mode>[_nofilter]_<timestamp>.json`. It
never writes `results/retrieval_<mode>.json` — that file is the one the README
table is read from and only `promote.py` moves it (see *Promotion* below). It
used to be written in place on every invocation, which is how commit 88131af
changed the committed hybrid MRR from 0.8912 to 0.875 as a side effect of a run
nobody had decided to publish, while README.md went on quoting the older draw.

**This suite needs no provider key and is reproducible run to run** — both
statements became true on 2026-08-04 and neither was before. The corpus text,
its context prefixes and its embedding vectors are all committed under
`data/fixtures/`, and the golden questions' vectors under
`query_embeddings.json`, because the embedding endpoint does not return
bit-identical vectors for identical input: 99–141 of 284 chunk rows differed on
every pair of ingests, and 2–4 of 38 questions differ between passes. Every
`ORDER BY` in the retrieval SQL now carries a `(regulation, article_ref, idx)`
tiebreak for the same reason — RRF ties are arithmetic, and an untied
`ORDER BY f.score DESC` made question A07's rank-1 result a function of the
`LIMIT`. `--query-embeddings live` embeds each question through the production
path instead; it is the control that pinning did not change what is measured,
and the mode is recorded in every results file. The application always embeds
live, because a user's question arrives as text.

Metrics:

- **hit-rate@k** — share of questions where at least one expected article
  appears in the top-k chunks.
- **MRR** — mean reciprocal rank of the first chunk matching any expected
  article (1.0 = always rank 1; 0 = never found).
- **mean article recall** — average fraction of a question's expected
  articles found in the top-k (matters for cross-regulation questions that
  expect articles from both laws).

The `--regulation-filter on` mode passes the regulation implied by the
expected articles to the search (cross-regulation questions always run
unfiltered); comparing on vs. off shows what metadata filtering buys.
`vector_only` / `text_only` inline the corresponding single arm of the
`HYBRID_SQL` CTE from `app/retrieval/hybrid.py` — keep them in sync if the
schema changes.

Cost: `text_only` is completely free (pure Postgres). `hybrid` and
`vector_only` embed each query once (`text-embedding-3-small`, ~fractions of
a cent for the whole set). Deterministic given a fixed corpus and embedding
model.

**Corpus caveat:** with the smoke ingest (`make ingest-smoke` — Articles 1–10
of each regulation, no annexes) questions targeting later articles (e.g. AI
Act Art. 50, GDPR Art. 33) cannot be hit — low absolute scores are expected.
Compare modes and filters against each other, not against 1.0. Run the full
corpus (`make ingest-fixture` — 212 articles + 13 annexes → 284 chunks, AI Act
167 and GDPR 117), which the same golden set exercises end to end. Every
committed result under `results/` was measured on that corpus and promoted from
it; `make corpus-digest-check` confirms an ingest reproduces it before the
numbers mean anything.

## 2. RAGAS eval — `run_evals.py`

End-to-end quality: drives the real chat API (login as
`analyst@example.com` / `demo1234`, `POST /chat`, parse the SSE stream) for
each non-out-of-scope question, fetches judge contexts via `hybrid_search`
for the same query (full chunk texts, not the 300-char citation snippets),
then scores with RAGAS 0.4.x.

```bash
# prerequisites: docker compose up postgres, backend running on :8000,
# users seeded (uv run python -m app.seed), corpus ingested, OPENAI_API_KEY set
uv run --group evals python ../evals/run_evals.py --smoke        # first 3 questions
uv run --group evals python ../evals/run_evals.py                # all 38 scorable questions
uv run --group evals python ../evals/run_evals.py --questions A01 G03 X07
```

Flags: `--smoke`, `--questions <ids>`, `--concurrency` (default 3), `--k`
(contexts per question), `--api-base` (default `http://localhost:8000`).
Env: `RAGAS_JUDGE_MODEL` (default `gpt-4o-mini`), `EVAL_EMAIL`,
`EVAL_PASSWORD`, `EVAL_API_BASE`.

Metrics (all 0–1, higher is better):

- **faithfulness** — are the answer's claims supported by the retrieved
  contexts? (hallucination detector)
- **answer_relevancy** — does the answer actually address the question?
  (detects evasive/off-target answers)
- **context_precision** — are the relevant chunks ranked above the
  irrelevant ones, judged against the reference answer? (retrieval ranking)
- **context_recall** — is everything the reference answer needs present in
  the retrieved contexts? (retrieval coverage)

Also recorded, and gated at zero: **answers carrying no inline `[n]` marker**.
Not a judged metric — a format check the four metrics above are blind to. An
answer that cites in prose ("as set out in AI Act, Art. 50(1)") is just as
faithful and just as relevant, and arrives in the UI with every source
unlinked, because `[n]` is what the citation panel binds to. A run recorded
before this check existed has no such field, and the gate fails it rather than
assume the answers were clean.

RAGAS 0.4.x API notes (verified against the installed 0.4.3): metrics come
from `ragas.metrics.collections` (`Faithfulness`, `AnswerRelevancy`,
`ContextPrecision`, `ContextRecall`), the judge is built with
`ragas.llms.base.llm_factory(model, client=AsyncOpenAI())` and
`ragas.embeddings.base.embedding_factory("openai", model=..., client=...,
interface="modern")`, and each metric is scored per-sample via
`await metric.ascore(...)` returning a `MetricResult` with `.value`.
The lowercase `ragas.metrics.faithfulness`-style instances are deprecated in
0.4.x. Heads-up: ragas 0.4.3 hard-imports the removed
`langchain_community.chat_models.vertexai` module; the script registers a
stub module before `import ragas` (see `_shim_vertexai`).

Cost: every question makes one real agent run (router + tool calls +
grounding check) plus ~4 judge calls. Ballpark with `gpt-4o-mini` judges:
roughly $0.05–0.15 for the full 38-question set, a few cents for `--smoke`.
LLM-judged scores are not perfectly reproducible run to run.

## 3. Red team — `run_redteam.py` + `redteam/cases.yaml`

14 adversarial cases against the live API: 5 prompt injections (instruction
override, role hijack, `</source>` tag smuggling, system-prompt probe,
exfiltration ask — one per heuristic in `app/security/injection.py`), 2 PII
probes (synthetic email/phone/IBAN/card shaped to match
`app/security/redaction.py`), 5 out-of-scope legal-advice asks, and 2 benign
controls that must NOT be refused.

```bash
uv run --group evals python ../evals/run_redteam.py --smoke   # 4 representative cases
uv run --group evals python ../evals/run_redteam.py           # all 14, exit 1 on any fail
```

Assertions:

- **refused** — the answer contains refusal/deflection language. Injection
  cases expect the deterministic input-guard refusal; out-of-scope cases use
  per-case marker lists (heuristic substring matching — eyeball the
  `answer_preview` in the JSON when a deflection case fails).
- **not_echoed** — none of the planted PII strings appear in the answer
  (the guard redacts them before the model sees the message).
- **answered** — control is not refused, has substance, and mentions at
  least one expected term.

If the router classifies a case as a report request, the run stops at the
human-approval interrupt; the script then judges the `approval_required`
draft and records `approval_interrupted: true`.

Cost: guard-refused injections short-circuit before the model (free); the
remaining ~9 cases are one agent run each — well under $0.05 total with
`gpt-4o-mini`.

## Promotion — `promote.py`

Every suite writes timestamped runs. The files README.md quotes —
`results/ragas.json`, `results/redteam.json`, `results/retrieval_<mode>.json` —
are written by nothing but this script, on four rules:

1. **Only the newest run.** If the latest run is worse, that is the number.
2. **Only a run that passes the gate**, including its declared baseline: a
   retrieval run that does not reproduce `thresholds.yaml` to the last digit is
   refused. Publishing a genuinely new number therefore means declaring it in
   `thresholds.yaml` first and then promoting the run that measured it.
3. **Only from a clean worktree**, ignoring other files under `results/` — those
   are outputs of other runs, not inputs to this one, and blocking on them would
   make promoting the three retrieval modes in sequence impossible.
4. **Stamped** with the source filename, the promotion time, and the full 40-char
   commit sha. `backend/tests/test_eval_gate.py` fails if a stamp stops being
   reachable from `HEAD`, which is what a rebase once did to one of them.

```bash
uv run --group evals python ../evals/promote.py retrieval_hybrid
uv run --group evals python ../evals/promote.py retrieval_vector_only
uv run --group evals python ../evals/promote.py retrieval_text_only
uv run --group evals python ../evals/promote.py ragas
uv run --group evals python ../evals/promote.py redteam
```

Subset runs (`--smoke`, `--questions`), unfiltered ablations
(`--regulation-filter off`) and runs measured at a different `--k` are not
promotable — the first two are refused by the gate, the last two cannot even be
candidates.

## Files

```
evals/
├── golden_questions.yaml    45 golden Q/A pairs (see header comment for schema)
├── run_retrieval_eval.py    hit-rate@k / MRR, hybrid vs single-arm ablations
├── run_evals.py             RAGAS 0.4.x over the live chat API
├── run_redteam.py           adversarial suite, exit 1 on failure
├── redteam/cases.yaml       14 red-team cases
├── promote.py               the only writer of the committed results
├── gate.py                  thresholds, floors, ceilings, baseline drift
└── results/                 timestamped runs (retrieval_<mode>_<ts>.json,
                             ragas_<ts>.json, redteam_<ts>.json) plus the
                             promoted retrieval_<mode>.json / ragas.json /
                             redteam.json the README cites
```
