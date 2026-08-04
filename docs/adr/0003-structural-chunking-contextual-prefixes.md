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
- Recitals are not ingested in v1. Annexes were not either — see the addendum below.

## Addendum, 2026-08-04: the unit is the article *or the annex*, and the annex is not optional

"Chunk on article boundaries" quietly assumed every citable thing is an article. It is not. AI Act Art. 6(2) says *"AI systems referred to in Annex III shall be considered high-risk"* — the classification rule is in the article, the list of use cases it classifies is in the annex, and nowhere else. A corpus of articles alone cannot answer "is my system high-risk?", which is the most common practical question this assistant gets.

Annexes carry no `ti-art` marker, so the original paragraph-level parser had appended them (and the section headings, and the OJ trailer) to whichever article came last: AI Act "Art. 113" was a 523-paragraph, ~50,000-character blob of mostly Annex III, cited to users as *Entry into force and application*. Excluding annexes fixed the mislabelling and lost the content. Neither state is acceptable, and the reason both were reachable is that the design had one kind of unit where the document has two.

The decision, therefore: **an ingested unit is any citable subdivision — an `Article` or an `Annex`** (`app/ingestion/eurlex.Unit`). Both are parsed from their own EUR-Lex container id (`art_113`, `anx_III`), both go through the same chunker, and both land in the same table. Each kind declares the two facts that distinguish it — the printed ref (`Art. 6`, `Annex III`) and the EUR-Lex anchor prefix (`art`, `anx`) — on the class, so the parser, the chunker and `app/retrieval/citations.py` cannot drift into deep-linking `Annex III` at `#art_III`, which resolves to nothing.

- **Annexes are chunked, never stored whole.** Annex III is 66 paragraphs. As one unit it would outrank and crowd out the articles that cite it, which is the pre-fix Art. 113 failure with a correct label on it.
- `chunks.article_ref` keeps its column name and now holds either shape. Renaming it would invalidate every persisted citation payload and eval baseline to buy a tidier identifier.
- Zero annexes is a valid corpus (the GDPR has none), so nothing special-cases their absence — but a *fixture* with no `annexes` key fails closed, because that means the file predates annex ingestion rather than that the regulation has none.
- The re-cut corpus is 212 articles + 13 annexes → 280 chunks. Every eval figure committed before this change was measured on a different corpus and needs re-running; see the note in the README Results section.

## Addendum, 2026-08-04 (later the same day): the unit of body text is the cell, not the paragraph

The annex work above landed a corpus in which **Annex I was 238 characters** — the numbers 1 to 12 and nothing else. The list of Union harmonisation legislation that Art. 6(1) points at for high-risk classification had been reduced to its own numbering.

EUR-Lex renders numbered lists as tables, one cell for the marker and one for the text. Most rows wrap both in `<p class="oj-normal">`. One row shape does not: `<td></td><td><p>1.</p></td><td><span>Directive 2006/42/EC …</span></td>`. The parser extracted with `container.find_all("p")` — a whitelist of exactly one tag — so it took every marker and dropped every directive.

What makes this the more instructive failure of the two is that **nothing was structurally wrong afterwards**. The unit existed, carried its heading, had a non-empty `paragraphs`, produced a chunk, and deep-linked correctly. Every guard in the module counted units; none compared the text a container holds against the text taken out of it. Measured across all 225 units, four were affected: Annex I (capture ratio 0.034), Annex VII (0.452), Annex XI (0.755) and Art. 108 (0.998, four stray semicolons). The other 221 — including all 99 GDPR articles and Annex III — were byte-identical.

Two decisions follow.

**Extraction is structural, not tag-named.** A block element ends a paragraph; a paragraph's text is whatever is not a nested block (`_BLOCK_ELEMENTS` in `app/ingestion/eurlex.py`). If EUR-Lex swaps `<span>` for `<em>`, `<font>`, or a bare text node, the text is still captured, because none of them is named anywhere. Only a genuinely new *block* element would need the set extended, and that failure is loud — two paragraphs run together — rather than silent.

**And a coverage check, because the rule above will eventually be wrong too.** `_check_capture` compares the word characters a unit carries against the word characters its container holds, and raises `ParseError` below `MIN_CAPTURE_RATIO = 0.95`. The floor is not a round number: the new extractor scores 1.000 on every one of the 225 units because it partitions the container, and the rule it replaces scored 0.034 / 0.452 / 0.755 on the three annexes. Any floor in (0.755, 1.0] separates them, so 0.95 is chosen for headroom against a future extractor dropping something incidental, not for sensitivity. The tests pin both directions: the `<span>` shape parses, and restoring the `<p>`-only rule raises.

**Corpus: 212 articles + 13 annexes → 284 chunks** (AI Act 167, GDPR 117). Superseding the 280 recorded above, which was the count with Annex I empty.

### Ingestion moved to Cellar; Formex considered and deferred

The EUR-Lex web endpoint answers **HTTP 202 with an empty body** under bot protection — observed for over an hour continuously, across 30 retries, on the day this was written. `FetchError` names that case precisely so it is never mistaken for a parse failure, but a fresh ingest was still a coin flip.

Documents are now fetched from **Cellar** (`publications.europa.eu/resource/celex/<CELEX>`), the Publications Office repository EUR-Lex itself reads from, via content negotiation. It served both regulations without complaint through exactly the window EUR-Lex was refusing. Parsing both sources produces **byte-identical units across all 225** — same publisher, same content, through the interface intended for programs.

`REGULATIONS` therefore carries two URLs, and the split is load-bearing: `fetch_url` is Cellar, `url` stays the EUR-Lex page, because a citation must link where a reader can check the claim. Fixtures record both.

**Formex XML** (`Accept: application/xml;notice=branch`) is available from the same endpoint and is the better long-term target: explicit `ARTICLE` and `ANNEX` elements would make this entire class of loss impossible by construction rather than caught by a coverage check afterwards. It is deferred, not rejected — it is a parser rewrite against a different schema, and the coverage check closes the immediate hole with a guard that also covers whatever the *next* markup surprise turns out to be.

Fixtures are regenerated with `make refresh-fixtures`. Until this ADR, the committed corpus could only be produced by a throwaway script — a reviewer could read the parser and read the fixture with no way to check that one had produced the other.

## Addendum, 2026-08-04 (third): the contextual prefixes are part of the corpus, so they are committed

"An optional LLM pass adds one situating sentence per chunk at ingest" was written as if it were a step in a build. It is not: it is a **sample from a model**. `temperature=0` narrows the distribution, but there is no seed and no provider guarantees one, so the pass returns different sentences on different days. Those sentences are embedded with the chunk text, which means the fixture did not determine the corpus — the ingest did.

Measured on the committed fixture, two ingests apart:

| | hybrid MRR | vector-only MRR | recall |
|---|---|---|---|
| ingest A | 0.888 | 0.901 | 0.910 |
| ingest B | 0.914 | 0.928 | 0.897 |

Same fixture, same parser, same chunker, same SQL. Two things break at once. `evals/thresholds.yaml` reasons that "retrieval is deterministic… floors sit just under baseline, because any real movement is a real change" — a floor 0.003 under its baseline is not a regression guard against a 0.026 spread, it is a coin flip that reds the build. And the previous addendum's own justification for committing the fixtures — that a reviewer should be able to check the corpus rather than take it on trust — did not actually hold, because the reviewer could reproduce the *text* and not the *chunks*.

**Decision: generate the sentences once, commit them next to the text, and read them at fixture ingest.** `data/fixtures/context_prefixes.json`, written by `app/ingestion/refresh_prefixes.py`, read by `app/ingestion/prefix_cache.py`.

- **Keyed by the model call, not by position.** The key is a SHA-256 over the rendered prompt plus the untruncated chunk content. A `(ref, idx)` key would let an edited fixture pair a chunk with a sentence written about its previous text — stale, embeds cleanly, invisible in every count this repo keeps. Content is hashed past the prompt's 4,000-character truncation so two long chunks that diverge late cannot collide, and `CONTEXT_PROMPT` is in the key so rewording it invalidates every entry instead of pairing new prompt semantics with old answers.
- **A miss refuses the ingest.** It names the chunks and names `make prefix-cache`. It does not generate the missing sentences — that is precisely the non-determinism being removed — and it does not fall back to the deterministic prefix, which would embed a corpus that is part contextual and part not: an unlabelled third corpus, and worse than one that will not load.
- **`--source network` still generates.** A fresh EUR-Lex parse has no cache by definition; its text may differ from the fixture, so its keys would miss anyway. That arm is irreproducible on purpose, and a corpus ingested through it is not what a committed eval number describes.
- **Regeneration is its own target and costs money.** One model call per chunk, 284 of them. `make refresh-fixtures` is free and stays free; folding the prefixes into it would bill anyone who re-parses. The link between them is a test, not a dependency: `backend/tests/test_prefix_cache.py` fails offline the moment the committed cache stops covering the committed fixture, which is the only reliable signal that the two have drifted.

Verified the way the property is stated: `make ingest-fixture` twice, then a SHA-256 over every `chunks.context_prefix` in the database ordered by `(regulation, article_ref, idx)` — identical both times. The negative control is the same measurement with the old behaviour restored, which produces a different digest on every run.

> **Correction, later the same day.** The sentence above is true and the claim it was taken to support — "two ingests of the committed corpus produce byte-identical chunks" — was false. What was verified is the *text* digest. `chunks.embedding` was never hashed, and it differed on every pair of ingests. The next addendum is what that cost and what it took to make the claim true; the paragraph is left standing because the shape of the mistake is the useful part. **A guarantee has to be checked on the column that carries it.**

## Addendum, 2026-08-04 (fourth): the vectors are part of the corpus too, and so is the ORDER BY

Committing the prefixes fixed the input the previous addendum was looking at. It did not make the corpus reproduce, because retrieval does not rank on `context_prefix` — it ranks on `chunks.embedding`, and nothing was watching that column. Measured across five ingests of the byte-identical committed fixture (text digest `3cecee3e…` all five times, so the vectors were the only variable), over all ten pairs:

| | |
|---|---|
| rows differing bit-exactly | **99–141 of 284** (35–50%) on every pair |
| components differing, in a row that differs | 933–1511 of 1536 (median 1248) — the vector is redrawn, not rounded |
| largest single cosine deviation | 1.070e-04 |
| largest single-component delta | 3.265e-03 |
| perturbation to the ranking quantity, \|Δd(q,c)\| over 38 × 284 | mean 2.92e-05, p95 1.37e-04, max 1.52e-03 |

For scale, the two closest *distinct* chunks in this corpus sit 3.73e-02 apart, so the drift is ~350× smaller than the smallest real gap. It is small. It is also not zero, and `evals/thresholds.yaml` sets retrieval floors 0.003 under baseline on the reasoning that the number cannot move by itself.

**Decision: commit the vectors the same way, keyed to include the model.** `data/fixtures/chunk_embeddings.json`, written by `app/ingestion/refresh_embeddings.py`, read by `app/ingestion/chunk_embeddings.py` over the format in `app/embedding_cache.py`.

- **Keyed by the whole embedding call.** SHA-256 over `model \x1f dimensions \x1f text`, where the text is the exact prefix-plus-content string that was sent. The model belongs in the key because `text-embedding-3-small` and `-3-large` embed the same string into different geometries: a text-only key would *hit* on every chunk after a model swap and hand the ingest vectors that mean nothing in the configured space — an ingest that succeeds, a corpus that ranks like noise, no error anywhere. The dimension count is in the key for the same reason (these models support reduced widths). `embedded_text()` is defined once and called by both the cache writer and the ingest, because two definitions of that string is how a cache silently stops covering a corpus while every count still reads 284.
- **base64 float32, not a JSON array of decimals and not a side-car blob.** float32 because `chunks.embedding` is pgvector's `vector`, i.e. float4 — committing float64 would commit 8 bytes of which Postgres keeps 4, and the round trip would stop being exact. base64 because the decimal-array alternative is ~6 MB for the same data and no more readable (nobody reads 436,224 numbers), while a binary side-car diffs as one opaque object; one base64 string per entry keeps `git diff` naming the chunks whose vectors moved. 2.2 MiB, and each file states its own size.
- **A miss refuses the ingest**, names the chunks and names `make embedding-cache`. No fallback to the endpoint — that is the non-determinism being removed — and, unlike the prefix case, no partially-cached corpus is even expressible.
- **`--no-contextual` is refused on `--source fixture`.** It embeds a different string per chunk, so it is a different corpus, and no eval number here was measured on it. `make ingest-smoke` is now `--source fixture --max-units 10`, which exercises the real path.
- **A fixture ingest calls no provider at all.** That is the second win and arguably the larger one: the committed corpus is now reproducible by a reviewer with no API key, and CI can verify it on every push for free.

**The query side moves too, and is pinned for the eval only.** Three independent passes over the 38 in-scope golden questions differ in 2, 3 and 4 of them (max cosine distance between two embeddings of one question 6.21e-07; embedding the same string batched vs singly also differs, so batch shape is an input). In |Δd(q,c)| that is mean 1.31e-06 against the corpus side's 2.92e-05 — 22× smaller — and swapping the query pass while holding the corpus fixed moved no metric and reordered no question's top 6. It is nonetheless an input the floors assume is fixed, so `evals/query_embeddings.json` pins it and `run_retrieval_eval.py` reads it by default; `--query-embeddings live` runs the production path as the control. **The application still embeds live**, because a user's question arrives as text — the guarantee is about the eval, and the mode is recorded in every results file so it cannot be read as anything wider.

**And the SQL had to order totally.** Pinning both inputs still left one committed number moving, and it was not embeddings. RRF scores are sums of `1/(k + rank)` over small integers, so exact ties are arithmetic rather than accident. Question A07 produced two chunks with bit-identical fused scores (AI Act Art. 4 at vector rank 1, Art. 66 at text rank 1 — `0x1.0c9714fbcda3bp-6` both), and `ORDER BY f.score DESC` left the winner to the plan: `final_k=2` returned Art. 4, `final_k=3` Art. 66, `final_k=6` Art. 66, `final_k=8` Art. 4. Same rows, same scores, only the LIMIT varied. That is a production defect before it is an eval one — the rank-1 citation a user sees depends on how many results were asked for — and it was worth 0.0132 of hybrid MRR across ingests. Every `ORDER BY` in `HYBRID_SQL` and in the two ablation variants now ends in `(regulation, article_ref, idx) COLLATE "C"`: unique over the corpus, derived from the fixture (so unlike `chunks.id`, a fresh uuid4 per ingest, it is the same on every ingest), byte-collated so it does not depend on the machine's ICU version, and free — the window functions already force a full sort, confirmed with EXPLAIN. The text arm's `LIMIT :arm_size` also had no `ORDER BY` at all, so which twenty matches reached fusion was plan-dependent; it has one now.

Verified the way the property is now stated, which is the part the previous addendum got wrong: `make ingest-fixture` **three** times with no `OPENAI_API_KEY` in the environment, then `make corpus-digest`, which hashes the text *and* `chunks.embedding` — `text 9d3a1677…`, `vector f79fb682…`, identical all three times. Two negative controls. Perturbing one component of one row by 1e-3 leaves the text digest unchanged and changes the vector digest, so the text-only check that shipped the false claim would have passed this corrupted corpus. And embedding the same 284 strings through the API again differs from the committed vectors in 141 of 284 rows (largest component delta 3.235e-03), so the cache is removing real variance rather than recording what the endpoint would have returned anyway.
