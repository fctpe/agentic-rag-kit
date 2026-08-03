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
