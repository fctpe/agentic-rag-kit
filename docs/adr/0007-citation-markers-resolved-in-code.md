# ADR 0007: The inline citation marker is resolved in code, not requested from the model

**Status:** accepted · 2026-08-04

## Context

The `[n]` marker is the only thing joining a sentence to the source card it came from. The UI links exactly one shape (`/\[(\d{1,3})\]/`), so anything else the model writes renders as literal text and the source it names is never linked. None of the four RAGAS metrics can see that: faithfulness judges whether a claim is supported, not whether the answer said where from.

Two separate defects sat behind this.

**The bracket shape.** The model does not usually cite in prose. It welds the reference *into* the bracket. Across every committed eval artifact, 126 brackets take a merged shape — `[4, Art. 4(5)]` (114), `[2(a)]` (10), `[7, 8]` (2) — and every citable answer that carried no linkable marker contained at least one of them. None of the 126 links to anything.

**The number space.** `CitationCollector` was constructed per HTTP request, so turn 2 on a thread re-issued `[1]`, while the model still had turn 1's answer — and turn 1's `[1]` — in its message history. A marker reused from an earlier turn rendered as a *working* button pointing at a different regulation. It failed silently and reported `grounded: true`, because `GROUNDING_PROMPT` explicitly instructs the judge to ignore numbering.

## What was rejected

**Another prompt instruction.** Already tried and measured. A single sentence telling the model the bracket holds only the number moved unmarked answers from 2 to 9 and 10 out of 37 over two full runs; it quoted the wrong shape verbatim as a counter-example and the model copied it. Reverted in `bb4e582`. Deleting the "Answer style" paragraph instead does fix the shape (37/37 over two full runs) — but a prompt cannot carry a guarantee, there is no test that fails without it, and it costs ~35% of the spelled-out article references the paragraph exists to produce plus a paid RAGAS re-run. It is evidence about the cause, not a mechanism.

**Widening the frontend regex.** A pattern loose enough to match `[4, Art. 4(5)]` still cannot decide anything about `[7, 8]`: two sources, one bracket, one possible `data-cite`. It converts a visible failure into a wrong link.

**Structured output, or a second model call that resolves prose references to indices.** Both rebuild the generation contract or add a judged step to the critical path of every answer, to fix a formatting defect. The measured rate on HEAD is 1 answer in 38.

## Decision

A deterministic node, `resolve_citations`, on the single path from "the model has answered" to the user. No model call, no measurable latency.

It rewrites the three merged shapes:

| written | shipped |
|---|---|
| `[n, Art. 4(5)]` | `(Art. 4(5)) [n]` |
| `[n(a)]` | `[n]` |
| `[n, m]` | `[n][m]` |

and fails closed on everything else, where *closed* means **not linked**:

* an index that is not in the retrieved source list is **removed**, never renumbered and never guessed at;
* a bracket whose spelled-out reference contradicts the source it numbers — `[5, Art. 4(7)]` where source 5 is GDPR Art. 28 — is **left exactly as written**, so it stays visible as text rather than becoming a confidently wrong link;
* an unrecognised shape is left alone.

All three are reported in a new `citation_issues` state channel, so a bracket that does not link is never silently swallowed. Answers that genuinely cite nothing — refusals, out-of-scope declines — contain no brackets and pass through untouched; nothing here can invent a citation.

Source ids are also now stable for a **thread**: `CitationCollector` is seeded from the checkpointed `retrieved_sources`, so an id, once issued, means that chunk forever.

## Why the node, and not `verify()`

Three different places read the answer: `approval_gate` puts `messages[-1]` in the interrupt payload, `verify()` judges `messages[-1]`, and `/chat` persists the last `AIMessage`. Normalising in any one of them would have shown the reviewer text A and shipped the user text B — the failure the approval gate exists to prevent. The node writes the resolved text back into the message with the same id, so `add_messages` replaces it and all three reads see one string.

## Consequences

**`citation_issues` is a separate channel from `grounding_issues`, deliberately.** `docs/deployment.md` tells operators to alert on `ragkit.grounded=false`; a bracket naming the wrong article is a citation-resolution defect, not an unsupported claim. Folding them together would make that alert fire on formatting.

**SSE.** No event was added and no ordering changed. `citation_issues` rides on the two events that carry the finished text — `done` and `approval_required` — because it describes that text. On the report path it is read from graph state rather than from the stream: `resolve_citations` ran in the request *before* the resume, so a stream-sourced field would ship empty on every report.

**qa streams raw tokens, then corrects.** The node runs after the last token is sent, and `useChatStream` already replaces the bubble with `done.content`. A merged bracket is visible as raw text for the window between the last token and `done`. Buffering qa tokens to hide that would delete the streaming guarantee at the top of `app/api/chat.py` and the "Checking sources… / Unverified" window `ChatMessage.tsx` exists to provide. The flicker is the cheaper trade.

**The number space grows monotonically across a thread.** Bounded in practice: chunks are ~700 tokens and `sources_to_excerpts` already truncates at 60k chars for the verifier.

**Sub-provision granularity is not addressed.** 114 of the 126 merged brackets are the model reaching for "Art. 4(5)" when the citation scheme's finest unit is "Art. 4" — the panel can show two indistinguishable `GDPR · Art. 4` cards. Fixing that means a re-chunk and re-ingest, which moves the corpus digest and every retrieval number. It is not the cause of the merged bracket, and it is out of scope here.

## Where it is held to this

Offline and free, on every push:

* `backend/tests/test_citation_markers.py` — each shape, each fail-closed branch, and a **replay of every answer in the committed eval artifacts** against its real source list. `pytest tests/test_citation_markers.py -s -k merged` prints the census — 120 of 126 merged brackets become links, and the 6 refusals are all article-vs-source disagreements. The number is printed, not pinned to a literal: the denominator is whatever `ragas_*.json` is committed, and a promoted run would make a hard-coded count stale.
* `backend/tests/test_citation_pipeline.py` — the resolved text is the checkpointed message, the approval payload, and the persisted answer; and turn 2 does not re-issue turn 1's ids.
* `frontend/tests/citation-markers.test.mjs` — the client half of the contract, which had **no test at all** before this. `CITE_PATTERN` was the only definition of a source marker anywhere in the repo and it was asserted only by a 38-question eval that costs money to run.

Every one carries a negative control, because a test written to confirm the design proves nothing.
