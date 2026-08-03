---
name: grounded-citation-or-refuse
description: Answer from retrieved source text with citations, or refuse — never fill the gap from parametric memory.
---

# Grounded citation or refuse

**Intent:** A regulatory assistant is useful only if every claim traces to the text.
An answer that is right in substance but unsourced is indistinguishable, to the
reader, from one the model invented — so both are failures.

**Evidence:** The retrieval events for the turn, the source text they returned, and
the claims in the final answer.

**Decision:** Every substantive claim in the answer is supported by text present in
the retrieved sources for that turn.

**Execution:** Cite the source alongside the claim. Where retrieval returned nothing
relevant, say so and stop.

**Recovery:** When sources are thin, narrow the answer to what they support rather
than widening it from memory. Partial answers are acceptable; unsourced ones are not.

**Failure modes:** Answering from parametric memory when retrieval returned nothing;
citing a source that does not contain the claim; hedging into a plausible-sounding
generality to avoid saying "the corpus does not cover this".

## Verification must fail closed

**Intent:** The grounding check is the last line of defence, so the way it fails
decides whether the whole guarantee holds.

**Decision:** An unparseable or errored verification verdict is treated as
*ungrounded*.

**Failure modes:** Reporting an errored check as "verified"; dropping the check
silently when the verifier throws.
