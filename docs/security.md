# Security & governance

## OWASP Top 10 for LLM Applications (2025) — what this kit demonstrably mitigates

| Risk | Mitigation in this codebase |
|---|---|
| LLM01 Prompt injection | Input heuristics (`app/security/injection.py`) refuse and audit-log override/hijack/probe/smuggling patterns; retrieved chunks are delimited as `<source>` data blocks the system prompt declares untrusted; report output passes a human gate. The red-team suite (`evals/run_redteam.py`) asserts all three layers. |
| LLM02 Sensitive information disclosure | Structural PII redaction (`app/security/redaction.py`) runs at the **API boundary**, in `/chat`, before the message is handed to anything that persists it — the message table, the graph, the model, or a trace. It runs there and not in `guard_input` because LangGraph checkpoints the input super-step before the first node executes (ADR 0004). `guard_input` still redacts, as a boundary for any caller that drives the graph directly, and unions its findings with the labels the route reports. `backend/tests/test_redaction_boundary.py` walks the whole checkpoint history for a canary and carries the negative control that proves the node-level shape leaked. Redaction events land in the audit log without the raw values, and spans and log lines carry counts, ids, model names and verdicts only — never prompts, queries or retrieved text (ADR 0006). Failures are held to the same rule: `error_fields` puts the exception *type* and its frames on the span and in the log and drops `str(err)`, because a SQLAlchemy `DBAPIError` message embeds the bound parameters and the query is one of them. |
| LLM05 Improper output handling | Assistant output is rendered as markdown text, never executed or interpolated; citations are built server-side from retrieved rows, never parsed from model output. |
| LLM09 Misinformation | Grounded citations with article-level deep links plus a post-answer grounding audit. What that audit verifies, exactly: whether each substantive claim is supported by the text of **any** retrieved source, and it reports the claims no source supports. What it does not verify is attribution — `GROUNDING_PROMPT` explicitly tells the judge not to report citation-numbering mismatches, so a claim that cites the wrong article passes as long as some retrieved source supports the claim itself. Observed: *"All AI systems must process personal data in accordance with the GDPR (AI Act, Art. 50(3))"* was judged grounded; Art. 50(3) is about emotion-recognition disclosure. The tradeoff is deliberate — judging citation numbers made the check flag **correct** enumerations as unsupported, which is a worse failure for a governance signal an operator alerts on — but it means `grounded=true` says "nothing here is invented", not "every reference is right". `grounded` fails closed: a run that verified nothing — refused input, an answer built on zero retrieved sources, a stop at the token budget — reports `grounded=false` with the reason in `grounding_issues`, so it is never the one case the operator alert cannot see. |
| LLM10 Unbounded consumption | Input length caps, tool-round caps (`MAX_TOOL_ROUNDS`), a per-request token budget read from `usage_metadata` (`MAX_TOTAL_TOKENS`), a graph `recursion_limit`, per-call LLM timeouts, and retrieval `final_k`. Over budget, the run ends with a message saying so — it never returns a truncated answer as if it were complete (ADR 0005). |

## RBAC

Three roles (`viewer` → chat, `analyst` → search/approvals/resume, `admin` → audit). JWT (HS256, 12h TTL) with scrypt password hashing. Every privileged route declares its minimum role via `require_role`.

## Audit trail (EU AI Act Art. 12 framing)

`audit_log` is append-only — the application exposes no update or delete path — and tamper-evident: each entry stores the previous entry's hash plus its own, so a row edited or deleted *around* the application breaks the chain. `GET /audit/verify` (admin) walks it in the order the database assigned (`seq`, a bigserial) and reports the first break with the entry id and whether the row failed its own hash or its link. Logged: logins (incl. failures), queries, answers (with grounding verdict + redaction/injection flags), approval requests and decisions with the deciding user. The human approval gate for report-type output maps to the Art. 14 human-oversight requirement. Chain design: ADR 0005.

**What the chain proves, and what it does not.** It is evidence, not proof, and the difference is worth stating precisely because "tamper-evident" gets overclaimed:

| Change made directly in the database | Detected? |
|---|---|
| A row's actor, action, resource, detail or timestamp edited | Yes — the row fails its own `entry_hash`. |
| A row deleted or moved from between two others | Yes — its successor's `prev_hash` no longer matches. |
| The newest N rows deleted (truncation) | **No.** The chain links backwards, so a shorter chain still walks clean. The only signal is `checked` falling between two polls of `/audit/verify` — which is why the deployment checklist says to poll it and keep the count. |
| The whole chain recomputed after an edit | **No.** The hash is an unkeyed sha256 over public columns and the algorithm is in this repo, so anyone with `UPDATE` on `audit_log` can rewrite it consistently. |

Closing either gap means putting something an attacker with `UPDATE` on one table cannot reach: an HMAC or KMS key (rejected in ADR 0005 — key management this kit deliberately does not have), or writing each head hash and count to an append-only store outside the database, which is a genuinely cheap improvement and deliberately not built here. Shipping the chain plus this table is more honest than shipping a stronger-sounding claim.

## Regulatory timeline accuracy

The corpus and docs reflect the **2026 Digital Omnibus** (formally adopted June 2026): Annex III high-risk obligations apply from **2 December 2027** (not August 2026); Art. 50 transparency obligations largely remain on the August 2026 schedule. The assistant answers from the ingested regulation text and does not track post-adoption amendments automatically — re-ingest to refresh.

## Known gaps (deliberate v1 scope)

- Redaction is regex-based (emails, phones, IBANs, cards); names and free-text PII need the documented Presidio swap (`redact_pii` is the single seam).
- No per-document ACLs — all authenticated roles search the same corpus.
- JWTs are not revocable before expiry (no token denylist).
- The token budget only bites when the provider reports `usage_metadata`; with a provider that reports none, `MAX_TOOL_ROUNDS` and the recursion limit are the only bounds left.
- The hash chain detects change, not who made it — it is not a signature, and a Postgres advisory lock (not a database constraint) is what keeps concurrent appends from forking it. The table above says what it misses.
- Demo seeding uses a shared password env var; SSO/OIDC is out of scope for the kit.
