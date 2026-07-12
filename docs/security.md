# Security & governance

## OWASP Top 10 for LLM Applications (2025) — what this kit demonstrably mitigates

| Risk | Mitigation in this codebase |
|---|---|
| LLM01 Prompt injection | Input heuristics (`app/security/injection.py`) refuse and audit-log override/hijack/probe/smuggling patterns; retrieved chunks are delimited as `<source>` data blocks the system prompt declares untrusted; report output passes a human gate. The red-team suite (`evals/run_redteam.py`) asserts all three layers. |
| LLM02 Sensitive information disclosure | Structural PII redaction (`app/security/redaction.py`) runs in `guard_input`, **before** text can reach the model, the checkpointer, or a Langfuse trace. Redaction events land in the audit log without the raw values. |
| LLM05 Improper output handling | Assistant output is rendered as markdown text, never executed or interpolated; citations are built server-side from retrieved rows, never parsed from model output. |
| LLM09 Misinformation | Grounded citations with article-level deep links plus a post-answer grounding audit that flags unsupported claims to the user. |
| LLM10 Unbounded consumption | Input length caps, tool-round caps (`MAX_TOOL_ROUNDS`), retrieval `final_k`, and per-request model calls bounded by graph shape. |

## RBAC

Three roles (`viewer` → chat, `analyst` → search/approvals/resume, `admin` → audit). JWT (HS256, 12h TTL) with scrypt password hashing. Every privileged route declares its minimum role via `require_role`.

## Audit trail (EU AI Act Art. 12 framing)

`audit_log` is append-only — the application exposes no update or delete path. Logged: logins (incl. failures), queries, answers (with grounding verdict + redaction/injection flags), approval requests and decisions with the deciding user. The human approval gate for report-type output maps to the Art. 14 human-oversight requirement.

## Regulatory timeline accuracy

The corpus and docs reflect the **2026 Digital Omnibus** (formally adopted June 2026): Annex III high-risk obligations apply from **2 December 2027** (not August 2026); Art. 50 transparency obligations largely remain on the August 2026 schedule. The assistant answers from the ingested regulation text and does not track post-adoption amendments automatically — re-ingest to refresh.

## Known gaps (deliberate v1 scope)

- Redaction is regex-based (emails, phones, IBANs, cards); names and free-text PII need the documented Presidio swap (`redact_pii` is the single seam).
- No per-document ACLs — all authenticated roles search the same corpus.
- JWTs are not revocable before expiry (no token denylist).
- Demo seeding uses a shared password env var; SSO/OIDC is out of scope for the kit.
