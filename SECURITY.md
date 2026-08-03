# Security

The full threat model — OWASP LLM Top 10 (2025) mapping, EU AI Act Art. 12/14 framing, and the deliberate v1 gaps — is [docs/security.md](docs/security.md). This file is the short version and the reporting channel.

## Authentication and authorization

- JWT (HS256, 12 h TTL) from `POST /auth/login`; passwords are scrypt-hashed (n=2¹⁴, r=8, p=1) and compared with `secrets.compare_digest` (`app/security/rbac.py`). The app refuses to start when `JWT_SECRET` is unset, shorter than 16 characters, or one of a few obvious placeholders (`app/main.py`) — a forgeable signing key is not something to discover from request logs.
- The token carries a role but the role is not read from it: `get_current_user` re-loads the user row every request, so a demotion applies on the next call. The token itself is **not revocable** before expiry — there is no denylist.
- Three ordered roles (`viewer` → chat, `analyst` → search/approvals/resume, `admin` → audit). Every privileged route declares its minimum through `require_role`.
- Role alone is not authorization, so ownership is checked per object. `POST /chat` rejects a `thread_id` owned by another user; `POST /chat/{thread_id}/resume` fails closed — a missing `Conversation` row is a 403 rather than a skipped check, because `/chat` creates that row before the graph ever runs; `GET /approvals` scopes to the caller's own conversations unless they are an admin, since an approval payload *is* the draft report. Below that there are **no per-document ACLs**: every authenticated role searches the same corpus.

## Untrusted input

- Redaction runs in `guard_input` before text can reach the model, the checkpointer or a trace, and again at the database boundary in `app/api/chat.py` before the message row is written. The scope is structural identifiers only — emails, phone numbers, IBANs, card numbers (`app/security/redaction.py`). **Names and free-text PII are not covered.** `redact_pii` is the single seam for a Presidio swap.
- Prompt injection is handled in three layers rather than one: input heuristics that refuse and audit-log override/hijack/probe/smuggling/exfiltration patterns (`app/security/injection.py`), retrieved chunks delimited as `<source>` blocks the system prompt declares untrusted data, and a human approval gate on report-type output. `evals/run_redteam.py` asserts all three. The heuristics are regexes and will not catch a paraphrase; they are the cheap layer, and the two structural layers are why the cheap layer is allowed to stay cheap.

## Audit trail

`audit_log` is append-only — the application exposes no update or delete path — and hash-chained: every row stores the previous row's hash alongside its own. `GET /audit/verify` (admin) walks the chain in the order the database assigned (`seq`, a bigserial) and names the first row that failed its own hash or its link. Appends are serialized on a Postgres advisory lock, because two writers reading the same tail would fork the chain, and a fork is indistinguishable from tampering.

It is tamper-*evidence*, not tamper-proofing. Two gaps are worth stating here rather than leaving to a reader's optimism:

- **Truncation is undetectable.** The chain links backwards, so deleting the newest N rows leaves a shorter chain that still walks clean. The only signal is the entry count falling between two polls of `/audit/verify`, which is why the deployment checklist says to poll it and keep the count.
- **The hash is unkeyed.** It is a plain sha256 over public columns and the algorithm is in this repo, so anyone with `UPDATE` on `audit_log` can edit a row and recompute the whole chain consistently. Closing that requires something an attacker with write access to one table cannot reach — an HMAC or KMS key, or writing each head hash and count to an append-only store outside the database. [docs/security.md](docs/security.md#audit-trail-eu-ai-act-art-12-framing) records why neither is built here.

## Abuse guards

- A run is bounded on four independent axes, because each one alone has a way around it: input length (8 000 characters, refused in `guard_input`), tool rounds (`MAX_TOOL_ROUNDS = 6`), a per-request token budget accumulated from `usage_metadata` (`MAX_TOTAL_TOKENS = 80_000`), and the graph `recursion_limit`. Individual model calls time out at 60 s.
- Over budget, the run stops and says so. It never returns a truncated answer as though it were complete, and the grounding verdict for such a run is `false` with the reason attached — a run that verified nothing is not a verified run.
- The token budget only bites when the provider reports `usage_metadata`. Against a provider that reports none, the round cap and the recursion limit are the only bounds left.

## Reporting

Open a GitHub security advisory or email the address on the profile of [@fctpe](https://github.com/fctpe). Please do not open public issues for vulnerabilities.
