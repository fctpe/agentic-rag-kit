# Deployment

## Local (development)

```bash
cp .env.example .env            # set OPENAI_API_KEY, JWT_SECRET
make db                         # postgres + pgvector on :5434
make migrate && make seed
make ingest                     # full corpus (or: make ingest-smoke)
make api                        # FastAPI on :8000
make dev                        # Next.js on :3000
```

## Docker (full stack)

```bash
docker compose --profile full up -d --build
docker compose exec backend uv run --no-sync alembic upgrade head
docker compose exec backend uv run --no-sync python -m app.seed
docker compose exec backend uv run --no-sync python -m app.ingestion.pipeline
```

## Observability

Two independent sinks, both optional, neither required for the app to run.

**Langfuse.** Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`. The low-friction default is **Langfuse Cloud EU region** (`https://cloud.langfuse.com`) — trace data stays in the EU. For strict data-residency, self-host Langfuse v3 alongside this stack (web + worker + ClickHouse + Redis + MinIO, all containers on UTC); it is intentionally **not** part of the main compose file to keep the pilot footprint small. With the keys unset, no handler is installed.

**OTLP spans.** Set `OTEL_EXPORTER_OTLP_ENDPOINT` to a collector base URL (`/v1/traces` is appended) and `OTEL_EXPORTER_OTLP_HEADERS` to whatever auth it needs, in `key=value,key2=value2` form. Langfuse itself ingests OTLP at `$LANGFUSE_HOST/api/public/otel` with an `Authorization=Basic <base64 public:secret>` header, so this standardises the integration above rather than competing with it. Spans follow the GenAI semantic conventions: a run root, one per graph node, one per tool call, one per retrieval, and `gen_ai.usage.input_tokens` / `output_tokens` on every model call. Set `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` from your provider's pricing page to get `ragkit.usage.cost_usd` alongside them; left at 0 the spans carry tokens and no cost, because a fabricated cost is worse than an absent one. **With the endpoint unset there is no exporter, no background thread and no network call** — see ADR 0006 for why the conventions are pinned rather than floated.

**Logs.** Always on, always JSON, one object per line on stdout, level from `LOG_LEVEL`. Every line carries a `request_id` taken from an upstream `X-Request-Id` or minted per request and echoed back on the response; `trace_id` and `span_id` join it whenever tracing is enabled.

## Production checklist

- [ ] Real `JWT_SECRET` (`openssl rand -hex 32`) and a non-default `SEED_PASSWORD` — or replace seeding with your IdP
- [ ] Postgres with backups; the audit log and LangGraph checkpoints live there too
- [ ] TLS termination in front of FastAPI (the app serves plain HTTP)
- [ ] Restrict CORS origins in `app/main.py` to your frontend host
- [ ] Rate limiting at the proxy layer — the app bounds a single request (tool rounds, a per-request token budget, a graph recursion limit, per-call LLM timeouts; see ADR 0005), not the rate at which requests arrive
- [ ] Poll `GET /audit/verify` (admin) on a schedule — it walks the audit hash chain and reports the first entry where it breaks. **Keep the `checked` count between polls**: a chain that only ever grows is the check, because truncation from the newest end walks clean (the table in [docs/security.md](security.md#audit-trail-eu-ai-act-art-12-framing) says what else it misses)
- [ ] Ship stdout to your log store and set `OTEL_EXPORTER_OTLP_ENDPOINT`; alert on `ragkit.grounded=false` and on runs that end at the token budget. `grounded` fails closed, so that alert also covers the runs that verified *nothing* — refusals, and answers built on zero retrieved sources; `gen_ai.evaluation.score.label` on the `node.verify` span separates `ungrounded` from `unverified`
- [ ] Re-run `make eval` and `make redteam` after any prompt or retrieval change; wire them into CI as a merge gate
- [ ] Periodic re-ingest to track regulation amendments
