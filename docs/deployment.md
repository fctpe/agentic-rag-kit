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

Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`. The low-friction default is **Langfuse Cloud EU region** (`https://cloud.langfuse.com`) — trace data stays in the EU. For strict data-residency, self-host Langfuse v3 alongside this stack (web + worker + ClickHouse + Redis + MinIO, all containers on UTC); it is intentionally **not** part of the main compose file to keep the pilot footprint small. Tracing is fully optional: with the keys unset, the app runs without emitting anything.

## Production checklist

- [ ] Real `JWT_SECRET` (`openssl rand -hex 32`) and a non-default `SEED_PASSWORD` — or replace seeding with your IdP
- [ ] Postgres with backups; the audit log and LangGraph checkpoints live there too
- [ ] TLS termination in front of FastAPI (the app serves plain HTTP)
- [ ] Restrict CORS origins in `app/main.py` to your frontend host
- [ ] Rate limiting at the proxy layer (the app enforces token/tool budgets, not request rates)
- [ ] Re-run `make eval` and `make redteam` after any prompt or retrieval change; wire them into CI as a merge gate
- [ ] Periodic re-ingest to track regulation amendments
