# Deployment

## Local (development)

```bash
cp .env.example .env            # set OPENAI_API_KEY, JWT_SECRET
make db                         # postgres + pgvector on :5434
make migrate && make seed
make ingest-fixture             # full corpus, offline (or: make ingest-smoke / make ingest)
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

The image build context is `backend/`, so `data/fixtures/` is not in the image and
`--source fixture` has nothing to read there. In-container ingestion goes to live
EUR-Lex; run `make ingest-fixture` on the host against the same database if the
endpoint is refusing to serve (see the README limitation).

## Kubernetes

Kustomize base plus a local overlay under [`deploy/`](../deploy). The base carries no
namespace and no Secret — the overlay picks the namespace, the images and the hostnames, and
the values come from outside the tree. `.github/workflows/k8s.yml` runs exactly the sequence
below on kind for every push.

```bash
docker build -t ragkit-backend:dev backend
docker build --build-arg NEXT_PUBLIC_API_BASE=http://api.ragkit.localtest.me \
  -t ragkit-frontend:dev frontend

kind create cluster --name ragkit --config deploy/kind-cluster.yaml
kind load docker-image ragkit-backend:dev ragkit-frontend:dev --name ragkit
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.15.1/deploy/static/provider/kind/deploy.yaml
kubectl -n ingress-nginx wait --for=condition=available deploy/ingress-nginx-controller --timeout=300s

# The namespace first: its pod-security label has to exist before any pod is admitted.
kubectl apply -f deploy/overlays/local/namespace.yaml

PG_PASSWORD="$(openssl rand -hex 16)"
kubectl -n ragkit create secret generic ragkit-postgres \
  --from-literal=POSTGRES_USER=rag \
  --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
  --from-literal=POSTGRES_DB=ragkit
kubectl -n ragkit create secret generic ragkit-secrets \
  --from-literal=DATABASE_URL="postgresql+asyncpg://rag:${PG_PASSWORD}@postgres:5432/ragkit" \
  --from-literal=JWT_SECRET="$(openssl rand -hex 32)" \
  --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY"   # from your environment, not from a file

kubectl apply -k deploy/overlays/local
kubectl -n ragkit wait --for=condition=complete job/ragkit-migrate --timeout=300s
kubectl -n ragkit rollout status deploy/ragkit-backend --timeout=300s
curl -H "Host: api.ragkit.localtest.me" http://127.0.0.1/health
```

**Secrets never enter the tree.** `envFrom.secretRef` names `ragkit-secrets`, nothing
generates it, and it is not `optional` — a backend pod with no Secret fails to start instead of
coming up on defaults. In a real cluster, point External Secrets (or your own sync) at the ARNs
the Terraform module outputs; the two `kubectl create secret` calls above are the local
equivalent, and `$OPENAI_API_KEY` is read from your shell.

**The migration Job gates the rollout.** `job/ragkit-migrate` runs `alembic upgrade head`
(which is also what creates the pgvector extension), and every backend pod has a
`wait-for-schema` init container that blocks until `alembic current` reports `(head)`. Kubernetes
applies a Job and a Deployment at the same time; without that init container the new pods come
up against the old schema. A pod stuck in `Init:0/1` means the Job has not finished — read
`kubectl -n ragkit logs job/ragkit-migrate`.

**Re-deploying a new image tag needs the old Job gone.** A Job's pod template is immutable, so
`kubectl apply -k` fails on the second image tag. Run
`kubectl -n ragkit delete job/ragkit-migrate --ignore-not-found` first.

**Liveness and readiness are different endpoints, on purpose.**

| probe | path | touches Postgres | what a failure does |
|---|---|---|---|
| liveness | `/health/live` | no | **restarts the container** |
| readiness | `/health` | `SELECT 1` | removes the pod from the Service |

Wire liveness to `/health` and a Postgres failover restart-loops every replica at once, exactly
when the pods were fine and the database was not. `/health` used to return `{"status": "ok"}`
without opening a connection, which made any readiness probe pointed at it a test that the HTTP
server could answer itself; it now fails closed with 503, and `backend/tests/test_health.py`
plus the outage step in `.github/workflows/k8s.yml` hold the split.

**Two hostnames, not one.** `NEXT_PUBLIC_API_BASE` is inlined into the browser bundle at build
time, so the frontend image has to be built with the same `api.` hostname the Ingress serves —
setting it in the Deployment does nothing.

## Managed infrastructure (Terraform)

[`deploy/terraform`](../deploy/terraform) covers the parts Kubernetes cannot hold: the Postgres
instance, the Secrets Manager container the app's credentials live in, and the two DNS records
pointing at the ingress load balancer.

```bash
cd deploy/terraform
terraform init -backend=false && terraform validate   # no credentials needed; runs in CI
```

It is a module — no provider block, no backend, no VPC. Pass in an existing DB subnet group and
security groups; the caller configures the region and where state lives.

Three deliberate omissions:

- **No password anywhere.** `manage_master_user_password = true` has RDS generate and rotate the
  master password into a secret it owns. A `random_password` or a variable would put the
  production database password in Terraform state in plaintext.
- **No secret *values*.** The `aws_secretsmanager_secret` is versionless; every argument
  Terraform sees ends up in state, so `JWT_SECRET` and `OPENAI_API_KEY` are written once with
  `aws secretsmanager put-secret-value` (the command is in `main.tf`). Terraform owns the
  container, never the contents.
- **No `CREATE EXTENSION vector`.** It is the first statement of the first Alembic migration, so
  the migration Job does it. Doing it from Terraform would mean a second provider with a live
  route into a private subnet, for one statement the app already owns.

## Observability

Two independent sinks, both optional, neither required for the app to run.

**Langfuse.** Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`. The low-friction default is **Langfuse Cloud EU region** (`https://cloud.langfuse.com`) — trace data stays in the EU. For strict data-residency, self-host Langfuse v3 alongside this stack (web + worker + ClickHouse + Redis + MinIO, all containers on UTC); it is intentionally **not** part of the main compose file to keep the pilot footprint small. With the keys unset, no handler is installed.

**OTLP spans.** Set `OTEL_EXPORTER_OTLP_ENDPOINT` to a collector base URL (`/v1/traces` is appended) and `OTEL_EXPORTER_OTLP_HEADERS` to whatever auth it needs, in `key=value,key2=value2` form. Langfuse itself ingests OTLP at `$LANGFUSE_HOST/api/public/otel` with an `Authorization=Basic <base64 public:secret>` header, so this standardises the integration above rather than competing with it. Spans follow the GenAI semantic conventions: a run root, one per graph node, one per tool call, one per retrieval, and `gen_ai.usage.input_tokens` / `output_tokens` on every model call. Set `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` from your provider's pricing page to get `ragkit.usage.cost_usd` alongside them; left at 0 the spans carry tokens and no cost, because a fabricated cost is worse than an absent one. **With the endpoint unset there is no exporter, no background thread and no network call** — see ADR 0006 for why the conventions are pinned rather than floated.

**Logs.** Always on, always JSON, one object per line on stdout, level from `LOG_LEVEL`. Every line carries a `request_id` taken from an upstream `X-Request-Id` or minted per request and echoed back on the response; `trace_id` and `span_id` join it whenever tracing is enabled.

## Production checklist

- [ ] Real `JWT_SECRET` (`openssl rand -hex 32`) and a non-default `SEED_PASSWORD` — or replace seeding with your IdP
- [ ] Postgres with backups; the audit log and LangGraph checkpoints live there too
- [ ] TLS termination in front of FastAPI (the app serves plain HTTP) — on Kubernetes that is a
      `tls:` block and an issuer annotation on the Ingress, which the base leaves to your overlay
- [ ] Health checks split the way the table above splits them: readiness on `/health`, liveness on
      `/health/live`. A load balancer pointed at `/health` for both will pull the whole fleet
      during a failover *and* restart it
- [ ] Restrict CORS origins in `app/main.py` to your frontend host
- [ ] Rate limiting at the proxy layer — the app bounds a single request (tool rounds, a per-request token budget, a graph recursion limit, per-call LLM timeouts; see ADR 0005), not the rate at which requests arrive
- [ ] Poll `GET /audit/verify` (admin) on a schedule — it walks the audit hash chain and reports the first entry where it breaks. **Keep the `checked` count between polls**: a chain that only ever grows is the check, because truncation from the newest end walks clean (the table in [docs/security.md](security.md#audit-trail-eu-ai-act-art-12-framing) says what else it misses)
- [ ] Ship stdout to your log store and set `OTEL_EXPORTER_OTLP_ENDPOINT`; alert on `ragkit.grounded=false` and on runs that end at the token budget. `grounded` fails closed, so that alert also covers the runs that verified *nothing* — refusals, and answers built on zero retrieved sources; `gen_ai.evaluation.score.label` on the `node.verify` span separates `ungrounded` from `unverified`. It does **not** cover citation resolution: a bracket the `resolve_citations` node refused to link is reported in `citation_issues` on the `done` / `approval_required` payload and logged as `citation markers could not all be linked`, deliberately outside the grounding verdict so this alert does not fire on formatting (ADR 0007)
- [ ] Re-run `make eval` and `make redteam` after any prompt or retrieval change; wire them into CI as a merge gate
- [ ] Periodic re-ingest to track regulation amendments
