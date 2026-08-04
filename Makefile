.PHONY: frontend-deps db up migrate seed ingest ingest-fixture ingest-smoke prefix-cache embedding-cache corpus-digest api dev test lint eval eval-retrieval redteam promote gate require-env

# Every app target runs from backend/, which puts the repo-root .env out of
# reach of both pydantic-settings (it resolves env_file against the working
# directory) and the provider SDKs (they read the process environment). uv
# loads it for them, so `cp .env.example .env` is the only setup step there is.
# Every target runs from backend/, so the repo-root .env is out of reach of both
# pydantic-settings (which resolves env_file against cwd) and the provider SDKs.
UV = uv run --env-file ../.env
# Same, but the file is optional: uv aborts on a missing --env-file, and the two
# targets that need no secret must still work before anyone has written one.
UV_OPTENV = uv run $(if $(wildcard .env),--env-file ../.env,)

require-env:
	@test -f .env || { \
	  echo "Missing .env — compose reads it via env_file, and the host targets load it with uv."; \
	  echo "  cp .env.example .env"; \
	  echo "  # then set OPENAI_API_KEY, and JWT_SECRET=\$$(openssl rand -hex 32)"; \
	  exit 1; \
	}

db:              ## start postgres only (local dev)
	# --wait, because `up -d` returns when the container is created, not when
	# Postgres accepts connections. Without it `make db migrate` races and the
	# migration dies on connection refused.
	docker compose up -d --wait postgres

up: require-env  ## full stack in docker
	docker compose --profile full up -d --build

# No require-env on these two deliberately. They need DATABASE_URL and no
# secret, and its default is the database `make db` just started — so a missing
# .env cannot make them do the wrong thing, only the documented thing. The gate
# belongs where running without configuration would actually be wrong (ingest,
# api, up); putting it everywhere teaches people to ignore it.
migrate:
	cd backend && $(UV_OPTENV) alembic upgrade head

seed:
	cd backend && $(UV_OPTENV) python -m app.seed

ingest: require-env          ## full corpus from live EUR-Lex, contextual prefixes (needs OPENAI_API_KEY)
	cd backend && $(UV) python -m app.ingestion.pipeline

# No require-env on either: since the embedding vectors are committed alongside
# the text and the prefixes, a fixture ingest calls no provider at all. It needs
# a database and nothing else, which is what makes the committed corpus — and so
# every committed retrieval number — reproducible by someone with no key.
ingest-fixture:              ## the committed corpus: text, prefixes and vectors all read from data/fixtures/ (no key)
	cd backend && $(UV_OPTENV) python -m app.ingestion.pipeline --source fixture

ingest-smoke:                ## 10 units per regulation from data/fixtures/ (no key)
	cd backend && $(UV_OPTENV) python -m app.ingestion.pipeline --source fixture --max-units 10

corpus-digest:               ## SHA-256 of the ingested chunk text AND of chunks.embedding
	cd backend && $(UV_OPTENV) python -m app.ingestion.corpus_digest

# No require-env: this only fetches and parses, so it needs no provider key.
refresh-fixtures:            ## regenerate data/fixtures/ from Cellar (no key needed)
	cd backend && $(UV_OPTENV) python -m app.ingestion.refresh_fixtures

# Deliberately NOT a dependency of refresh-fixtures, and deliberately not run by
# anything else. It is one model call per chunk (284), so nobody should reach it
# by typing a target that sounds free. `make ingest-fixture` fails closed when
# the committed cache no longer covers the corpus, which is the prompt to run it.
prefix-cache: require-env    ## regenerate data/fixtures/context_prefixes.json — COSTS MONEY (one model call per chunk)
	cd backend && $(UV) python -m app.ingestion.refresh_prefixes

# The vectors, for the same reason and with the same rules. Both regulations and
# the golden question set, or neither: they are three files describing one
# measurement, and a corpus embedded with one model against questions embedded
# with another is not a measurement of anything. Order matters — the embedded
# string is prefix + content, so this runs AFTER `make prefix-cache`, never
# before, or it would commit vectors for text that has since been rewritten.
embedding-cache: require-env ## regenerate the committed embedding vectors (corpus + golden queries) — COSTS MONEY
	cd backend && $(UV) python -m app.ingestion.refresh_embeddings
	cd backend && $(UV) --group evals python ../evals/query_embeddings.py

api: require-env
	cd backend && $(UV) uvicorn app.main:app --reload --port 8000

dev: frontend-deps
	cd frontend && npm run dev

frontend-deps:
	# The backend installs itself through `uv run`; the frontend has no
	# equivalent, so a clean clone reached `npm run dev` with no node_modules
	# and died on `next: command not found`.
	@test -d frontend/node_modules || (cd frontend && npm ci)

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check app tests && uv run ruff format --check app tests

eval: require-env            ## RAGAS metrics over the golden dataset (needs running stack + key)
	cd backend && $(UV) --group evals python ../evals/run_evals.py

eval-retrieval: require-env  ## retrieval ablation: hybrid vs vector-only vs text-only (no judge)
	cd backend && $(UV) --group evals python ../evals/run_retrieval_eval.py --mode hybrid
	cd backend && $(UV) --group evals python ../evals/run_retrieval_eval.py --mode vector_only
	cd backend && $(UV) --group evals python ../evals/run_retrieval_eval.py --mode text_only

redteam: require-env         ## prompt-injection / PII / refusal suite
	cd backend && $(UV) --group evals python ../evals/run_redteam.py

promote:         ## promote the newest passing run to the committed baseline (SUITE=ragas|redteam|retrieval_hybrid|retrieval_vector_only|retrieval_text_only)
	cd backend && uv run --group evals python ../evals/promote.py $(SUITE)

gate:            ## check the committed baselines against evals/thresholds.yaml (offline, free)
	cd backend && uv run --group dev --group evals pytest tests/test_eval_gate.py -q
