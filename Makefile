.PHONY: db up migrate seed ingest ingest-smoke api dev test lint eval redteam require-env

require-env:
	@test -f .env || { \
	  echo "Missing .env — the backend service reads it via compose env_file."; \
	  echo "  cp .env.example .env"; \
	  echo "  # then set OPENAI_API_KEY, and JWT_SECRET=\$$(openssl rand -hex 32)"; \
	  exit 1; \
	}

db:              ## start postgres only (local dev)
	docker compose up -d postgres

up: require-env  ## full stack in docker
	docker compose --profile full up -d --build

migrate:
	cd backend && uv run alembic upgrade head

seed:
	cd backend && uv run python -m app.seed

ingest:          ## full corpus with contextual prefixes (needs OPENAI_API_KEY)
	cd backend && uv run python -m app.ingestion.pipeline

ingest-smoke:    ## 10 articles per regulation, no LLM prefixes
	cd backend && uv run python -m app.ingestion.pipeline --no-contextual --max-articles 10

api:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

dev:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check app tests && uv run ruff format --check app tests

eval:            ## RAGAS metrics over the golden dataset (needs running stack + key)
	cd backend && uv run --group evals python ../evals/run_evals.py

eval-retrieval:  ## retrieval ablation: hybrid vs vector-only vs text-only (no judge)
	cd backend && uv run --group evals python ../evals/run_retrieval_eval.py --mode hybrid
	cd backend && uv run --group evals python ../evals/run_retrieval_eval.py --mode vector_only
	cd backend && uv run --group evals python ../evals/run_retrieval_eval.py --mode text_only

redteam:         ## prompt-injection / PII / refusal suite
	cd backend && uv run --group evals python ../evals/run_redteam.py

promote:         ## promote the newest passing run to the committed baseline (SUITE=ragas|redteam)
	cd backend && uv run --group evals python ../evals/promote.py $(SUITE)

gate:            ## check the committed baselines against evals/thresholds.yaml (offline, free)
	cd backend && uv run --group dev --group evals pytest tests/test_eval_gate.py -q
