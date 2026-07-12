.PHONY: db up migrate seed ingest ingest-smoke api dev test lint eval redteam

db:              ## start postgres only (local dev)
	docker compose up -d postgres

up:              ## full stack in docker
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

redteam:         ## prompt-injection / PII / refusal suite
	cd backend && uv run --group evals python ../evals/run_redteam.py
