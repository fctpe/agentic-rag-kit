from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api import admin, auth, chat, search
from app.config import get_settings
from app.db import dispose_engine
from app.observability import request_context, setup_observability, shutdown_observability

_WEAK_JWT_SECRETS = {"", "change-me", "changeme", "secret", "dev"}


def _checkpointer_url() -> str:
    # The LangGraph Postgres saver speaks psycopg, not asyncpg.
    return get_settings().database_url.replace("+asyncpg", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_observability()
    if settings.jwt_secret.strip().lower() in _WEAK_JWT_SECRETS or len(settings.jwt_secret) < 16:
        raise RuntimeError(
            "JWT_SECRET is unset or too weak. Set a strong secret: openssl rand -hex 32"
        )
    async with AsyncPostgresSaver.from_conn_string(_checkpointer_url()) as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        yield
    await dispose_engine()
    shutdown_observability()


app = FastAPI(
    title="agentic-rag-kit",
    description="Agentic RAG over the EU AI Act and GDPR with citations, "
    "human-in-the-loop approvals, and an append-only audit trail.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-Id"],
)


@app.middleware("http")
async def correlate(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """One correlation id per request, on every log line it produces. Bound
    before call_next so the SSE generator — which outlives this middleware —
    inherits it with the task context."""
    with request_context(request.headers.get("x-request-id")) as request_id:
        response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


app.include_router(auth.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
