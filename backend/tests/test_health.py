"""Liveness and readiness answer different questions, and the probes in
deploy/base/backend-deployment.yaml depend on it.

Kubernetes restarts a container that fails liveness and only pulls a pod out of
the Service when it fails readiness. Wire both to the same database-touching
handler and one Postgres failover restart-loops every replica at once. So:
readiness fails closed when the session is broken, liveness never looks.

Runs on in-memory SQLite for the healthy case and on stub sessions for the
failure cases — `SELECT 1` is the whole check, and nothing here needs pgvector.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("JWT_SECRET", "test-secret-long-enough-to-pass-startup-checks")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.api import health  # noqa: E402
from app.db import get_session  # noqa: E402


class _RefusedSession:
    """A session whose every query dies the way an unreachable Postgres does."""

    async def execute(self, *_args, **_kwargs):
        raise ConnectionRefusedError("[Errno 111] Connect call failed")


class _HungSession:
    """A session that never answers — the failure a probe timeout exists for."""

    async def execute(self, *_args, **_kwargs):
        await asyncio.sleep(3600)


def _client(session) -> AsyncClient:
    app = FastAPI()
    app.include_router(health.router)

    async def override():
        yield session

    app.dependency_overrides[get_session] = override
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def sqlite_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


async def test_readiness_reports_ok_when_the_database_answers(sqlite_session):
    async with _client(sqlite_session) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


async def test_readiness_fails_closed_when_the_database_is_unreachable():
    async with _client(_RefusedSession()) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    # The type is what an operator reads off a probe; the DSN it was refused on
    # must not be in there (error_fields, ADR 0006).
    assert response.json() == {
        "status": "unavailable",
        "database": "builtins.ConnectionRefusedError",
    }


async def test_readiness_gives_up_before_the_kubelet_does(monkeypatch):
    monkeypatch.setattr(health, "READINESS_TIMEOUT_SECONDS", 0.05)
    async with _client(_HungSession()) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["database"] == "builtins.TimeoutError"


async def test_liveness_ignores_a_dead_database():
    """The one that matters: a database blip must not get the pod restarted."""
    async with _client(_RefusedSession()) as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
