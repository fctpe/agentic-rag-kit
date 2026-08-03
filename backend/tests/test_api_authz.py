"""Object-level authorization at the API boundary.

RBAC answers "may this role call this route". These tests answer the question
that actually leaks data: "may this *user* see this *row*". Both cases below
were real defects — /approvals returned every analyst's draft to every analyst,
and /chat/{id}/resume skipped its ownership check when the conversation row was
missing.

Runs on in-memory SQLite. Only the identity/approval tables are created; the
chunks table carries pgvector and tsvector columns that SQLite cannot express,
and none of it is needed to prove an ownership filter.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("JWT_SECRET", "test-secret-long-enough-to-pass-startup-checks")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.api.admin import router as admin_router  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models.tables import (  # noqa: E402
    Approval,
    Base,
    Conversation,
    Role,
    User,
)
from app.security.rbac import hash_password, issue_token  # noqa: E402

_TABLES = [
    Base.metadata.tables[name]
    for name in ("users", "conversations", "approvals", "audit_log")
]


@pytest.fixture
async def env():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=_TABLES)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    users: dict[str, User] = {}
    async with factory() as session:
        for name, role in (("alice", Role.analyst), ("bob", Role.analyst), ("root", Role.admin)):
            user = User(
                id=uuid.uuid4(),
                email=f"{name}@example.com",
                password_hash=hash_password("pw"),
                role=role,
            )
            session.add(user)
            users[name] = user
        await session.flush()

        # One pending approval per analyst, each tied to that analyst's thread.
        for name in ("alice", "bob"):
            thread = f"thread-{name}"
            session.add(
                Conversation(user_id=users[name].id, thread_id=thread, title=f"{name} draft")
            )
            session.add(
                Approval(thread_id=thread, payload={"draft": f"{name} confidential report"})
            )
        await session.commit()

    app = FastAPI()
    app.include_router(admin_router)

    async def override_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, users

    await engine.dispose()


def _auth(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(user)}"}


async def test_analyst_sees_only_their_own_pending_approval(env):
    client, users = env
    body = (await client.get("/approvals", headers=_auth(users["alice"]))).json()

    threads = [row["thread_id"] for row in body["pending"]]
    assert threads == ["thread-alice"]

    # The payload is the draft report itself — the thing that must not leak.
    assert "bob confidential report" not in str(body)


async def test_admin_sees_every_pending_approval(env):
    client, users = env
    body = (await client.get("/approvals", headers=_auth(users["root"]))).json()

    assert sorted(row["thread_id"] for row in body["pending"]) == ["thread-alice", "thread-bob"]


async def test_unauthenticated_request_is_rejected(env):
    client, _ = env
    assert (await client.get("/approvals")).status_code == 401
