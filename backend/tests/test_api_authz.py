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
from app.security.rbac import (  # noqa: E402
    hash_password,
    issue_token,
    may_decide_approval,
)

_TABLES = [
    Base.metadata.tables[name] for name in ("users", "conversations", "approvals", "audit_log")
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


class TestListingAndDecisionAgree:
    """The two surfaces had drifted: /approvals was admin-wide, /chat/{id}/resume
    was owner-only, so every cross-user row in an admin's queue was a 403 waiting
    to happen. Both now derive from `may_decide_approval`, and this pins the
    correspondence rather than restating either rule.
    """

    async def test_every_listed_approval_is_one_the_caller_can_decide(self, env):
        client, users = env
        owners = {"thread-alice": users["alice"].id, "thread-bob": users["bob"].id}

        for name in ("alice", "bob", "root"):
            caller = users[name]
            body = (await client.get("/approvals", headers=_auth(caller))).json()
            listed = [row["thread_id"] for row in body["pending"]]
            assert listed, f"{name} was shown an empty queue"
            for thread in listed:
                assert may_decide_approval(owners[thread], caller), (
                    f"{name} was shown {thread} but cannot decide it"
                )

    async def test_every_unlisted_approval_is_one_the_caller_cannot_decide(self, env):
        client, users = env
        owners = {"thread-alice": users["alice"].id, "thread-bob": users["bob"].id}

        for name in ("alice", "bob", "root"):
            caller = users[name]
            body = (await client.get("/approvals", headers=_auth(caller))).json()
            listed = {row["thread_id"] for row in body["pending"]}
            for thread in set(owners) - listed:
                assert not may_decide_approval(owners[thread], caller), (
                    f"{name} can decide {thread} but was never shown it"
                )

    async def test_an_admin_can_decide_another_users_report(self, env):
        _, users = env
        assert may_decide_approval(users["alice"].id, users["root"])

    async def test_an_analyst_cannot_decide_another_analysts_report(self, env):
        _, users = env
        assert not may_decide_approval(users["alice"].id, users["bob"])

    async def test_a_missing_conversation_is_denied_to_everyone(self, env):
        _, users = env
        # Absence is not permission: an unknown thread_id was once resumable
        # by any analyst because the ownership check was skipped entirely.
        assert not may_decide_approval(None, users["root"])
        assert not may_decide_approval(None, users["alice"])
