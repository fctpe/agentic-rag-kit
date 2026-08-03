"""Tamper-evidence of the audit log.

The log is framed as EU AI Act Art. 12 record-keeping, so "the application has
no delete path" is not the claim that matters — what matters is whether a change
made *around* the application is detectable. These tests edit and delete rows
with direct SQL and assert /audit/verify names the entry where the chain breaks.

Same in-memory SQLite + httpx ASGI setup as test_api_authz.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

os.environ.setdefault("JWT_SECRET", "test-secret-long-enough-to-pass-startup-checks")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import delete, event, select, update  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    async_sessionmaker,
    create_async_engine,
)

from app.api import admin  # noqa: E402
from app.api.admin import router as admin_router  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models.tables import AuditLog, Base, Role, User  # noqa: E402
from app.security.audit import chain_hash, record_audit  # noqa: E402
from app.security.rbac import hash_password, issue_token  # noqa: E402

_TABLES = [Base.metadata.tables[name] for name in ("users", "audit_log")]


@pytest.fixture
async def env():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=_TABLES)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    users: dict[str, User] = {}
    async with factory() as session:
        for name, role in (("root", Role.admin), ("alice", Role.analyst)):
            user = User(
                id=uuid.uuid4(),
                email=f"{name}@example.com",
                password_hash=hash_password("pw"),
                role=role,
            )
            session.add(user)
            users[name] = user
        await session.commit()

    app = FastAPI()
    app.include_router(admin_router)

    async def override_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, factory, users

    await engine.dispose()


def _auth(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(user)}"}


async def _write(factory, *actions: str) -> None:
    for action in actions:
        async with factory() as session:
            await record_audit(session, action, resource=f"res-{action}")


async def _entries(factory) -> list[AuditLog]:
    async with factory() as session:
        rows = await session.scalars(select(AuditLog).order_by(AuditLog.seq))
        return list(rows)


async def test_first_entry_opens_the_chain(env):
    _, factory, _ = env
    await _write(factory, "auth.login")

    entry = (await _entries(factory))[0]
    assert entry.prev_hash == ""
    assert entry.entry_hash == chain_hash(entry)


async def test_each_entry_carries_the_previous_hash(env):
    _, factory, _ = env
    await _write(factory, "auth.login", "chat.query", "chat.answered")

    entries = await _entries(factory)
    assert [entry.prev_hash for entry in entries[1:]] == [
        entry.entry_hash for entry in entries[:-1]
    ]


async def test_verify_accepts_an_untouched_chain(env):
    client, factory, users = env
    await _write(factory, "auth.login", "chat.query", "chat.answered")

    body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
    assert body == {"ok": True, "checked": 3, "broken_at": None, "reason": None}


async def test_empty_log_verifies(env):
    client, _, users = env
    body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
    assert body["ok"] is True
    assert body["checked"] == 0


async def test_verify_detects_an_edited_entry(env):
    client, factory, users = env
    await _write(factory, "auth.login", "chat.query", "chat.answered")
    target = (await _entries(factory))[1]

    async with factory() as session:
        await session.execute(
            update(AuditLog).where(AuditLog.id == target.id).values(resource="scrubbed")
        )
        await session.commit()

    body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
    assert body["ok"] is False
    assert body["broken_at"] == str(target.id)
    assert body["checked"] == 1
    assert "entry_hash" in body["reason"]


async def test_verify_detects_a_deleted_entry(env):
    client, factory, users = env
    await _write(factory, "auth.login", "chat.query", "chat.answered")
    entries = await _entries(factory)

    async with factory() as session:
        await session.execute(delete(AuditLog).where(AuditLog.id == entries[1].id))
        await session.commit()

    body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
    assert body["ok"] is False
    # The deleted row is gone; the break shows up as its successor's dangling link.
    assert body["broken_at"] == str(entries[2].id)
    assert "prev_hash" in body["reason"]


async def test_verify_requires_admin(env):
    client, _, users = env
    assert (await client.get("/audit/verify", headers=_auth(users["alice"]))).status_code == 403
    assert (await client.get("/audit/verify")).status_code == 401


class TestChainOrderComesFromTheDatabase:
    """The finding: the writer linked onto the tail found by
    (created_at DESC, id DESC) while the verifier walked (created_at, id). A
    uuid4 does not break ties in write order and a wall clock is not monotonic,
    so an untouched log reported tampering. Both ends now use the sequence the
    database assigns.
    """

    async def test_a_clock_that_steps_backwards_does_not_break_the_chain(self, env, monkeypatch):
        # An NTP correction mid-traffic, reduced to three writes. Ordering by
        # created_at puts them in the reverse of the order they were written,
        # which is a fork the walk reports as tampering.
        client, factory, users = env
        base = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
        steps = iter([base + timedelta(seconds=2), base + timedelta(seconds=1), base])
        monkeypatch.setattr("app.security.audit.utcnow", lambda: next(steps))
        await _write(factory, "auth.login", "chat.query", "chat.answered")

        body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
        assert body == {"ok": True, "checked": 3, "broken_at": None, "reason": None}

    async def test_writes_inside_one_clock_tick_do_not_break_the_chain(self, env, monkeypatch):
        # A clock too coarse to separate three appends — the case the reviewer
        # hit — leaves (created_at, id) ordering on the uuid4 alone.
        client, factory, users = env
        monkeypatch.setattr(
            "app.security.audit.utcnow", lambda: datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
        )
        await _write(factory, "auth.login", "chat.query", "chat.answered")

        body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
        assert body == {"ok": True, "checked": 3, "broken_at": None, "reason": None}
        # Write order, from the database, regardless of what the clock said.
        assert [entry.resource for entry in await _entries(factory)] == [
            "res-auth.login",
            "res-chat.query",
            "res-chat.answered",
        ]

    async def test_the_sequence_is_assigned_by_the_database(self, env):
        _, factory, _ = env
        await _write(factory, "auth.login", "chat.query", "chat.answered")

        seqs = [entry.seq for entry in await _entries(factory)]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 3


async def test_verify_pages_the_walk_instead_of_loading_the_table(env, monkeypatch):
    """The finding: /audit/verify is on the request path and deployment.md tells
    operators to poll it, but it materialised the whole audit_log through the ORM
    in one select. Counting statements is the only way to say that from outside.
    """
    client, factory, users = env
    monkeypatch.setattr(admin, "VERIFY_PAGE_SIZE", 2)
    await _write(factory, "a", "b", "c", "d", "e")

    selects: list[str] = []

    def count(conn, cursor, statement, parameters, context, executemany):
        if "FROM audit_log" in statement:
            selects.append(statement)

    event.listen(Engine, "before_cursor_execute", count)
    try:
        body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
    finally:
        event.remove(Engine, "before_cursor_execute", count)

    assert body == {"ok": True, "checked": 5, "broken_at": None, "reason": None}
    # Five rows, two per page: three pages of rows plus the empty one that ends
    # the walk. One statement would mean the whole table came back at once.
    assert len(selects) == 4
    assert all("LIMIT" in statement for statement in selects)


async def test_truncating_the_newest_entries_is_not_detected(env):
    """A limitation, pinned so the docs and the code cannot drift apart.

    docs/security.md and ADR 0005 state it: the chain links each entry to the one
    before it, so deleting from the *end* leaves a shorter chain that still walks
    clean. Only the `checked` count falling between two polls shows it.
    """
    client, factory, users = env
    await _write(factory, "auth.login", "chat.query", "chat.answered")
    entries = await _entries(factory)

    async with factory() as session:
        await session.execute(delete(AuditLog).where(AuditLog.seq >= entries[1].seq))
        await session.commit()

    body = (await client.get("/audit/verify", headers=_auth(users["root"]))).json()
    assert body["ok"] is True
    assert body["checked"] == 1
