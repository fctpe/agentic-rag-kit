"""Append-only audit log, made tamper-evident by a hash chain.

"Append-only by convention" is not a record-keeping guarantee (EU AI Act
Art. 12): anyone with database access can edit or drop a row. Each entry
therefore hashes its own fields together with the previous entry's hash, so a
later edit or deletion is detectable — see GET /audit/verify.
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import AuditLog, utcnow

# Appends are serialized on this advisory lock: two writers that read the same
# tail concurrently would fork the chain, and a fork is indistinguishable from
# tampering when the chain is walked. Postgres only — SQLite has no equivalent,
# and the tests that run on it write from a single task. The key is arbitrary;
# it only has to be unique among advisory-lock users of this database.
_CHAIN_LOCK_KEY = 0x41554449


def _utc_iso(value: datetime) -> str:
    # SQLite drops tzinfo on the round trip, so a naive value is the UTC it was
    # written as; without this the hash differs between write and verify.
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC).isoformat()


def chain_hash(entry: AuditLog) -> str:
    """sha256 over the previous entry's hash plus this entry's own fields."""
    payload = json.dumps(
        {
            "prev": entry.prev_hash,
            "id": str(entry.id),
            "user_id": str(entry.user_id) if entry.user_id else None,
            "action": entry.action,
            "resource": entry.resource,
            "detail": entry.detail,
            "created_at": _utc_iso(entry.created_at),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


async def record_audit(
    session: AsyncSession,
    action: str,
    user_id: uuid.UUID | None = None,
    resource: str = "",
    detail: dict[str, Any] | None = None,
) -> None:
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _CHAIN_LOCK_KEY})
    # Chain order is the database's own sequence; the verifier walks the same
    # column. Ordering by (created_at, id) forked the chain on a pristine log —
    # a Python clock is not monotonic and a uuid4 is not a write-order tiebreak.
    previous = await session.scalar(select(AuditLog).order_by(AuditLog.seq.desc()).limit(1))
    entry = AuditLog(
        id=uuid.uuid4(),
        user_id=user_id,
        action=action,
        resource=resource,
        detail=detail or {},
        created_at=utcnow(),
        prev_hash=previous.entry_hash if previous else "",
    )
    entry.entry_hash = chain_hash(entry)
    session.add(entry)
    await session.commit()
