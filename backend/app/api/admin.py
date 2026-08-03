from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.tables import Approval, ApprovalStatus, AuditLog, Conversation, Role, User
from app.security.audit import chain_hash
from app.security.rbac import require_role

router = APIRouter(tags=["admin"])

# Rows per /audit/verify query. The walk has to cover the whole chain, but
# deployment.md tells operators to poll this endpoint and the log only grows, so
# it may not hold the whole table at once.
VERIFY_PAGE_SIZE = 500


@router.get("/audit")
async def list_audit(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
    _admin: User = Depends(require_role(Role.admin)),
) -> dict:
    rows = await session.scalars(
        select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
    )
    return {
        "entries": [
            {
                "id": str(entry.id),
                "user_id": str(entry.user_id) if entry.user_id else None,
                "action": entry.action,
                "resource": entry.resource,
                "detail": entry.detail,
                "created_at": entry.created_at.isoformat(),
            }
            for entry in rows
        ]
    }


@router.get("/audit/verify")
async def verify_audit_chain(
    session: AsyncSession = Depends(get_session),
    _admin: User = Depends(require_role(Role.admin)),
) -> dict:
    """Walk the audit hash chain in write order and report the first break.

    An edited row fails its own hash; a row deleted or reordered *between* two
    others breaks its successor's link. Both are reported at the entry where the
    walk stops — everything after it is unverifiable, so there is nothing useful
    to say about it. The walk covers the whole chain; there is no incremental
    mode.

    Two things it does not prove, both stated in docs/security.md: deleting the
    newest rows leaves the remainder walking clean, and the hash is an unkeyed
    sha256 over columns anyone with UPDATE on this table can rewrite.
    """
    expected = ""
    checked = 0
    after = 0
    while True:
        page = list(
            await session.scalars(
                select(AuditLog)
                .where(AuditLog.seq > after)
                .order_by(AuditLog.seq)
                .limit(VERIFY_PAGE_SIZE)
            )
        )
        if not page:
            return {"ok": True, "checked": checked, "broken_at": None, "reason": None}
        for entry in page:
            reason = ""
            if entry.prev_hash != expected:
                reason = "prev_hash does not match the preceding entry"
            elif entry.entry_hash != chain_hash(entry):
                reason = "entry contents do not match entry_hash"
            if reason:
                return {
                    "ok": False,
                    "checked": checked,
                    "broken_at": str(entry.id),
                    "reason": reason,
                }
            expected = entry.entry_hash
            checked += 1
        after = page[-1].seq
        # Paging bounds the queries; expunging bounds the memory. Without this
        # the session's identity map still ends up holding every row walked.
        for entry in page:
            session.expunge(entry)


@router.get("/approvals")
async def list_pending_approvals(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(require_role(Role.analyst)),
) -> dict:
    # Object-level authorization, matching /chat/{thread_id}: an approval payload
    # is the draft report itself, so listing every pending approval to every
    # analyst leaked other users' drafts. Admins keep the full view.
    query = (
        select(Approval)
        .where(Approval.status == ApprovalStatus.pending)
        .order_by(Approval.requested_at.desc())
    )
    if user.role != Role.admin:
        query = query.join(Conversation, Conversation.thread_id == Approval.thread_id).where(
            Conversation.user_id == user.id
        )
    rows = await session.scalars(query)
    return {
        "pending": [
            {
                "id": str(approval.id),
                "thread_id": approval.thread_id,
                "payload": approval.payload,
                "requested_at": approval.requested_at.isoformat(),
            }
            for approval in rows
        ]
    }
