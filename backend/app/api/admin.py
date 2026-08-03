from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.tables import Approval, ApprovalStatus, AuditLog, Conversation, Role, User
from app.security.rbac import require_role

router = APIRouter(tags=["admin"])


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
        query = query.join(
            Conversation, Conversation.thread_id == Approval.thread_id
        ).where(Conversation.user_id == user.id)
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
