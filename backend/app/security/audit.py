import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import AuditLog


async def record_audit(
    session: AsyncSession,
    action: str,
    user_id: uuid.UUID | None = None,
    resource: str = "",
    detail: dict[str, Any] | None = None,
) -> None:
    session.add(AuditLog(user_id=user_id, action=action, resource=resource, detail=detail or {}))
    await session.commit()
