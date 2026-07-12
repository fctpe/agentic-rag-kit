from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.tables import Role, User
from app.retrieval.hybrid import hybrid_search, to_citation_dicts
from app.security.audit import record_audit
from app.security.rbac import require_role

router = APIRouter(tags=["search"])


@router.get("/search")
async def search(
    q: str = Query(min_length=2, max_length=500),
    regulation: str | None = Query(default=None, pattern="^(ai_act|gdpr)$"),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(require_role(Role.analyst)),
) -> dict:
    chunks = await hybrid_search(session, q, regulation=regulation)
    await record_audit(
        session, "search", user_id=user.id, detail={"q": q, "regulation": regulation}
    )
    return {"query": q, "results": to_citation_dicts(chunks)}
