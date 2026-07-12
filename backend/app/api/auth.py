from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.tables import User
from app.security.audit import record_audit
from app.security.rbac import issue_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str
    email: str


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, session: AsyncSession = Depends(get_session)) -> LoginResponse:
    user = await session.scalar(select(User).where(User.email == body.email))
    if user is None or not verify_password(body.password, user.password_hash):
        await record_audit(session, "auth.login_failed", detail={"email": body.email})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    await record_audit(session, "auth.login", user_id=user.id)
    return LoginResponse(token=issue_token(user), role=user.role.value, email=user.email)
