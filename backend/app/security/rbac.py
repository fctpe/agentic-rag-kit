import hashlib
import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models.tables import Role, User

_bearer = HTTPBearer(auto_error=False)

_ROLE_ORDER = {Role.viewer: 0, Role.analyst: 1, Role.admin: 2}


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return secrets.compare_digest(digest.hex(), digest_hex)


def issue_token(user: User) -> str:
    settings = get_settings()
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value,
        "exp": datetime.now(UTC) + timedelta(hours=settings.jwt_ttl_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> User:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        payload = jwt.decode(
            credentials.credentials, get_settings().jwt_secret, algorithms=["HS256"]
        )
    except jwt.PyJWTError as err:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from err

    user = await session.scalar(select(User).where(User.email == payload.get("email")))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
    return user


def require_role(minimum: Role):
    async def dependency(user: User = Depends(get_current_user)) -> User:
        if _ROLE_ORDER[user.role] < _ROLE_ORDER[minimum]:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires role {minimum.value} or higher",
            )
        return user

    return dependency


def may_decide_approval(conversation_owner_id, user: User) -> bool:
    """Whether `user` may approve or reject the pending approval on a thread
    owned by `conversation_owner_id`.

    The single rule behind two surfaces that had drifted apart: GET /admin/approvals
    listed every pending approval to an admin, while POST /chat/{id}/resume
    rejected any thread the caller did not own — so an admin's queue was a list
    of drafts nobody could act on. Anything the listing shows a user, this must
    return True for; tests/test_api_authz.py asserts exactly that correspondence.

    `None` owner means no Conversation row, which means the thread is not
    resumable at all — fail closed rather than treating absence as permission.
    """
    if conversation_owner_id is None:
        return False
    return user.role == Role.admin or conversation_owner_id == user.id
