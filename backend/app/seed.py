"""Seed demo users (one per role). Passwords come from SEED_PASSWORD or
default to 'demo1234' — fine for local demos, rotate for anything shared.

    uv run python -m app.seed
"""

import asyncio
import os

from sqlalchemy import select

from app.db import dispose_engine, get_session_factory
from app.models.tables import Role, User
from app.security.rbac import hash_password

DEMO_USERS = [
    ("viewer@demo.local", Role.viewer),
    ("analyst@demo.local", Role.analyst),
    ("admin@demo.local", Role.admin),
]


async def main() -> None:
    password = os.environ.get("SEED_PASSWORD", "demo1234")
    factory = get_session_factory()
    async with factory() as session:
        for email, role in DEMO_USERS:
            existing = await session.scalar(select(User).where(User.email == email))
            if existing is None:
                session.add(User(email=email, password_hash=hash_password(password), role=role))
                print(f"created {email} ({role.value})")
            else:
                print(f"exists  {email} ({existing.role.value})")
        await session.commit()
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
