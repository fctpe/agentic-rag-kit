"""Liveness and readiness, which are not the same question.

`/health/live` asks whether this process is still serving its event loop, and
touches nothing else. Kubernetes *restarts* a container that fails it, so
checking Postgres here would turn a single failover into a rolling restart of
every replica at once — at the exact moment the pods are fine and the database
is not. It stays a process check on purpose.

`/health` asks whether this pod should receive traffic, and for this app that
means Postgres is reachable: every route past `/auth` reads or writes it, and
the checkpointer that lets a pending approval survive a restart lives there
too. It fails closed with 503, so the pod leaves the Service endpoints and
comes back on its own once the database does. It used to return
`{"status": "ok"}` without opening a connection, which made any readiness probe
pointed at it a test that the HTTP server could answer itself.

Both stay unauthenticated, as the single endpoint they replace was — a probe
cannot hold a token. The readiness query runs on the shared pool, so hammering
it costs a pooled `SELECT 1` rather than a new connection per request, and the
proxy-level rate limiting the production checklist already asks for covers the
rest.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.observability import error_fields

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

# Deliberately under the probe's own timeoutSeconds (3, in
# deploy/base/backend-deployment.yaml): a hung socket should be reported as
# not-ready by this handler, with a log line naming the exception type, rather
# than cut off by the kubelet with nothing written down anywhere.
READINESS_TIMEOUT_SECONDS = 2.0


@router.get("/health/live")
async def live() -> dict:
    """Liveness: the process is up. No database, no dependencies — see module docstring."""
    return {"status": "ok"}


@router.get("/health")
async def ready(response: Response, session: AsyncSession = Depends(get_session)) -> dict:
    """Readiness: the process is up *and* Postgres answers."""
    try:
        await asyncio.wait_for(session.execute(text("SELECT 1")), READINESS_TIMEOUT_SECONDS)
    except Exception as err:
        # Type and frames only — str(err) is where a DBAPIError keeps the bound
        # query text (error_fields, ADR 0006). The type is echoed to the caller
        # because "ConnectionRefusedError" is what an operator needs off a
        # probe; the DSN it was refused on is not.
        failure = error_fields(err)
        logger.warning("readiness check failed", extra={"fields": failure})
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "database": failure["exception.type"]}
    return {"status": "ok", "database": "ok"}
