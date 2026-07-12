"""Langfuse tracing, enabled only when keys are configured.

Traces carry the full graph execution (router, tool calls, grounding check)
plus model token usage. PII is redacted in guard_input BEFORE anything can
reach a trace — see docs/security.md.
"""

from typing import Any

from app.config import get_settings


def get_trace_callbacks(user_id: str, thread_id: str) -> list[Any]:
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return []
    return [CallbackHandler()]


def trace_metadata(user_id: str, thread_id: str) -> dict[str, Any]:
    return {
        "langfuse_user_id": user_id,
        "langfuse_session_id": thread_id,
    }
