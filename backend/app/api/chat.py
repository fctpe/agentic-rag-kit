"""Chat over SSE, with resumable human-in-the-loop approvals.

Events emitted:
  token              {"text": "..."}                 model output as it streams
  approval_required  {"draft": ..., "citations": []} graph interrupted at the gate
  citations          {"citations": [...]}
  grounding          {"grounded": bool, "issues": []}
  done               {"thread_id": ..., "content": ...}
  error              {"message": ...}
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agent.graph import build_graph
from app.agent.tools import CitationCollector
from app.db import get_session_factory
from app.models.tables import (
    Approval,
    ApprovalStatus,
    Conversation,
    Message,
    Role,
    User,
    utcnow,
)
from app.observability import get_trace_callbacks, trace_metadata
from app.security.audit import record_audit
from app.security.rbac import require_role
from app.security.redaction import redact_pii

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    status: str = Field(pattern="^(approved|rejected)$")
    comment: str = Field(default="", max_length=2000)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_graph(
    graph_input: Any,
    thread_id: str,
    user: User,
    checkpointer: Any,
) -> AsyncIterator[str]:
    factory = get_session_factory()
    async with factory() as session:
        collector = CitationCollector()
        graph = build_graph(session, collector, checkpointer=checkpointer)
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "callbacks": get_trace_callbacks(),
            "metadata": trace_metadata(str(user.id), thread_id),
        }

        try:
            async for mode, payload in graph.astream(
                graph_input, config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, metadata = payload
                    if (
                        isinstance(chunk, AIMessageChunk)
                        and chunk.content
                        and metadata.get("langgraph_node") == "agent"
                    ):
                        yield _sse("token", {"text": str(chunk.content)})
                elif mode == "updates":
                    if "__interrupt__" in payload:
                        interrupt_value = payload["__interrupt__"][0].value
                        yield _sse("approval_required", interrupt_value)
                    verify_update = payload.get("verify")
                    if verify_update:
                        yield _sse("citations", {"citations": verify_update.get("citations", [])})
                        yield _sse(
                            "grounding",
                            {
                                "grounded": verify_update.get("grounded", True),
                                "issues": verify_update.get("grounding_issues", []),
                            },
                        )
        except Exception as err:
            yield _sse("error", {"message": f"Agent run failed: {err}"})
            return

        snapshot = await graph.aget_state(config)
        values = snapshot.values
        pending_approval = any(task.interrupts for task in snapshot.tasks)

        if pending_approval:
            interrupt_payload = snapshot.tasks[0].interrupts[0].value
            session.add(Approval(thread_id=thread_id, payload=interrupt_payload))
            await session.commit()
            await record_audit(session, "approval.requested", user_id=user.id, resource=thread_id)
        else:
            messages = values.get("messages", [])
            final = next(
                (
                    str(message.content)
                    for message in reversed(messages)
                    if isinstance(message, AIMessage) and message.content
                ),
                "",
            )
            conversation = await session.scalar(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )
            if conversation:
                session.add(
                    Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=final,
                        citations=values.get("citations", []),
                    )
                )
                await session.commit()
            await record_audit(
                session,
                "chat.answered",
                user_id=user.id,
                resource=thread_id,
                detail={
                    "grounded": values.get("grounded", True),
                    "pii_found": values.get("pii_found", []),
                    "injection_signals": values.get("injection_signals", []),
                },
            )
            yield _sse("done", {"thread_id": thread_id, "content": final})


@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    user: User = Depends(require_role(Role.viewer)),
) -> StreamingResponse:
    thread_id = body.thread_id or uuid.uuid4().hex
    # Redact before the raw message is persisted anywhere — the same guarantee
    # the agent graph makes, applied at the DB boundary too.
    stored_content = redact_pii(body.message).text
    factory = get_session_factory()
    async with factory() as session:
        conversation = await session.scalar(
            select(Conversation).where(Conversation.thread_id == thread_id)
        )
        if conversation is None:
            conversation = Conversation(
                user_id=user.id, thread_id=thread_id, title=stored_content[:120]
            )
            session.add(conversation)
        elif conversation.user_id != user.id:
            # Object-level ownership: a valid thread_id belonging to someone
            # else must not be readable or extendable.
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your conversation")
        await session.flush()
        session.add(Message(conversation_id=conversation.id, role="user", content=stored_content))
        await session.commit()
        await record_audit(session, "chat.query", user_id=user.id, resource=thread_id)

    graph_input = {"messages": [HumanMessage(content=body.message)]}
    return StreamingResponse(
        _stream_graph(graph_input, thread_id, user, request.app.state.checkpointer),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Thread-Id": thread_id},
    )


@router.post("/chat/{thread_id}/resume")
async def resume(
    thread_id: str,
    body: ResumeRequest,
    request: Request,
    user: User = Depends(require_role(Role.analyst)),
) -> StreamingResponse:
    factory = get_session_factory()
    async with factory() as session:
        conversation = await session.scalar(
            select(Conversation).where(Conversation.thread_id == thread_id)
        )
        # Fail closed. A missing Conversation row used to skip the ownership
        # check entirely, so an unknown thread_id was resumable by any analyst.
        # Every resumable thread has a Conversation — /chat creates it before the
        # graph ever runs — so absence means the thread is not this user's.
        if conversation is None or conversation.user_id != user.id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your conversation")
        approval = await session.scalar(
            select(Approval)
            .where(Approval.thread_id == thread_id)
            .where(Approval.status == ApprovalStatus.pending)
            .order_by(Approval.requested_at.desc())
        )
        if approval:
            approval.status = (
                ApprovalStatus.approved if body.status == "approved" else ApprovalStatus.rejected
            )
            approval.decided_at = utcnow()
            approval.decided_by = user.id
            await session.commit()
        await record_audit(
            session,
            "approval.decided",
            user_id=user.id,
            resource=thread_id,
            detail={"status": body.status, "comment": body.comment},
        )

    command: Command = Command(resume={"status": body.status, "comment": body.comment})
    return StreamingResponse(
        _stream_graph(command, thread_id, user, request.app.state.checkpointer),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
