"""Chat over SSE, with resumable human-in-the-loop approvals.

Events emitted:
  token              {"text": "..."}                 model output as it streams (qa only)
  drafting           {"reason": "awaiting_approval"}  report run; tokens suppressed
  approval_required  {"draft": ..., "citations": [], "citation_issues": []}
  citations          {"citations": [...]}
  grounding          {"grounded": bool, "issues": []}
  done               {"thread_id": ..., "content": ..., "citation_issues": []}
  error              {"message": ..., "request_id": ...}   no exception text, ever

Ordering guarantee, stated exactly: a *report* never reaches the user before
its approval decision — no tokens are streamed for one, and the draft is
carried in `approval_required`. A *qa* answer does stream, and its grounding
verdict lands afterwards, so until `grounding` arrives the answer is unverified
and the UI says so. "Verified before it is shown" holds for reports; for qa the
honest claim is "shown unverified, then verified before the run completes".

`citation_issues` rides on the two events that carry the finished text, and on
nothing else: they describe that text, so they must not arrive without it. No
event was added and no ordering changed. What DID change for qa: the tokens on
screen are the model's raw output, and `done.content` is the resolved answer the
resolve_citations node checkpointed. The frontend already replaces the bubble
with `done.content`, so a bracket the model merged is visible as raw text for
the window between the last token and `done`, then corrected. Buffering qa
tokens to hide that flicker would delete the streaming guarantee stated above
and the "Checking sources…" window the UI is built on, which is a worse trade.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agent.graph import RECURSION_LIMIT, build_graph
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
from app.observability import (
    GEN_AI_AGENT_NAME,
    GEN_AI_CONVERSATION_ID,
    GEN_AI_OPERATION_NAME,
    GROUNDED,
    REQUEST_ID,
    SERVICE,
    TOKENS_USED,
    error_fields,
    get_request_id,
    get_trace_callbacks,
    trace_metadata,
    tracer,
)
from app.security.audit import record_audit
from app.security.rbac import may_decide_approval, require_role
from app.security.redaction import redact_pii

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


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
    # Root span for the run: every node, tool, retrieval and grounding span
    # nests under it, and ragkit.request.id joins the trace to the log lines.
    with tracer.start_as_current_span(
        f"invoke_agent {SERVICE}",
        attributes={
            GEN_AI_OPERATION_NAME: "invoke_agent",
            GEN_AI_AGENT_NAME: SERVICE,
            GEN_AI_CONVERSATION_ID: thread_id,
            REQUEST_ID: get_request_id(),
        },
    ) as run_span:
        async with factory() as session:
            collector = CitationCollector()
            graph = build_graph(session, collector, checkpointer=checkpointer)
            config: dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
                "callbacks": get_trace_callbacks(),
                "metadata": trace_metadata(str(user.id), thread_id),
                "recursion_limit": RECURSION_LIMIT,
            }

            # Report-type runs do not stream. The router node settles task_type
            # before the agent node produces its first token, so this is known
            # in time to suppress them.
            #
            # It used to stream the draft, and the frontend deleted the bubble
            # when `approval_required` arrived — so the unapproved draft was
            # rendered, then withdrawn. Harmless while author and reviewer were
            # the same person; not harmless now that an admin can decide someone
            # else's report, and never what ADR 0001 said the gate did. A report
            # reaches the user in the approval payload or not at all.
            streaming_suppressed = False
            try:
                # Seed the source-id space from what this thread already
                # retrieved. The collector is per-request, so without this every
                # turn restarted at [1] while the model still had the previous
                # turn's answer — and its [1] — in history. Reusing an earlier
                # marker then produced a working button pointing at a different
                # regulation, silently, with grounded=true. The collector is
                # mutable and the tools close over it, so seeding after
                # build_graph is enough. Inside the try because a checkpointer
                # that cannot be read is a failed run, not a fresh thread.
                prior = await graph.aget_state(config)
                collector.seed(prior.values.get("retrieved_sources", []))
                async for mode, payload in graph.astream(
                    graph_input, config, stream_mode=["messages", "updates"]
                ):
                    if mode == "messages":
                        chunk, metadata = payload
                        if (
                            isinstance(chunk, AIMessageChunk)
                            and chunk.content
                            and metadata.get("langgraph_node") == "agent"
                            and not streaming_suppressed
                        ):
                            yield _sse("token", {"text": str(chunk.content)})
                    elif mode == "updates":
                        router_update = payload.get("router")
                        if router_update and router_update.get("task_type") == "report":
                            streaming_suppressed = True
                            yield _sse("drafting", {"reason": "awaiting_approval"})
                        if "__interrupt__" in payload:
                            interrupt_value = payload["__interrupt__"][0].value
                            yield _sse("approval_required", interrupt_value)
                        verify_update = payload.get("verify")
                        if verify_update:
                            yield _sse(
                                "citations", {"citations": verify_update.get("citations", [])}
                            )
                            yield _sse(
                                "grounding",
                                {
                                    # Fail closed everywhere this value is read:
                                    # a run that did not verify is not grounded,
                                    # and the default is the last place that can
                                    # still quietly say otherwise.
                                    "grounded": verify_update.get("grounded", False),
                                    "issues": verify_update.get("grounding_issues", []),
                                },
                            )
            except Exception as err:
                # The SSE error event was all anyone ever saw; the failure went
                # nowhere. Type and frames only — str(err) is where a DBAPIError
                # keeps the bound query text (error_fields, ADR 0006).
                failure = error_fields(err)
                logger.error(
                    "agent run failed", extra={"fields": {"thread_id": thread_id, **failure}}
                )
                run_span.add_event("exception", failure)
                run_span.set_status(Status(StatusCode.ERROR, failure["exception.type"]))
                # The same `str(err)` the log line goes out of its way to omit
                # was interpolated straight into the response body, so the bound
                # query text ADR 0006 keeps out of the logs went to the browser
                # instead. The request id is the join: it is on the root span
                # and on every log line for this run, so an operator can find
                # the failure from what the user quotes without the user ever
                # holding it.
                yield _sse(
                    "error",
                    {
                        "message": "Agent run failed. Quote this request id when reporting it.",
                        "request_id": get_request_id(),
                    },
                )
                return

            snapshot = await graph.aget_state(config)
            values = snapshot.values
            pending_approval = any(task.interrupts for task in snapshot.tasks)
            run_span.set_attribute(TOKENS_USED, values.get("tokens_used", 0))
            run_span.set_attribute(GROUNDED, values.get("grounded", False))

            if pending_approval:
                interrupt_payload = snapshot.tasks[0].interrupts[0].value
                session.add(Approval(thread_id=thread_id, payload=interrupt_payload))
                await session.commit()
                await record_audit(
                    session, "approval.requested", user_id=user.id, resource=thread_id
                )
                logger.info("run paused for approval", extra={"fields": {"thread_id": thread_id}})
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
                        "grounded": values.get("grounded", False),
                        "pii_found": values.get("pii_found", []),
                        "injection_signals": values.get("injection_signals", []),
                    },
                )
                logger.info(
                    "run answered",
                    extra={
                        "fields": {
                            "thread_id": thread_id,
                            "grounded": values.get("grounded", False),
                            "tokens_used": values.get("tokens_used", 0),
                        }
                    },
                )
                yield _sse(
                    "done",
                    {
                        "thread_id": thread_id,
                        "content": final,
                        # Read from state, not from an update seen during this
                        # stream: on the report path resolve_citations ran in
                        # the request BEFORE this one, so the resume request
                        # never sees its update. State is where the answer's
                        # citation defects actually live.
                        "citation_issues": values.get("citation_issues", []),
                    },
                )


@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    user: User = Depends(require_role(Role.viewer)),
) -> StreamingResponse:
    thread_id = body.thread_id or uuid.uuid4().hex
    # Redact ONCE, here, before the text is handed to anything that persists it.
    #
    # This used to redact only `stored_content` and pass `body.message` raw into
    # the graph, on the theory that guard_input redacts before any node sees it.
    # It does — but the checkpointer writes the *input* super-step before the
    # first node runs, so the raw message landed in `checkpoints` at steps -1
    # and 0 and only became redacted at step 1. docs/security.md promises
    # redaction "before the model, the checkpointer, or a trace"; two of those
    # three held. The graph input is the boundary, so redaction belongs here.
    # tests/test_security.py::TestRedactionBeforeCheckpoint walks the whole
    # checkpoint history for a canary.
    redaction = redact_pii(body.message)
    stored_content = redaction.text
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

    # `pii_found` rides along because guard_input can no longer discover it:
    # the text it receives is already clean. It still re-runs redact_pii as
    # defence in depth (idempotent — the placeholders carry no digits), so the
    # node keeps its guarantee for any caller that drives the graph directly.
    graph_input = {
        "messages": [HumanMessage(content=stored_content)],
        "pii_found": redaction.found,
    }
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
        # check entirely, so an unknown thread_id was resumable by any analyst;
        # `may_decide_approval` treats absence as denial.
        #
        # It is also what /admin/approvals filters by, so the queue an admin is
        # shown is exactly the queue an admin can act on. Those two had drifted:
        # the listing was admin-wide, the decision was owner-only, and every
        # cross-user item in an admin's queue returned 403.
        if not may_decide_approval(conversation.user_id if conversation else None, user):
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
