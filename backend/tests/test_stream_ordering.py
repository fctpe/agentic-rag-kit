"""What the user is allowed to see, and when.

ADR 0001 said the approval gate stops report output "before anything reaches
the user". The SSE assembly did not implement that: it streamed the draft's
tokens from the agent node, and the frontend deleted the bubble once
`approval_required` arrived. The draft was rendered and then withdrawn.

This drives the real `_stream_graph` against a stub graph that replays the
event sequence LangGraph produces, because that assembly loop is where the
ordering lives. Both task types, and the negative control that proves the
report assertion is not passing because nothing streamed at all.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

os.environ.setdefault("JWT_SECRET", "test-secret-long-enough-to-pass-startup-checks")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from app.api import chat as chat_api  # noqa: E402
from app.observability import request_context  # noqa: E402

DRAFT = "CONFIDENTIAL DRAFT: prohibited practices gap analysis"


class _Snapshot:
    def __init__(self, values: dict[str, Any], tasks: list[Any]):
        self.values = values
        self.tasks = tasks


class _Task:
    def __init__(self, interrupts: list[Any]):
        self.interrupts = interrupts


class _Interrupt:
    def __init__(self, value: dict[str, Any]):
        self.value = value


class _StubGraph:
    """Replays the (mode, payload) sequence LangGraph emits for one run."""

    def __init__(self, events: list[tuple[str, Any]], snapshot: _Snapshot):
        self._events = events
        self._snapshot = snapshot

    async def astream(self, *_args: Any, **_kwargs: Any):
        for event in self._events:
            yield event

    async def aget_state(self, *_args: Any, **_kwargs: Any) -> _Snapshot:
        return self._snapshot


def _chunk(text: str) -> tuple[str, tuple[AIMessageChunk, dict[str, str]]]:
    return "messages", (AIMessageChunk(content=text), {"langgraph_node": "agent"})


def _report_events() -> list[tuple[str, Any]]:
    return [
        ("updates", {"guard_input": {"pii_found": []}}),
        ("updates", {"router": {"task_type": "report"}}),
        *[_chunk(part) for part in DRAFT.split(" ")],
        (
            "updates",
            {"__interrupt__": (_Interrupt({"type": "approval_required", "draft": DRAFT}),)},
        ),
    ]


def _qa_events() -> list[tuple[str, Any]]:
    return [
        ("updates", {"guard_input": {"pii_found": []}}),
        ("updates", {"router": {"task_type": "qa"}}),
        *[_chunk(part) for part in ["Article", "6", "sets", "the", "conditions."]],
        ("updates", {"verify": {"citations": [], "grounded": True, "grounding_issues": []}}),
    ]


@pytest.fixture
def collect(monkeypatch: pytest.MonkeyPatch):
    """Runs `_stream_graph` over a stub graph and returns the raw SSE frames."""

    @asynccontextmanager
    async def _session():
        class _S:
            def add(self, _obj: Any) -> None: ...

            async def commit(self) -> None: ...

            async def scalar(self, *_a: Any, **_k: Any) -> None:
                # No Conversation row: the persistence branch is skipped, which
                # is fine — these tests are about what the stream emits.
                return None

        yield _S()

    def _factory():
        return _session()

    async def _no_audit(*_args: Any, **_kwargs: Any) -> None: ...

    monkeypatch.setattr(chat_api, "get_session_factory", lambda: _factory)
    monkeypatch.setattr(chat_api, "record_audit", _no_audit)

    async def _run(events: list[tuple[str, Any]], snapshot: _Snapshot) -> list[str]:
        monkeypatch.setattr(chat_api, "build_graph", lambda *a, **k: _StubGraph(events, snapshot))

        class _User:
            id = "00000000-0000-0000-0000-000000000001"

        return [frame async for frame in chat_api._stream_graph({}, "thread-1", _User(), None)]

    return _run


def _events_named(frames: list[str], name: str) -> list[str]:
    return [frame for frame in frames if frame.startswith(f"event: {name}\n")]


async def test_a_qa_answer_streams(collect):
    """Negative control. Without it, the report assertion below is satisfied by
    a stream loop that emits nothing at all."""
    frames = await collect(
        _qa_events(),
        _Snapshot({"messages": [AIMessage(content="Article 6 sets the conditions.")]}, []),
    )
    assert _events_named(frames, "token"), "qa runs must still stream"
    assert _events_named(frames, "grounding")


async def test_a_report_streams_no_tokens_before_approval(collect):
    frames = await collect(
        _report_events(),
        _Snapshot({}, [_Task([_Interrupt({"type": "approval_required", "draft": DRAFT})])]),
    )
    assert not _events_named(frames, "token"), (
        "an unapproved draft reached the user as streamed tokens"
    )
    assert _events_named(frames, "approval_required")


async def test_no_draft_text_appears_before_the_approval_frame(collect):
    """Stronger than counting token events: no frame emitted before
    `approval_required` may contain any of the draft's words."""
    frames = await collect(
        _report_events(),
        _Snapshot({}, [_Task([_Interrupt({"type": "approval_required", "draft": DRAFT})])]),
    )
    approval_at = next(
        i for i, frame in enumerate(frames) if frame.startswith("event: approval_required\n")
    )
    before = "".join(frames[:approval_at])
    for word in ("CONFIDENTIAL", "prohibited", "gap"):
        assert word not in before, f"draft word {word!r} leaked before the approval gate"


async def test_the_user_is_told_a_report_is_being_drafted(collect):
    """Suppressing tokens must not leave the UI on 'Thinking…' with no
    explanation — silence is how a suppressed stream reads as a hang."""
    frames = await collect(
        _report_events(),
        _Snapshot({}, [_Task([_Interrupt({"type": "approval_required", "draft": DRAFT})])]),
    )
    drafting = _events_named(frames, "drafting")
    assert drafting and "awaiting_approval" in drafting[0]


class _ExplodingGraph:
    """Fails partway through the stream, the way a dropped connection does."""

    def __init__(self, err: Exception):
        self._err = err

    async def astream(self, *_args: Any, **_kwargs: Any):
        yield ("updates", {"router": {"task_type": "qa"}})
        raise self._err

    async def aget_state(self, *_args: Any, **_kwargs: Any):
        raise AssertionError("the run failed; state must not be read")


# The bound query text a DBAPIError carries in `str(err)` — the exact thing
# error_fields exists to keep out of the logs (ADR 0006).
LEAKY = (
    "(psycopg.errors.UndefinedColumn) column chunks.embedding does not exist\n"
    "[SQL: SELECT chunks.content FROM chunks WHERE chunks.tenant = 'acme-legal' "
    "AND chunks.author_email = 'priya.shah@example.invalid']"
)


async def test_a_failed_run_does_not_ship_the_exception_to_the_browser(collect, monkeypatch):
    """The log line goes out of its way to omit `str(err)`, then the SSE error
    event interpolated it. Everything ADR 0006 keeps out of the logs — bound
    query text, and any personal data inside it — went to the client instead."""
    monkeypatch.setattr(
        chat_api, "build_graph", lambda *a, **k: _ExplodingGraph(RuntimeError(LEAKY))
    )

    class _User:
        id = "00000000-0000-0000-0000-000000000001"

    frames = [frame async for frame in chat_api._stream_graph({}, "thread-1", _User(), None)]
    body = "".join(frames)

    errors = _events_named(frames, "error")
    assert len(errors) == 1
    # Assert on the whole stream, not just the error frame: the point is that
    # this text is nowhere on the wire.
    assert "psycopg" not in body
    assert "priya.shah@example.invalid" not in body
    assert "acme-legal" not in body
    assert "SELECT" not in body
    assert "RuntimeError" not in body


async def test_the_failure_is_still_findable_by_request_id(collect, monkeypatch):
    """Control for the test above. Redacting the error into a blank message
    would satisfy every assertion there while leaving the user with nothing to
    quote and the operator with nothing to search."""
    monkeypatch.setattr(
        chat_api, "build_graph", lambda *a, **k: _ExplodingGraph(RuntimeError(LEAKY))
    )

    class _User:
        id = "00000000-0000-0000-0000-000000000001"

    with request_context("req-abc123") as request_id:
        frames = [frame async for frame in chat_api._stream_graph({}, "thread-1", _User(), None)]

    payload = json.loads(_events_named(frames, "error")[0].split("data: ", 1)[1])
    assert payload["request_id"] == request_id
    assert payload["message"]
