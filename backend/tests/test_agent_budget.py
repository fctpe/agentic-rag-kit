"""Backpressure on the agent loop.

docs/deployment.md claims the app enforces token budgets; these tests hold it to
that. A run that blows MAX_TOTAL_TOKENS must stop with an honest message rather
than return a truncated answer, and it must leave no dangling tool_use behind —
the corruption the finalize node already exists to prevent.

A stub chat model drives the graph, so nothing here touches a provider or a
database: the tool node turns tool failures into tool messages, which is enough
to exercise the full loop.
"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError

from app.agent.graph import (
    LLM_TIMEOUT_SECONDS,
    MAX_QA_SUPERSTEPS,
    MAX_TOOL_ROUNDS,
    MAX_TOTAL_TOKENS,
    RECURSION_LIMIT,
    build_graph,
)
from app.agent.tools import CitationCollector
from app.retrieval import embedder


class StubModel(BaseChatModel):
    """Requests a tool call until `tool_calls_before_answer` is spent, then
    answers. Every reply reports `total_tokens` of usage."""

    total_tokens: int
    tool_calls_before_answer: int = 0
    answer: str = "Art. 5 lists the prohibited practices."
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls += 1
        tool_calls = (
            [{"name": "search_corpus", "args": {"query": "prohibited"}, "id": f"call_{self.calls}"}]
            if self.calls <= self.tool_calls_before_answer
            else []
        )
        message = AIMessage(
            content="" if tool_calls else self.answer,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": self.total_tokens,
                "output_tokens": 0,
                "total_tokens": self.total_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StubModel":
        return self


def _build(monkeypatch, model: StubModel) -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def fake_init(name: str, **kwargs: Any) -> StubModel:
        captured.update(kwargs)
        return model

    monkeypatch.setattr("app.agent.graph.init_chat_model", fake_init)
    return build_graph(session=None, collector=CitationCollector()), captured


async def _run(graph: Any, recursion_limit: int | None = None) -> dict[str, Any]:
    config = {"recursion_limit": recursion_limit} if recursion_limit else {}
    return await graph.ainvoke(
        {"messages": [HumanMessage(content="Which practices does Art. 5 prohibit?")]}, config
    )


async def test_over_budget_run_stops_with_an_honest_message(monkeypatch):
    graph, _ = _build(
        monkeypatch, StubModel(total_tokens=MAX_TOTAL_TOKENS, tool_calls_before_answer=99)
    )
    state = await _run(graph)

    final = state["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "token budget" in final.content
    assert state["tokens_used"] >= MAX_TOTAL_TOKENS
    # Nothing was verified, so the run must not report itself as grounded.
    assert state["grounded"] is False


async def test_over_budget_run_leaves_no_dangling_tool_call(monkeypatch):
    graph, _ = _build(
        monkeypatch, StubModel(total_tokens=MAX_TOTAL_TOKENS, tool_calls_before_answer=99)
    )
    state = await _run(graph)

    requested = {
        call["id"]
        for message in state["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    answered = {
        message.tool_call_id for message in state["messages"] if isinstance(message, ToolMessage)
    }
    assert requested
    assert requested == answered


async def test_within_budget_the_run_answers_normally(monkeypatch):
    graph, _ = _build(monkeypatch, StubModel(total_tokens=10))
    state = await _run(graph)

    assert state["messages"][-1].content == "Art. 5 lists the prohibited practices."
    # router + agent, both counted from usage_metadata.
    assert state["tokens_used"] == 20


async def test_recursion_limit_admits_a_maximal_tool_loop(monkeypatch):
    # The limit must bound runaway loops without cutting off a run that uses the
    # whole tool budget and then the finalize node.
    graph, _ = _build(
        monkeypatch,
        StubModel(total_tokens=1, tool_calls_before_answer=MAX_TOOL_ROUNDS + 2),
    )
    state = await _run(graph, recursion_limit=RECURSION_LIMIT)

    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= MAX_TOOL_ROUNDS
    assert state["messages"][-1].content == "Art. 5 lists the prohibited practices."


async def test_a_maximal_qa_run_needs_the_supersteps_claimed(monkeypatch):
    # MAX_QA_SUPERSTEPS is what RECURSION_LIMIT is built on, and until now no
    # command produced the number. Run the longest legitimate qa path at exactly
    # that limit, then one superstep short of it.
    def maximal() -> StubModel:
        return StubModel(total_tokens=1, tool_calls_before_answer=MAX_TOOL_ROUNDS + 2)

    graph, _ = _build(monkeypatch, maximal())
    state = await _run(graph, recursion_limit=MAX_QA_SUPERSTEPS)
    assert state["messages"][-1].content == "Art. 5 lists the prohibited practices."

    graph, _ = _build(monkeypatch, maximal())
    with pytest.raises(GraphRecursionError):
        await _run(graph, recursion_limit=MAX_QA_SUPERSTEPS - 1)


async def test_model_is_built_with_a_per_call_timeout(monkeypatch):
    _, kwargs = _build(monkeypatch, StubModel(total_tokens=1))
    assert kwargs["timeout"] == LLM_TIMEOUT_SECONDS


def test_the_embedder_is_built_with_a_per_call_timeout(monkeypatch):
    # The other half of the hung-call bound: the vector arm's embedding call runs
    # inside hybrid_search, inside the tool node, holding the request's session.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    monkeypatch.setattr(embedder, "_embedder", None)
    assert embedder.get_embedder().request_timeout == embedder.EMBEDDING_TIMEOUT_SECONDS
