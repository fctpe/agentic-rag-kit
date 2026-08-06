"""What may occupy the reviewer's approval queue.

The router settles `task_type` from the request, before the agent has run. So a
request phrased as a report but aimed outside the corpus — "write a compliance
report on NIS2" — is classified `report`, refused by the agent, and used to be
gated anyway. A demo instance accumulated four pending approvals of which three
were declines.

An approval queue that is mostly noise is clicked through, and then the Art. 14
human-oversight control the gate implements is decoration. So a report drafted
from zero retrieved sources takes the ordinary answer path.

The negative control is the point of this file: without it, "refusals do not
reach the gate" is satisfied just as well by a gate nothing ever reaches.
A stub model drives the graph; no provider, no database.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.agent.tools import CitationCollector

REFUSAL = "I'm sorry, but the NIS2 Directive is outside the corpus of the EU AI Act and the GDPR."
REPORT_REQUEST = "Write me a compliance report on the NIS2 Directive."

# Shape mirrors app.agent.tools._source_dict; only the keys the gate and the
# verifier read are needed here.
SOURCE = {
    "index": 1,
    "chunk_id": "c1",
    "regulation": "ai_act",
    "document": "EU AI Act",
    "article": "Art. 5",
    "heading": "Prohibited practices",
    "url": "https://example.invalid/#art_5",
    "snippet": "snippet",
    "content": "Art. 5 lists the prohibited practices.",
    "score": 0.9,
}


class StubModel(BaseChatModel):
    """Answers immediately, never calls a tool."""

    answer: str

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        message = AIMessage(
            content=self.answer,
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StubModel":
        return self


def _build(monkeypatch, answer: str):
    monkeypatch.setattr(
        "app.agent.graph.init_chat_model", lambda name, **kwargs: StubModel(answer=answer)
    )
    return build_graph(session=None, collector=CitationCollector(), checkpointer=MemorySaver())


async def _run(graph, thread: str, sources: list[dict[str, Any]] | None = None):
    state: dict[str, Any] = {"messages": [HumanMessage(content=REPORT_REQUEST)]}
    if sources is not None:
        state["retrieved_sources"] = sources
    return await graph.ainvoke(state, {"configurable": {"thread_id": thread}})


async def test_a_report_request_is_still_classified_as_a_report(monkeypatch):
    """The routing fix must not work by quietly reclassifying the request —
    that would take the streaming suppression down with it (ADR 0001)."""
    graph = _build(monkeypatch, REFUSAL)
    result = await _run(graph, "t-classify")

    assert result["task_type"] == "report"


async def test_a_report_with_no_sources_does_not_reach_the_approval_gate(monkeypatch):
    graph = _build(monkeypatch, REFUSAL)
    result = await _run(graph, "t-refusal")

    assert not result.get("retrieved_sources")
    assert "__interrupt__" not in result, "a refusal was queued for human approval"
    # It ran to completion down the ordinary path.
    assert result["messages"][-1].content == REFUSAL


async def test_the_skipped_gate_does_not_upgrade_the_verdict(monkeypatch):
    """Skipping the gate must not make a zero-source answer look checked.
    verify() owns that call and fails closed; this pins that the new route
    still goes through it."""
    graph = _build(monkeypatch, REFUSAL)
    result = await _run(graph, "t-verdict")

    assert result["grounded"] is False
    assert result["grounding_issues"]


async def test_a_report_with_sources_still_stops_at_the_approval_gate(monkeypatch):
    """Negative control. Without this, every assertion above is satisfied by a
    gate that no run reaches at all."""
    graph = _build(monkeypatch, "### Compliance report\n\nArt. 5 prohibits [1].")
    result = await _run(graph, "t-real-report", sources=[SOURCE])

    assert "__interrupt__" in result, "a sourced report bypassed human approval"
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "approval_required"
    assert payload["citations"], "the reviewer must see what the draft is built on"
