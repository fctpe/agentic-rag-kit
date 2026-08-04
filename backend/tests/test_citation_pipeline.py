"""The plumbing that makes the resolved answer the ONLY answer.

Two properties nothing else in the suite covers:

1. `resolve_citations` writes back into the checkpointed message, so the draft
   the reviewer approves, the text the grounding verifier judges, and the string
   /chat persists are all one string. Normalising in verify() alone would have
   approved text A and shipped text B — the failure the approval gate exists to
   prevent.
2. Source ids are stable for a THREAD, not a request. The collector used to be
   rebuilt per HTTP request, so turn 2 re-issued [1] while the model still had
   turn 1's answer, and its [1], in history.

Everything here runs against a stub model and an in-memory checkpointer: no
provider, no database, no cost.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph
from app.agent.tools import CitationCollector
from app.retrieval.hybrid import RetrievedChunk
from tests.test_agent_budget import StubModel

MERGED_ANSWER = (
    "Data protection by design applies from the moment of determining the means "
    "[1, Art. 25(1)]. It also requires default minimisation [1(a)]."
)
RESOLVED_ANSWER = (
    "Data protection by design applies from the moment of determining the means "
    "(Art. 25(1)) [1]. It also requires default minimisation [1]."
)

SOURCES = [
    {
        "index": 1,
        "chunk_id": "chunk-a",
        "regulation": "gdpr",
        "document": "Regulation (EU) 2016/679",
        "article": "Art. 25",
        "heading": "Data protection by design and by default",
        "url": "https://eur-lex.europa.eu/...#art_25",
        "snippet": "Taking into account the state of the art…",
        "content": "Taking into account the state of the art, the controller shall…",
        "score": 0.0163,
    }
]


def _build(monkeypatch, model: StubModel, checkpointer: Any = None):
    def fake_init(name: str, **kwargs: Any) -> StubModel:
        return model

    monkeypatch.setattr("app.agent.graph.init_chat_model", fake_init)
    return build_graph(session=None, collector=CitationCollector(), checkpointer=checkpointer)


class TestTheAnswerInStateIsTheResolvedOne:
    async def test_the_merged_brackets_are_gone_from_the_checkpointed_message(
        self, monkeypatch
    ) -> None:
        graph = _build(monkeypatch, StubModel(total_tokens=1, answer=MERGED_ANSWER))
        state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What does data protection by design require?")],
                "retrieved_sources": SOURCES,
            }
        )
        assert state["messages"][-1].content == RESOLVED_ANSWER

    async def test_the_raw_answer_is_replaced_rather_than_appended(self, monkeypatch) -> None:
        # add_messages replaces by id. If the node ever produced a message with a
        # fresh id, the raw draft would still be in the transcript AND the
        # answer /chat returns would be picked by "last AIMessage with content".
        graph = _build(monkeypatch, StubModel(total_tokens=1, answer=MERGED_ANSWER))
        state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What does data protection by design require?")],
                "retrieved_sources": SOURCES,
            }
        )
        answers = [
            message
            for message in state["messages"]
            if isinstance(message, AIMessage) and message.content
        ]
        assert len(answers) == 1
        assert MERGED_ANSWER not in str(state["messages"])

    async def test_a_clean_answer_is_left_alone(self, monkeypatch) -> None:
        # Negative control: same node, same path, an answer with nothing to fix.
        clean = "Data protection by design applies from the outset (Art. 25(1)) [1]."
        graph = _build(monkeypatch, StubModel(total_tokens=1, answer=clean))
        state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What does data protection by design require?")],
                "retrieved_sources": SOURCES,
            }
        )
        assert state["messages"][-1].content == clean
        assert state.get("citation_issues") == []


class TestTheReviewerApprovesWhatTheUserGets:
    async def test_the_interrupt_payload_carries_the_resolved_draft(self, monkeypatch) -> None:
        graph = _build(
            monkeypatch,
            StubModel(total_tokens=1, answer=MERGED_ANSWER),
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "t-approval"}}
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Draft a report on data protection by design")],
                "retrieved_sources": SOURCES,
            },
            config,
        )
        snapshot = await graph.aget_state(config)
        payload = snapshot.tasks[0].interrupts[0].value

        assert payload["type"] == "approval_required"
        assert payload["draft"] == RESOLVED_ANSWER
        assert payload["citation_issues"] == []
        # And the same string is what a resume would persist, because it is the
        # message in state — not a second copy repaired somewhere downstream.
        assert snapshot.values["messages"][-1].content == payload["draft"]

    async def test_the_draft_really_took_the_report_path(self, monkeypatch) -> None:
        # Negative control for the test above: prove it interrupted at the gate
        # rather than running straight through as a qa answer, which would make
        # "payload == resolved" an assertion about a payload that never existed.
        graph = _build(
            monkeypatch,
            StubModel(total_tokens=1, answer=MERGED_ANSWER),
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "t-path"}}
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Draft a report on data protection by design")],
                "retrieved_sources": SOURCES,
            },
            config,
        )
        snapshot = await graph.aget_state(config)
        assert any(task.interrupts for task in snapshot.tasks)
        assert snapshot.values["task_type"] == "report"


class TestUnresolvableBracketsAreReportedNotHidden:
    async def test_an_index_that_does_not_exist_is_stripped_and_explained(
        self, monkeypatch
    ) -> None:
        graph = _build(
            monkeypatch,
            StubModel(total_tokens=1, answer="Fines reach EUR 35 000 000 [7]."),
        )
        state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What are the maximum fines?")],
                "retrieved_sources": SOURCES,
            }
        )
        assert state["messages"][-1].content == "Fines reach EUR 35 000 000."
        assert len(state["citation_issues"]) == 1
        assert "7" in state["citation_issues"][0]

    async def test_citation_issues_do_not_masquerade_as_grounding_issues(self, monkeypatch) -> None:
        # Separate channels on purpose: deployment.md tells operators to alert on
        # ragkit.grounded=false, and a bracket naming the wrong article is not an
        # unsupported claim. Folding them together would make that alert fire on
        # formatting.
        graph = _build(
            monkeypatch,
            StubModel(total_tokens=1, answer="Fines reach EUR 35 000 000 [7]."),
        )
        state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What are the maximum fines?")],
                "retrieved_sources": SOURCES,
            }
        )
        assert state["citation_issues"]
        assert state["citation_issues"][0] not in state.get("grounding_issues", [])


class TestSourceIdsAreStableForTheThread:
    """Turn 2 must not re-issue turn 1's numbers."""

    @staticmethod
    def _chunk(chunk_id: str, article: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            regulation="gdpr",
            document_title="Regulation (EU) 2016/679",
            source_url="https://eur-lex.europa.eu/...",
            article_ref=article,
            heading="…",
            content="…",
            score=1.0,
            vector_rank=None,
            text_rank=None,
        )

    def test_a_second_turn_continues_the_numbering(self) -> None:
        first = CitationCollector()
        first.numbered([self._chunk("a", "Art. 50"), self._chunk("b", "Art. 13")])
        exported = first.export_sources()
        assert [source["index"] for source in exported] == [1, 2]

        second = CitationCollector(seed=exported)
        second.numbered([self._chunk("c", "Art. 22")])
        indices = {source["chunk_id"]: source["index"] for source in second.export_sources()}
        assert indices == {"a": 1, "b": 2, "c": 3}

    def test_without_the_seed_the_new_chunk_steals_id_1(self) -> None:
        # Negative control: the exact defect the seed removes. Turn 2's GDPR
        # Art. 22 became [1] while turn 1's AI Act Art. 50 was also [1] in the
        # history the model was still reading from, and every marker it reused
        # rendered as a working button pointing at the wrong regulation.
        unseeded = CitationCollector()
        unseeded.numbered([self._chunk("c", "Art. 22")])
        assert unseeded.export_sources()[0]["index"] == 1

    def test_a_chunk_retrieved_again_keeps_its_original_id(self) -> None:
        first = CitationCollector()
        first.numbered([self._chunk("a", "Art. 50"), self._chunk("b", "Art. 13")])
        second = CitationCollector(seed=first.export_sources())
        assert second.numbered([self._chunk("a", "Art. 50")]).count('id="1"') == 1
        assert len(second.export_sources()) == 2  # no duplicate card

    def test_a_checkpoint_written_before_chunk_ids_still_reserves_its_numbers(self) -> None:
        # Threads that were live across the deploy have sources with no
        # chunk_id. Those ids must stay occupied, or a new chunk inherits a
        # number the model has already used for something else.
        legacy = [{key: value for key, value in SOURCES[0].items() if key != "chunk_id"}]
        collector = CitationCollector(seed=legacy)
        collector.numbered([self._chunk("new", "Art. 32")])
        assert [source["index"] for source in collector.export_sources()] == [1, 2]

    def test_the_chunk_id_never_reaches_the_ui_payload(self) -> None:
        from app.agent.tools import sources_to_citations

        collector = CitationCollector()
        collector.numbered([self._chunk("a", "Art. 50")])
        assert "chunk_id" in collector.export_sources()[0]
        assert "chunk_id" not in sources_to_citations(collector.export_sources())[0]
