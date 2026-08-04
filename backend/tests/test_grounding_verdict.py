"""What the grounding verifier is allowed to say, and what happens to the answer.

Two findings, one node.

The parser was fail-open in every case except malformed JSON. It read
``bool(verdict.get("grounded", True))``, so a verdict that parsed but omitted the
field defaulted to grounded, and ``{"grounded": "false"}`` was a non-empty string
and therefore true. The runs most likely to be wrong — a truncated verdict, a
judge that answered in the wrong shape — were exactly the runs that passed. The
existing tests only covered the zero-sources path, which never reaches the
parser at all.

And an ungrounded verdict appended a warning ``AIMessage``. ``/chat`` takes the
last AI message as the answer, so the warning became the persisted answer and
the real one was dropped. RAGAS then scored the warning: G11 in the 2026-08-04
run came back with answer_relevancy 0.48 for text that was never an answer.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.graph import build_graph
from app.agent.tools import CitationCollector
from tests.test_agent_budget import StubModel

pytestmark = pytest.mark.anyio

ANSWER = "Art. 5 lists the prohibited practices."

SOURCE = {
    "index": 1,
    "regulation": "ai_act",
    "document": "AI Act",
    "article": "Art. 5",
    "heading": "Prohibited practices",
    "url": "https://example.invalid/art5",
    "snippet": "Prohibited AI practices.",
    "content": "The following AI practices shall be prohibited.",
    "score": 0.5,
}


class VerdictModel(StubModel):
    """Answers normally, then returns `verdict` verbatim to the grounding call.

    The grounding call is the only one that arrives as a single rendered prompt
    rather than a message list, so it is told apart by the prompt's own opening
    line instead of by call ordering — which a retry would shift.
    """

    verdict: str = '{"grounded": true, "issues": []}'

    def _generate(
        self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        rendered = " ".join(str(m.content) for m in messages)
        if "auditing an assistant answer" in rendered:
            message = AIMessage(
                content=self.verdict,
                usage_metadata={"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        return super()._generate(messages, stop, run_manager, **kwargs)


def _run(monkeypatch, verdict: str):
    model = VerdictModel(total_tokens=10, verdict=verdict)
    monkeypatch.setattr("app.agent.graph.init_chat_model", lambda name, **kwargs: model)
    graph = build_graph(session=None, collector=CitationCollector())
    # Seeding retrieved_sources is what puts the verifier on the path at all:
    # with no citations the node short-circuits before it ever parses a verdict.
    return graph.ainvoke(
        {
            "messages": [("user", "What does Art. 5 prohibit?")],
            "retrieved_sources": [SOURCE],
        }
    )


class TestAVerdictThatDoesNotSayGroundedIsNotGrounded:
    @pytest.mark.parametrize(
        ("label", "verdict"),
        [
            ("field missing entirely", '{"issues": []}'),
            ("empty object", "{}"),
            ("the string false", '{"grounded": "false", "issues": []}'),
            ("the string true", '{"grounded": "true", "issues": []}'),
            ("zero", '{"grounded": 0, "issues": []}'),
            ("one", '{"grounded": 1, "issues": []}'),
            ("null", '{"grounded": null, "issues": []}'),
            ("a list", '{"grounded": [], "issues": []}'),
            ("issues missing", '{"grounded": true}'),
            ("an unexpected key", '{"grounded": true, "issues": [], "confidence": 0.4}'),
            ("prose in braces", '{"the answer looks fine to me"}'),
            ("not json at all", "The answer is well grounded."),
            ("truncated", '{"grounded": tr'),
        ],
    )
    async def test_it_fails_closed(self, monkeypatch, label, verdict):
        state = await _run(monkeypatch, verdict)
        assert state["grounded"] is False, label
        assert state["grounding_issues"] == [
            "Grounding check could not be completed; treat this answer as unverified."
        ], label

    async def test_a_real_true_verdict_still_passes(self, monkeypatch):
        # Control. Every case above would also hold if the verifier had simply
        # started answering False to everything, which would be a worse bug than
        # the one they exist to catch: a governance layer nobody can satisfy
        # gets switched off.
        state = await _run(monkeypatch, '{"grounded": true, "issues": []}')
        assert state["grounded"] is True
        assert state["grounding_issues"] == []

    async def test_a_real_false_verdict_carries_its_issues(self, monkeypatch):
        state = await _run(
            monkeypatch, '{"grounded": false, "issues": ["Art. 6 is not in the sources"]}'
        )
        assert state["grounded"] is False
        assert state["grounding_issues"] == ["Art. 6 is not in the sources"]


class TestTheAnswerSurvivesItsOwnGroundingWarning:
    """`/chat` persists and returns the last AI message. Appending the warning
    there made the warning the answer, so the reader lost the text they asked
    for and the eval harness scored a sentence about an answer as the answer."""

    async def test_an_ungrounded_run_keeps_the_answer_it_produced(self, monkeypatch):
        state = await _run(
            monkeypatch, '{"grounded": false, "issues": ["Art. 6 is not in the sources"]}'
        )

        assert state["messages"][-1].content == ANSWER
        # The verdict is still reachable — it just travels as state, the way the
        # zero-sources branch already did it, rather than displacing the answer.
        assert state["grounded"] is False
        assert state["grounding_issues"] == ["Art. 6 is not in the sources"]

    async def test_an_unparseable_verdict_also_keeps_the_answer(self, monkeypatch):
        state = await _run(monkeypatch, "not json")
        assert state["messages"][-1].content == ANSWER

    async def test_no_message_says_the_words_the_warning_used_to(self, monkeypatch):
        # Pins the mechanism rather than just the last index: an ungrounded run
        # must not put the warning anywhere in the transcript, because anything
        # that reads messages back would pick it up again.
        state = await _run(monkeypatch, '{"grounded": false, "issues": ["unsupported"]}')
        assert not any("Grounding check flagged" in str(m.content) for m in state["messages"])
