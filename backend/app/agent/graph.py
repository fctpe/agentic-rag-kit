"""The agent graph: guard -> route -> agent/tools loop -> approval -> verify.

Explicit StateGraph rather than a prebuilt agent: the graph shape IS the
governance story — input guarding, a durable human approval interrupt for
report-type output (EU AI Act Art. 14 framing), and a grounding check before
anything reaches the user. Interrupts persist in the Postgres checkpointer,
so a pending approval survives a backend restart.

Every run is bounded three ways (ADR 0005): tool rounds, tokens spent, and
graph supersteps. Hitting a bound ends the run with a message that says so;
none of them truncates an answer and presents it as finished.
"""

import logging
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.markers import resolve_markers
from app.agent.prompts import GROUNDING_PROMPT, REFUSAL_MESSAGE, ROUTER_PROMPT, SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.tools import (
    CitationCollector,
    build_tools,
    sources_to_citations,
    sources_to_excerpts,
)
from app.config import get_settings
from app.observability import (
    GEN_AI_EVALUATION_NAME,
    GEN_AI_EVALUATION_SCORE_LABEL,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    error_fields,
    record_model_call,
    traced_node,
    tracer,
)
from app.security.injection import assess_injection
from app.security.redaction import redact_pii

logger = logging.getLogger(__name__)


class GroundingVerdict(BaseModel):
    """What the grounding verifier is allowed to have said.

    Every field is required and `grounded` is strict, so the model cannot omit
    it, answer `"false"`, `0`, or `null`, or bury it under an extra key. Each of
    those used to be read as grounded — `.get("grounded", True)` supplied the
    default, and `bool()` supplied the rest. A verifier confident enough to
    return well-formed JSON and vague enough to leave out the verdict is exactly
    the run worth failing on.

    `extra="forbid"` is deliberate rather than tidy: an unexpected key usually
    means the schema drifted or the model answered a different question, and
    either way the verdict is not the one this code thinks it is reading.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: StrictBool
    issues: list[str]


TOOL_BUDGET_MESSAGE = (
    "Tool-call budget reached. Answer the user now using only the sources already retrieved; "
    "do not request more tools."
)

TOKEN_BUDGET_STUB = "Not executed — the run reached its token budget."

TOKEN_BUDGET_MESSAGE = (
    "This run stopped at its token budget ({used:,} of {budget:,} tokens) before it produced an "
    "answer. No partial answer is being returned — a truncated compliance answer is worse than "
    "none. Ask a narrower question, or split it into steps."
)

NO_SOURCES_ISSUE = (
    "No sources were retrieved, so no grounding check ran — treat this answer as unverified."
)

MAX_TOOL_ROUNDS = 6

# MAX_TOOL_ROUNDS bounds how many model calls a run makes, not how large they
# get: read_article returns every chunk of an article, so a single round can be
# arbitrarily long. This is the cost ceiling, summed from usage_metadata over
# every model call in the request (router, agent turns, finalize, grounding).
MAX_TOTAL_TOKENS = 80_000

# The longest single invocation is a maximal qa run: start, guard, router,
# MAX_TOOL_ROUNDS × (agent, tools), a last agent turn, finalize,
# resolve_citations, verify. A report run splits at the approval interrupt and
# needs fewer. The count is measured, not derived on paper —
# tests/test_agent_budget.py::test_a_maximal_qa_run_needs_the_supersteps_claimed
# runs one at exactly this limit and one superstep short of it.
MAX_QA_SUPERSTEPS = 2 * MAX_TOOL_ROUNDS + 7

# A backstop, not a working limit: the measured need plus headroom. Exceeding it
# raises GraphRecursionError, which the chat route surfaces as an SSE error event
# rather than as a silently short answer.
RECURSION_LIMIT = MAX_QA_SUPERSTEPS + 2

# The OpenAI client retries twice by default but has no timeout at all, so one
# hung call would pin a request and its DB session open indefinitely.
LLM_TIMEOUT_SECONDS = 60

REPORT_MARKERS = ("report", "memo", "checklist", "gap analysis", "executive summary", "draft")


def _spend(state: AgentState, response: BaseMessage) -> int:
    """Running token total for the request. A provider that reports no
    usage_metadata contributes 0 — the budget then enforces nothing and the run
    is bounded by MAX_TOOL_ROUNDS and RECURSION_LIMIT alone."""
    usage = getattr(response, "usage_metadata", None)
    return state.get("tokens_used", 0) + (int(usage["total_tokens"]) if usage else 0)


def build_graph(session: AsyncSession, collector: CitationCollector, checkpointer: Any = None):
    settings = get_settings()
    model = init_chat_model(settings.llm_model, temperature=0, timeout=LLM_TIMEOUT_SECONDS)
    tools = build_tools(session, collector)
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    @traced_node("guard_input")
    async def guard_input(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        text = str(last.content)
        if len(text) > settings.max_input_chars:
            return {
                "refused": True,
                "messages": [AIMessage(content="Input too long — please shorten your request.")],
                "grounded": False,
            }
        redaction = redact_pii(text)
        injection = assess_injection(redaction.text)
        # /chat redacts at the API boundary (so nothing raw reaches the
        # checkpointer) and passes what it stripped in as `pii_found`. Redacting
        # again here finds nothing and would erase that record, so union the two:
        # this node stays a real boundary for any caller that drives the graph
        # directly, without dropping the labels the route already found.
        pii_found = list(dict.fromkeys([*state.get("pii_found", []), *redaction.found]))
        updates: dict[str, Any] = {
            "pii_found": pii_found,
            "injection_signals": injection.signals,
            # A new turn on an existing thread starts its own budget; only a
            # resumed approval (which re-enters after this node) keeps the total.
            "tokens_used": 0,
            # Nothing has been verified yet, and a run that ends before verify()
            # — refused here, rejected at the approval gate — must not report the
            # answer it never produced as grounded. deployment.md tells operators
            # to alert on ragkit.grounded=false; that alert has to see this.
            "grounded": False,
        }
        # Replace the raw message with the redacted text whenever PII was found,
        # on BOTH the refusal and the pass branch — otherwise a message that is
        # both PII-bearing and injection-flagged leaves raw PII in the checkpoint.
        if redaction.found:
            updates["messages"] = [HumanMessage(content=redaction.text, id=last.id)]
        if injection.flagged:
            logger.warning(
                "input refused: injection signals",
                extra={"fields": {"signals": injection.signals}},
            )
            updates["refused"] = True
            updates["messages"] = [
                *updates.get("messages", []),
                AIMessage(content=REFUSAL_MESSAGE),
            ]
        return updates

    def after_guard(state: AgentState) -> Literal["router", "__end__"]:
        return "__end__" if state.get("refused") else "router"

    @traced_node("router")
    async def router(state: AgentState) -> dict[str, Any]:
        query = str(state["messages"][-1].content)
        if any(marker in query.lower() for marker in REPORT_MARKERS):
            return {"task_type": "report"}
        response = await model.ainvoke(ROUTER_PROMPT.format(query=query[:2000]))
        record_model_call(response)
        verdict = str(response.content).strip().lower()
        return {
            "task_type": "report" if verdict == "report" else "qa",
            "tokens_used": _spend(state, response),
        }

    @traced_node("agent")
    async def agent(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await model_with_tools.ainvoke(messages)
        record_model_call(response)
        return {"messages": [response], "tokens_used": _spend(state, response)}

    def after_agent(
        state: AgentState,
    ) -> Literal["tools", "finalize", "budget_exceeded", "resolve_citations"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            # The token budget is checked before the tool-round budget because
            # finalize costs one more model call — the thing there is no budget
            # left for.
            if state.get("tokens_used", 0) >= MAX_TOTAL_TOKENS:
                return "budget_exceeded"
            tool_rounds = sum(
                1 for message in state["messages"] if isinstance(message, ToolMessage)
            )
            # Out of tool rounds but the model still wants tools: never leave a
            # dangling tool_use (it corrupts the next turn on the thread). Force
            # a clean tool-free answer via the finalize node instead.
            return "tools" if tool_rounds < MAX_TOOL_ROUNDS else "finalize"
        return "resolve_citations"

    @traced_node("tools")
    async def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        results: list[ToolMessage] = []
        for call in last.tool_calls:
            with tracer.start_as_current_span(
                f"execute_tool {call['name']}",
                attributes={
                    GEN_AI_OPERATION_NAME: "execute_tool",
                    GEN_AI_TOOL_NAME: call["name"],
                    GEN_AI_TOOL_CALL_ID: call["id"] or "",
                },
            ) as span:
                tool = tools_by_name.get(call["name"])
                if tool is None:
                    output = f"Unknown tool {call['name']}"
                    span.set_status(Status(StatusCode.ERROR, output))
                else:
                    try:
                        output = await tool.ainvoke(call["args"])
                    except Exception as err:  # surfaced to the model for self-correction
                        # Self-correction is the model's business; an operator
                        # still needs the failure, which used to go nowhere.
                        # Type and frames only — str(err) is where a DBAPIError
                        # keeps the bound query text (error_fields, ADR 0006).
                        failure = error_fields(err)
                        logger.error(
                            "tool failed", extra={"fields": {"tool": call["name"], **failure}}
                        )
                        span.add_event("exception", failure)
                        span.set_status(Status(StatusCode.ERROR, failure["exception.type"]))
                        output = f"Tool {call['name']} failed: {err}"
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        # Persist retrieved chunks into checkpointed state so verify() and the
        # approval interrupt can read them even on a resume that skips tools.
        return {"messages": results, "retrieved_sources": collector.export_sources()}

    @traced_node("finalize")
    async def finalize(state: AgentState) -> dict[str, Any]:
        """Tool-round budget exhausted with pending tool calls: satisfy the
        dangling tool_calls, then answer once with tools unbound so no invalid
        tool_use/tool_result pair is ever checkpointed."""
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        stubs = [
            ToolMessage(content=TOOL_BUDGET_MESSAGE, tool_call_id=call["id"])
            for call in last.tool_calls
        ]
        history = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"], *stubs]
        answer = await model.ainvoke(history)
        record_model_call(answer)
        return {"messages": [*stubs, answer], "tokens_used": _spend(state, answer)}

    @traced_node("budget_exceeded")
    async def budget_exceeded(state: AgentState) -> dict[str, Any]:
        """Token budget gone while tool calls are still pending. Same rule as
        finalize — satisfy the dangling tool_calls so nothing invalid is
        checkpointed — but spend no further model call and say plainly that the
        run stopped short."""
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        stubs = [
            ToolMessage(content=TOKEN_BUDGET_STUB, tool_call_id=call["id"])
            for call in last.tool_calls
        ]
        used = state.get("tokens_used", 0)
        logger.warning(
            "run stopped at its token budget",
            extra={"fields": {"tokens_used": used, "token_budget": MAX_TOTAL_TOKENS}},
        )
        stop = AIMessage(content=TOKEN_BUDGET_MESSAGE.format(used=used, budget=MAX_TOTAL_TOKENS))
        # Redundant with guard_input's seed, and stated anyway: this node exists
        # precisely because no answer was produced, so it owns the verdict.
        return {"messages": [*stubs, stop], "grounded": False}

    @traced_node("resolve_citations")
    async def resolve_citations(state: AgentState) -> dict[str, Any]:
        """Make every surviving [n] resolve to a retrieved source, or remove it.

        One node, on the single path every answer takes, and it writes the
        result back into the message. That placement is the whole point: the
        approval gate reads `messages[-1]`, verify() reads `messages[-1]`, and
        /chat persists the last AIMessage. Normalising in any one of them would
        have shipped the reviewer one string and the user another — the exact
        failure the approval gate exists to prevent. Rewriting the checkpointed
        message means all three reads see the same text.

        No model call: this is a deterministic string pass, so it costs nothing
        and cannot itself hallucinate a citation. Brackets it cannot resolve
        safely are left as literal text and reported, never guessed at.
        """
        last = state["messages"][-1]
        original = str(last.content)
        resolved = resolve_markers(
            original, sources_to_citations(state.get("retrieved_sources", []))
        )
        updates: dict[str, Any] = {"citation_issues": resolved.issues}
        if resolved.issues:
            logger.info(
                "citation markers could not all be linked",
                extra={"fields": {"citation_issues": len(resolved.issues)}},
            )
        if resolved.text != original:
            # Same id, so add_messages REPLACES rather than appends: the answer
            # stays the last AIMessage and the raw text is gone from state.
            updates["messages"] = [last.model_copy(update={"content": resolved.text})]
        return updates

    def after_resolve(state: AgentState) -> Literal["approval_gate", "verify"]:
        if state.get("task_type") != "report":
            return "verify"
        # task_type is settled by the router from the REQUEST, before the agent
        # has run — that ordering is what lets /chat suppress streaming for a
        # report (ADR 0001 addendum). The cost is that an out-of-corpus request
        # *phrased* as a report ("write a compliance report on NIS2") is
        # classified `report`, then correctly refused by the agent, and the
        # refusal was still gated. Observed on a demo instance: three of the
        # four entries waiting in the reviewer's queue were declines.
        #
        # That is not cosmetic. A queue that is mostly noise gets clicked
        # through, and the thing that erodes is the Art. 14 human-oversight
        # control the gate implements — the same reasoning that keeps
        # citation_issues out of the grounded=false alert.
        #
        # A draft built on zero retrieved sources is not a report: there is no
        # corpus text to review it against, and verify() already reports it as
        # ungrounded with NO_SOURCES_ISSUE. Send it down the ordinary answer
        # path instead of into the queue.
        #
        # Deliberately narrow. The predicate is "nothing was retrieved", not
        # "the answer looks like a refusal": an agent that searches, gets rows,
        # and then declines still stops at the gate. Every error this can make
        # is toward MORE review, never toward releasing an unreviewed report —
        # which is the only direction a governance control may fail in.
        if not state.get("retrieved_sources"):
            return "verify"
        return "approval_gate"

    @traced_node("approval_gate")
    async def approval_gate(state: AgentState) -> dict[str, Any]:
        draft = str(state["messages"][-1].content)
        decision = interrupt(
            {
                "type": "approval_required",
                "draft": draft,
                "citations": sources_to_citations(state.get("retrieved_sources", [])),
                # The reviewer approves the text the user will get, including
                # what could not be linked in it. Deciding on a draft whose
                # citation defects are invisible is deciding on a different
                # document.
                "citation_issues": state.get("citation_issues", []),
            }
        )
        status = decision.get("status", "rejected") if isinstance(decision, dict) else "rejected"
        comment = decision.get("comment", "") if isinstance(decision, dict) else ""
        updates: dict[str, Any] = {"approval_decision": status, "approval_comment": comment}
        if status != "approved":
            updates["messages"] = [
                AIMessage(
                    content=(
                        "The draft report was rejected by the reviewer"
                        + (f' with comment: "{comment}"' if comment else ".")
                    )
                )
            ]
            updates["refused"] = True
        return updates

    def after_approval(state: AgentState) -> Literal["verify", "__end__"]:
        return "__end__" if state.get("refused") else "verify"

    @traced_node("verify")
    async def verify(state: AgentState) -> dict[str, Any]:
        span = trace.get_current_span()
        answer = str(state["messages"][-1].content)
        # Read from checkpointed state, not the per-request collector, so a
        # resumed approval still has its citations and grounding context.
        retrieved = state.get("retrieved_sources", [])
        citations = sources_to_citations(retrieved)
        if not citations:
            # Nothing retrieved, so nothing was evaluated — and an answer built
            # on zero sources is the one most likely to be hallucinated. Saying
            # grounded here was a fail-open that hid exactly that run from the
            # ragkit.grounded=false alert. The span label is the third value the
            # boolean cannot carry: unverified is not the same as ungrounded.
            logger.warning("answer produced with no retrieved sources")
            span.set_attribute(GEN_AI_EVALUATION_NAME, "grounding")
            span.set_attribute(GEN_AI_EVALUATION_SCORE_LABEL, "unverified")
            # The issue reaches the user through the existing grounding event —
            # no extra message, which would displace the answer itself in the
            # persisted transcript.
            return {"citations": [], "grounded": False, "grounding_issues": [NO_SOURCES_ISSUE]}
        excerpts = sources_to_excerpts(retrieved)
        response = await model.ainvoke(
            GROUNDING_PROMPT.format(sources=excerpts[:60000], answer=answer[:8000])
        )
        record_model_call(response)
        span.set_attribute(GEN_AI_EVALUATION_NAME, "grounding")
        try:
            raw = str(response.content).strip().removeprefix("```json").removesuffix("```")
            verdict = GroundingVerdict.model_validate_json(raw)
            grounded, issues = verdict.grounded, verdict.issues
        except (ValidationError, TypeError) as err:
            # Fail CLOSED: if the verifier itself errored, we cannot claim the
            # answer is grounded. Surfacing this beats silently shipping it —
            # the whole point of a governance layer.
            #
            # Only malformed JSON used to reach this branch. The happy path read
            # `bool(verdict.get("grounded", True))`, so a verdict that parsed but
            # said nothing — `{}`, a truncated object, a judge that answered in
            # prose wrapped in braces — defaulted to grounded, and `"grounded":
            # "false"` was a non-empty string and therefore true. The two
            # verdicts most likely to be wrong were the two that passed.
            logger.warning(
                "grounding verifier returned an unusable verdict",
                extra={"fields": error_fields(err)},
            )
            span.set_status(Status(StatusCode.ERROR, "grounding verdict unusable"))
            grounded = False
            issues = ["Grounding check could not be completed; treat this answer as unverified."]
        span.set_attribute(GEN_AI_EVALUATION_SCORE_LABEL, "grounded" if grounded else "ungrounded")
        # No warning AIMessage. `grounded` and `grounding_issues` are already the
        # channel this reaches the caller on, and appending a second one made the
        # warning the last AI message — which is what /chat persists and returns
        # as the answer. The answer itself was dropped from the transcript and
        # replaced by a note about it, and RAGAS then scored the note: G11 came
        # back with answer_relevancy 0.48 for text that was never an answer.
        # The zero-source branch above already declines to do this, and says why.
        return {
            "citations": citations,
            "grounded": grounded,
            "grounding_issues": issues,
            "tokens_used": _spend(state, response),
        }

    builder = StateGraph(AgentState)
    builder.add_node("guard_input", guard_input)
    builder.add_node("router", router)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools_node)
    builder.add_node("finalize", finalize)
    builder.add_node("budget_exceeded", budget_exceeded)
    builder.add_node("resolve_citations", resolve_citations)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("verify", verify)

    builder.add_edge(START, "guard_input")
    builder.add_conditional_edges("guard_input", after_guard, {"router": "router", "__end__": END})
    builder.add_edge("router", "agent")
    builder.add_conditional_edges(
        "agent",
        after_agent,
        {
            "tools": "tools",
            "finalize": "finalize",
            "budget_exceeded": "budget_exceeded",
            "resolve_citations": "resolve_citations",
        },
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("budget_exceeded", END)
    # finalize produces an answer the same way agent does, so it takes the same
    # path: there is exactly one route from "the model has answered" to the user,
    # and resolve_citations is on it.
    builder.add_edge("finalize", "resolve_citations")
    builder.add_conditional_edges(
        "resolve_citations", after_resolve, {"approval_gate": "approval_gate", "verify": "verify"}
    )
    builder.add_conditional_edges(
        "approval_gate", after_approval, {"verify": "verify", "__end__": END}
    )
    builder.add_edge("verify", END)

    return builder.compile(checkpointer=checkpointer)
