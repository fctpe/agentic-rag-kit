"""The agent graph: guard -> route -> agent/tools loop -> approval -> verify.

Explicit StateGraph rather than a prebuilt agent: the graph shape IS the
governance story — input guarding, a durable human approval interrupt for
report-type output (EU AI Act Art. 14 framing), and a grounding check before
anything reaches the user. Interrupts persist in the Postgres checkpointer,
so a pending approval survives a backend restart.
"""

import json
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompts import GROUNDING_PROMPT, REFUSAL_MESSAGE, ROUTER_PROMPT, SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.tools import CitationCollector, build_tools
from app.config import get_settings
from app.security.injection import assess_injection
from app.security.redaction import redact_pii

MAX_TOOL_ROUNDS = 6

REPORT_MARKERS = ("report", "memo", "checklist", "gap analysis", "executive summary", "draft")


def build_graph(session: AsyncSession, collector: CitationCollector, checkpointer: Any = None):
    settings = get_settings()
    model = init_chat_model(settings.llm_model, temperature=0)
    tools = build_tools(session, collector)
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    async def guard_input(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        text = str(last.content)
        if len(text) > settings.max_input_chars:
            return {
                "refused": True,
                "messages": [AIMessage(content="Input too long — please shorten your request.")],
            }
        redaction = redact_pii(text)
        injection = assess_injection(redaction.text)
        updates: dict[str, Any] = {
            "pii_found": redaction.found,
            "injection_signals": injection.signals,
        }
        if injection.flagged:
            updates["refused"] = True
            updates["messages"] = [AIMessage(content=REFUSAL_MESSAGE)]
        elif redaction.found:
            # Replace the raw message so PII never reaches the model or checkpoints.
            updates["messages"] = [
                HumanMessage(content=redaction.text, id=last.id),
            ]
        return updates

    def after_guard(state: AgentState) -> Literal["router", "__end__"]:
        return "__end__" if state.get("refused") else "router"

    async def router(state: AgentState) -> dict[str, Any]:
        query = str(state["messages"][-1].content)
        if any(marker in query.lower() for marker in REPORT_MARKERS):
            return {"task_type": "report"}
        response = await model.ainvoke(ROUTER_PROMPT.format(query=query[:2000]))
        verdict = str(response.content).strip().lower()
        return {"task_type": "report" if verdict == "report" else "qa"}

    async def agent(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def after_agent(state: AgentState) -> Literal["tools", "approval_gate", "verify"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            tool_rounds = sum(
                1 for message in state["messages"] if isinstance(message, ToolMessage)
            )
            if tool_rounds < MAX_TOOL_ROUNDS:
                return "tools"
        return "approval_gate" if state.get("task_type") == "report" else "verify"

    async def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        results: list[ToolMessage] = []
        for call in last.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                output = f"Unknown tool {call['name']}"
            else:
                try:
                    output = await tool.ainvoke(call["args"])
                except Exception as err:  # surfaced to the model for self-correction
                    output = f"Tool {call['name']} failed: {err}"
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    async def approval_gate(state: AgentState) -> dict[str, Any]:
        draft = str(state["messages"][-1].content)
        decision = interrupt(
            {
                "type": "approval_required",
                "draft": draft,
                "citations": collector.citations(),
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

    async def verify(state: AgentState) -> dict[str, Any]:
        answer = str(state["messages"][-1].content)
        citations = collector.citations()
        if not citations:
            return {"citations": [], "grounded": True, "grounding_issues": []}
        sources = collector.source_excerpts()
        response = await model.ainvoke(
            GROUNDING_PROMPT.format(sources=sources[:16000], answer=answer[:8000])
        )
        grounded, issues = True, []
        try:
            raw = str(response.content).strip().removeprefix("```json").removesuffix("```")
            verdict = json.loads(raw)
            grounded = bool(verdict.get("grounded", True))
            issues = [str(issue) for issue in verdict.get("issues", [])]
        except (json.JSONDecodeError, AttributeError):
            issues = ["Grounding check returned an unparseable verdict."]
        updates: dict[str, Any] = {
            "citations": citations,
            "grounded": grounded,
            "grounding_issues": issues,
        }
        if not grounded:
            updates["messages"] = [
                AIMessage(content="⚠ Grounding check flagged this answer: " + "; ".join(issues))
            ]
        return updates

    builder = StateGraph(AgentState)
    builder.add_node("guard_input", guard_input)
    builder.add_node("router", router)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools_node)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("verify", verify)

    builder.add_edge(START, "guard_input")
    builder.add_conditional_edges("guard_input", after_guard, {"router": "router", "__end__": END})
    builder.add_edge("router", "agent")
    builder.add_conditional_edges(
        "agent",
        after_agent,
        {"tools": "tools", "approval_gate": "approval_gate", "verify": "verify"},
    )
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges(
        "approval_gate", after_approval, {"verify": "verify", "__end__": END}
    )
    builder.add_edge("verify", END)

    return builder.compile(checkpointer=checkpointer)
