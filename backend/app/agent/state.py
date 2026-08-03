from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

TaskType = Literal["qa", "report"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    task_type: TaskType
    pii_found: list[str]
    injection_signals: list[str]
    refused: bool
    # Full retrieved chunks (JSON-serializable), written by the tools node so
    # they are checkpointed. verify() and the approval interrupt read these,
    # never the per-request collector — otherwise a resumed approval (which
    # does not re-run tools) would lose every citation.
    retrieved_sources: list[dict[str, Any]]
    # Total tokens spent by every model call in this request, from
    # usage_metadata. Reset by guard_input so a long-running thread does not
    # inherit the previous turn's spend, and carried across an approval resume,
    # which is the same run.
    tokens_used: int
    citations: list[dict[str, Any]]
    # False until a grounding check actually passes. guard_input seeds it so a
    # run that never reaches verify() — refused, rejected, out of budget — cannot
    # report a verdict nobody produced; grounding_issues says which case it was.
    grounded: bool
    grounding_issues: list[str]
    approval_decision: str
    approval_comment: str
