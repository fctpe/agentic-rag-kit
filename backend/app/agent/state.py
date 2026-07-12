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
    citations: list[dict[str, Any]]
    grounded: bool
    grounding_issues: list[str]
    approval_decision: str
    approval_comment: str
