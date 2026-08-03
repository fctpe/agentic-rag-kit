"""Tracing and structured logs, both off until they are configured.

Langfuse keeps its LangChain callback handler — it needs no collector — while
OTel spans go to whatever OTLP endpoint is set, which can be Langfuse itself
since it ingests OTLP. The rule this module started with holds for both: no
keys means no Langfuse handler, no endpoint means no exporter, no background
thread and no network call. With tracing off the OTel API hands back
non-recording spans, so the instrumentation costs a context attach.

PII is redacted in guard_input BEFORE anything can reach a trace or a log line
— see docs/security.md. Span attributes here carry counts, ids, model names
and verdicts; never message, query or retrieved text.
"""

import functools
import json
import logging
import os
import sys
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

from app.config import get_settings

SERVICE = "agentic-rag-kit"

# The GenAI semantic conventions are Development status and their generated
# Python constants live behind opentelemetry.semconv._incubating — a private
# path its maintainers reserve the right to move. Spelling the names out keeps
# the conventions we target reviewable in one place; ADR 0006 records which of
# them are stable and which are not.
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Ours, because the conventions define none: there is no gen_ai cost attribute,
# and the retrieval attributes they do define do not describe a fused two-arm
# query.
COST_USD = "ragkit.usage.cost_usd"
REQUEST_ID = "ragkit.request.id"
TOKENS_USED = "ragkit.tokens_used"
GROUNDED = "ragkit.grounded"
RETRIEVAL_ARM_SIZE = "ragkit.retrieval.arm_size"
RETRIEVAL_FINAL_K = "ragkit.retrieval.final_k"
RETRIEVAL_REGULATION = "ragkit.retrieval.regulation"
RETRIEVAL_RETURNED = "ragkit.retrieval.returned"
# Of the rows returned, how many each arm ranked. These are not arm sizes: both
# arms are capped at arm_size and fused inside one SQL statement (ADR 0002), so
# their contribution is only observable after fusion.
RETRIEVAL_FROM_VECTOR_ARM = "ragkit.retrieval.returned_from_vector_arm"
RETRIEVAL_FROM_TEXT_ARM = "ragkit.retrieval.returned_from_text_arm"

tracer = trace.get_tracer(SERVICE)
logger = logging.getLogger(__name__)

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_provider: TracerProvider | None = None


def get_trace_callbacks() -> list[Any]:
    """LangChain callback handlers for tracing. Per-run identity (user, thread)
    is attached separately via trace_metadata() on the graph config."""
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return []
    return [CallbackHandler()]


def trace_metadata(user_id: str, thread_id: str) -> dict[str, Any]:
    return {
        "langfuse_user_id": user_id,
        "langfuse_session_id": thread_id,
    }


def setup_observability() -> None:
    """JSON logging always; OTLP export only when an endpoint is configured."""
    global _provider
    _configure_logging()
    settings = get_settings()
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if not endpoint or _provider is not None:
        return
    # pydantic-settings reads .env without exporting it, and the exporter reads
    # its OTEL_* variables from the process environment. Bridging the two here
    # means one documented endpoint works from either place while the exporter
    # still applies its own path suffix and header parsing.
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
    if settings.otel_exporter_otlp_headers:
        os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", settings.otel_exporter_otlp_headers)
    _provider = TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE}))
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_provider)
    logger.info("otlp tracing enabled", extra={"fields": {"endpoint": endpoint}})


def shutdown_observability() -> None:
    """Flush the batch processor — spans still buffered at shutdown are lost
    otherwise, and that tail is exactly the request that took the app down."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


class JsonFormatter(logging.Formatter):
    """One JSON object per line. trace_id/span_id appear only while a span is
    recording, so a deployment with tracing off still correlates on
    request_id."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id.get(),
        }
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            payload["trace_id"] = format_trace_id(context.trace_id)
            payload["span_id"] = format_span_id(context.span_id)
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(get_settings().log_level.upper())
    # uvicorn installs its own handlers and disables propagation, which would
    # leave half the process emitting plain text next to the JSON.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True


@contextmanager
def request_context(supplied: str | None) -> Iterator[str]:
    """Bind the correlation id every log line carries. An upstream
    X-Request-Id wins, so a request stays followable across a proxy."""
    request_id = supplied or uuid.uuid4().hex
    token = _request_id.set(request_id)
    try:
        yield request_id
    finally:
        _request_id.reset(token)


def get_request_id() -> str:
    return _request_id.get()


def traced_node(name: str) -> Callable[[Any], Any]:
    """Span per LangGraph node, applied at the definition so the node bodies
    read as they did before."""

    def decorate(func: Any) -> Any:
        @functools.wraps(func)
        async def wrapper(state: Any) -> Any:
            with tracer.start_as_current_span(f"node.{name}"):
                return await func(state)

        return wrapper

    return decorate


def error_fields(err: BaseException) -> dict[str, str]:
    """Everything a failure is allowed to say on a span or in a log line.

    Never str(err), and never span.record_exception / logger.exception, both of
    which carry it: a SQLAlchemy DBAPIError puts the statement and its bound
    parameters in its message, and the user's query is one of those parameters.
    docs/security.md promises spans and log lines carry no query text at all.
    traceback.format_tb formats the frames only — the message line that ends a
    full traceback is exactly what it leaves out — so an operator still gets the
    type and the code path.
    """
    return {
        "exception.type": f"{type(err).__module__}.{type(err).__qualname__}",
        "exception.stacktrace": "".join(traceback.format_tb(err.__traceback__)),
    }


def record_model_call(response: Any) -> None:
    """GenAI request and usage attributes for one model call, on the current
    span.

    A provider that reports no usage_metadata gets no usage attributes rather
    than zeros — the same gap the token budget leaves open (ADR 0005), left
    visible the same way here.
    """
    span = trace.get_current_span()
    provider, _, model = get_settings().llm_model.partition(":")
    span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
    span.set_attribute(GEN_AI_REQUEST_MODEL, model or provider)
    if model:
        span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    input_tokens, output_tokens = int(usage["input_tokens"]), int(usage["output_tokens"])
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    cost = usd_cost(input_tokens, output_tokens)
    if cost is not None:
        span.set_attribute(COST_USD, cost)


def usd_cost(input_tokens: int, output_tokens: int) -> float | None:
    """None when neither price is configured — see config.Settings for why an
    unpriced call gets no number rather than a zero."""
    settings = get_settings()
    if not (settings.llm_input_price_per_mtok or settings.llm_output_price_per_mtok):
        return None
    return (
        input_tokens * settings.llm_input_price_per_mtok
        + output_tokens * settings.llm_output_price_per_mtok
    ) / 1_000_000
