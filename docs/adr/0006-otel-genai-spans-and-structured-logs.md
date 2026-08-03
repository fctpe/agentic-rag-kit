# ADR 0006: OpenTelemetry spans on the GenAI conventions, and structured JSON logs

**Status:** accepted · 2026-08-03

## The gap

The backend had no logging at all — not one `import logging` under `backend/app/` — and no tracing beyond thirty optional lines of Langfuse. A tool that raised turned its traceback into a `ToolMessage` for the model to read and nowhere else; a graph run that blew up produced an SSE `error` event and no server-side record of why. Neither the token budget of ADR 0005 nor the cost it exists to bound was visible anywhere after the request ended.

## Langfuse stays

It ingests OTLP, so OTel standardises the existing integration instead of replacing it: the Langfuse callback handler keeps working with no collector, and the same runs also emit vendor-neutral spans to whatever endpoint an operator points at — which can be Langfuse. Both halves keep the rule `observability.py` started with: **no keys, no handler; no endpoint, no exporter.** With `OTEL_EXPORTER_OTLP_ENDPOINT` unset nothing is installed at all — no `TracerProvider`, no `BatchSpanProcessor`, no background thread, no network call. The OTel API then hands back non-recording spans, so every instrumented call site costs a context attach. `tests/test_observability.py::test_unset_endpoint_builds_no_exporter_and_installs_no_provider` proves it by making the exporter constructor and `set_tracer_provider` raise if they are ever reached.

## The conventions are not stable, and the code is built for that

As of August 2026 the GenAI semantic conventions are still **Development** status. The `gen_ai.client` spans — our model calls, `gen_ai.operation.name=chat` — are the stable part. The `gen_ai.agent` spans are experimental, which covers the run root (`invoke_agent`, `gen_ai.agent.name`) and the tool spans, and `gen_ai.evaluation.*` on the grounding verifier is newer still. The dedicated conventions repository has no tagged release to pin against, so there is nothing to depend on that would make the names hold still. Three consequences, all deliberate:

- **The OTel packages are pinned exactly**, not floated: `opentelemetry-api`, `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http` at `==1.44.0`. A minor bump that renames an attribute would otherwise change what the spans mean without changing a line of this repo.
- **`app/observability.py` spells the attribute names out as string literals** and never imports them. The generated Python constants exist only under `opentelemetry.semconv._incubating`, a private path whose maintainers reserve the right to move it, and `opentelemetry-semantic-conventions` ships as a beta (`0.65b0`). Eleven literals in one module are a smaller surface than a private import.
- **A test holds the literals to the generated constants.** `test_attribute_names_match_the_generated_semantic_conventions` imports the incubating module — a test may depend on a private path where the app may not — and asserts every `GEN_AI_*` constant equals its generated twin, so an SDK bump that renames one fails CI instead of quietly emitting a name no backend recognises. That test needs **no dependency of its own**, and the dev group declares none: `opentelemetry-sdk==1.44.0` requires `opentelemetry-semantic-conventions==0.65b0` at runtime, so the SDK pin above already installs and pins it. Calling it "a dev dependency" would be wrong on both counts, and a second pin on a transitive runtime dependency has no future except to contradict the first. The surviving reason the app spells the names out is the one in the previous bullet — the constants live behind a private import path. `test_the_generated_constants_need_no_dependency_of_their_own` asserts the SDK's declaration, so an SDK that ever drops it fails here instead of at an import.

## What gets a span

One root span per run (`invoke_agent agentic-rag-kit`, carrying `gen_ai.conversation.id` and `ragkit.request.id`), one `node.<name>` span per LangGraph node, one `execute_tool <name>` span per tool call, one `retrieval hybrid` span, one `embeddings <model>` span. Model calls attach `gen_ai.request.model` and `gen_ai.provider.name` — `llm_model` is `"openai:gpt-4o-mini"`, and the provider prefix is not part of the model name — plus `gen_ai.usage.input_tokens` / `output_tokens` read from the LangChain response's `usage_metadata`. The grounding verifier is `node.verify`: it records `gen_ai.evaluation.name=grounding` and a `gen_ai.evaluation.score.label` of `grounded`, `ungrounded` or `unverified`, and when the verifier itself fails closed — an unparseable verdict is not a pass — the span carries an ERROR status alongside the label. The third label is the case where there was nothing to check at all, which `ragkit.grounded` reports as `false` along with every other unverified run (ADR 0005) but which an operator will want to separate from a verdict the judge actually returned.

**One deliberate deviation.** The conventions say a model-call span SHOULD be named `{operation} {model}`. Ours keep the node name — `node.router`, not `chat gpt-4o-mini` — because each of those nodes makes exactly one model call, and which node made it is the thing an operator needs that the model name cannot tell them. `gen_ai.operation.name=chat` is still on the span, so a backend classifying by attribute rather than by name sees it as an LLM call. The alternative was a nested child span per call, doubling the span count for the name alone.

**The two retrieval arms get attributes, not spans.** They are one SQL statement on purpose (ADR 0002); splitting the query to produce two spans would undo the decision the instrumentation exists to observe. `retrieval hybrid` therefore records `ragkit.retrieval.returned_from_vector_arm` and `..._from_text_arm` — how many of the *returned* rows each arm ranked, which is the only per-arm signal that survives fusion. It is the number that says how often the deliberately AND-semantics text arm stays silent. The one part of the vector arm that is separately timeable is the query embedding, and that gets its own span; it carries no token counts because LangChain's embeddings interface returns vectors and drops the provider's usage block.

## Cost is configured, never assumed

`LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` are USD per million tokens, and with both at their default of 0 the spans carry token counts and **no** `ragkit.usage.cost_usd` at all. A price table baked into the repo goes stale without anyone noticing, and a cost of `0.0` reads as "free" rather than "unpriced" — the same reason ADR 0005 refuses to approximate tokens when a provider reports no `usage_metadata`. Run-level cost is the sum of the child spans, which is a query against a backend, not a second accounting path in the app that could disagree with the first.

## What the telemetry deliberately does not carry

No message text, no user query, no retrieved article text — on spans or in log lines. The conventions define `gen_ai.input.messages`, `gen_ai.output.messages` and `gen_ai.retrieval.query.text` and all three are omitted. PII redaction already runs in `guard_input` before anything reaches a model, a checkpoint or a trace (docs/security.md), and keeping the span attributes to counts, ids, model names and verdicts means the trace pipeline is not a second place that guarantee has to hold.

**The failure paths were the hole in that.** `logger.exception(...)` and `span.record_exception(err)` both write `str(err)`, and a SQLAlchemy `DBAPIError` renders as `... [SQL: ...] [parameters: (...)]` — the bound parameters, one of which is the user's query. A promise that holds on the happy path and breaks the moment retrieval errors is not a promise. `observability.error_fields(err)` is what the two call sites (the tool node, the graph run in `/chat`) emit instead: `exception.type` fully qualified, and `exception.stacktrace` built with `traceback.format_tb`, which formats the *frames* and leaves out the message line a full traceback ends with. An operator keeps the type and the code path; the message never leaves the process. The tool message handed back to the model still carries the full error — self-correction needs it, and that text lives behind the same redaction boundary as the rest of the checkpoint. `TestFailuresCarryNoQueryText` runs a tool failure whose message carries a canary and asserts where the canary may and may not appear.

## Structured logs, correlated with or without tracing

One JSON object per line on stdout, root handler replaced at startup — uvicorn's own handlers are cleared and re-pointed at it, or half the process would log plain text next to the JSON. Every line carries `request_id`, bound by an HTTP middleware from an upstream `X-Request-Id` or minted per request, and echoed back on the response. `trace_id` and `span_id` appear **only** while a span is recording: with tracing off a trace id would be sixteen zero bytes, and correlation falls back to the request id that always exists. The middleware binds the id before `call_next`, so the SSE generator — which outlives the middleware — inherits it with its task context.

Four things now log that previously went nowhere: a tool that raised (the model still gets its self-correction text), a graph run that failed behind the SSE `error` event, a run that stops at its token budget, and an answer produced with no retrieved sources. The first two log frames without the exception message, for the reason above.

## Reproducing the claims

```bash
cd backend
uv run python -c "from importlib.metadata import version; print(version('opentelemetry-sdk'))"   # 1.44.0
uv run python -c "from importlib.metadata import requires; print(requires('opentelemetry-sdk'))" # pins semantic-conventions==0.65b0
uv run pytest tests/test_observability.py -q                                                    # 18 passed
```

Rejected: importing the constants from `opentelemetry.semconv._incubating` (a private path, for eleven strings); pinning `opentelemetry-semantic-conventions` in the dev group (the SDK pin already pins it; a duplicate can only diverge); a hardcoded per-model price table (stale numbers presented as current cost); splitting the hybrid query into two spans (undoes ADR 0002); a second token accumulator for run-level cost (a number that can disagree with the budget's); putting prompts and retrieved text on spans, and `str(err)` on the failure paths (a second place the redaction guarantee would have to hold).
