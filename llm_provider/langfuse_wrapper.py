"""Wraps any autogen ChatCompletionClient so every `.create()` call is
reported to Langfuse as a "generation" observation (model, input messages,
output, token usage, latency, errors) - without changing call behavior.
Same wrap-and-delegate pattern as fallback_client.FallbackChatCompletionClient.

Deliberately does NOT import from Server/shared/ - analyzerEngine is meant to
stay importable as its own root (see worker_service/engine_bootstrap.py's
docstring), so this reads Langfuse credentials via analyzerEngine's own
config.get_settings(), same as every provider module in llm_provider/providers/.

No-ops (zero overhead beyond one dict lookup) if LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY aren't set - Langfuse tracing is additive, same as the
Loki logging setup in shared/logging_config.py. The Langfuse SDK also
catches and logs its own internal errors rather than raising them, so a
bad/unreachable Langfuse host degrades to "no tracing", not a broken LLM call.
"""
import json
import logging
import time
from typing import Sequence

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema

from config import get_settings

logger = logging.getLogger("llm_provider.langfuse")

# Separate logger for the FULL raw request/response of every LLM call (whole message list -
# system prompt + every prior user/assistant/tool message the model is about to see - plus the
# complete tool schemas exposed on that call, and the model's raw response). `logger` above stays
# a one-line-per-call summary (provider/model/duration/token counts) for normal operation.
#
# Turned OFF by default (setLevel(WARNING) below) - a full message/tool-schema dump on every
# single LLM call is far too long/noisy for normal operation. To turn it back on temporarily
# (e.g. to debug exactly what's being sent to the model), set this logger's level to INFO -
# `logging.getLogger("llm_provider.calls").setLevel(logging.INFO)` - from wherever you configure
# logging (shared/logging_config.py, or just at the top of a one-off script). Every call site
# below checks isEnabledFor(INFO) before doing the (non-trivial) serialization work, so leaving
# this at WARNING costs nothing beyond the one isEnabledFor check per call.
calls_logger = logging.getLogger("llm_provider.calls")
calls_logger.setLevel(logging.WARNING)

_langfuse_client = None
_langfuse_checked = False


def _get_langfuse():
    """Lazy singleton, same shape as llm_provider's other module-level state.
    Returns None (and stays None) if Langfuse isn't configured, so callers
    can just check `if langfuse is None: skip tracing` once per call."""
    global _langfuse_client, _langfuse_checked
    if _langfuse_checked:
        return _langfuse_client
    _langfuse_checked = True

    settings = get_settings()
    public_key = settings.get("LANGFUSE_PUBLIC_KEY")
    secret_key = settings.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.info("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - LLM call tracing disabled")
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse is not installed - add it to analyzerEngine/requirements.txt")
        return None

    _langfuse_client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=settings.get("LANGFUSE_HOST", "http://langfuse-web:3000"),
    )
    return _langfuse_client


def _serialize_messages(messages):
    """Best-effort JSON-able view of autogen LLMMessage objects for Langfuse's
    `input` field - these are typed objects (SystemMessage/UserMessage/...),
    not plain dicts, so pull role/content defensively instead of assuming a
    fixed shape that might not hold across every message type.

    This IS the "whole prompt" - autogen's model_context accumulates the entire conversation
    (system message, the original task, every prior assistant turn, every tool call and its
    result) and hands the FULL list back to ChatCompletionClient.create() on every single call,
    never just the newest message - so serializing `messages` in full, as calls_logger's request
    line does below, already captures the whole prompt AND every previous message in one shot."""
    out = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or type(m).__name__
        content = getattr(m, "content", None)
        if not isinstance(content, (str, list, dict, type(None))):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def _serialize_tools(tools):
    """Full schema (name, description, parameters JSON-schema) for every tool exposed to the
    model on this call - not just a count. `tools` items are either autogen_core `Tool`
    instances (real objects with a `.schema` property) or plain `ToolSchema` dicts already -
    autogen's AssistantAgent passes the former, but this stays defensive either way."""
    out = []
    for t in tools or []:
        try:
            schema = t.schema if hasattr(t, "schema") else t
            out.append({
                "name": schema.get("name"),
                "description": schema.get("description"),
                "parameters": schema.get("parameters"),
            })
        except Exception:
            out.append({"name": getattr(t, "name", str(t)), "description": None, "parameters": None})
    return out


def _safe_json(value) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        return str(value)


class LangfuseTracedChatCompletionClient(ChatCompletionClient):
    """Wraps `inner` (a plain provider client, or a FallbackChatCompletionClient
    - either way, whatever actually ends up serving the call) so each
    `.create()` becomes one Langfuse generation. Wrap the OUTERMOST client in
    provider.py so a fallback-provider call still produces exactly one trace
    entry, tagged with whichever provider actually handled it."""

    def __init__(self, inner: ChatCompletionClient, provider_name: str, model: str = None):
        self._inner = inner
        self._provider_name = provider_name
        self._model = model

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                      extra_create_args={}, cancellation_token=None) -> CreateResult:
        """Every LLM call, from every agent (hypothesis/tabular/document/orchestrator all go
        through this same wrapper - it's the outermost layer LLMProvider._wrap applies
        unconditionally, see provider.py), passes through here. Duration is logged via
        `logger` below on every call regardless of whether Langfuse itself is configured -
        Langfuse's own latency_s metadata (inside _create_and_trace) is for the Langfuse UI;
        this log line is what shows up in worker_service's own console/logs.

        `calls_logger` below additionally logs the COMPLETE request (every message in
        `messages` - the whole accumulated prompt, system message included, plus every earlier
        turn - and every tool's full name/description/parameter schema in `tools`) before the
        call, and the raw response content after - see _serialize_messages/_serialize_tools."""
        start = time.monotonic()
        call_id = id(messages)  # cheap correlation key to pair this call's request/response
        # line/error line in the log stream, since concurrent calls can interleave.
        if calls_logger.isEnabledFor(logging.INFO):
            calls_logger.info(
                "llm request [%x] %s/%s - %d message(s), %d tool(s), tool_choice=%s:\n%s",
                call_id, self._provider_name, self._model or "default", len(messages), len(tools), tool_choice,
                _safe_json({
                    "messages": _serialize_messages(messages),
                    "tools": _serialize_tools(tools),
                }),
            )
        try:
            result = await self._create_and_trace(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning(
                "llm call %s/%s failed after %.3fs (tools=%d): %s",
                self._provider_name, self._model or "default", elapsed, len(tools), exc,
            )
            calls_logger.warning("llm request [%x] failed after %.3fs: %s", call_id, elapsed, exc)
            raise
        elapsed = time.monotonic() - start
        usage = getattr(result, "usage", None)
        logger.info(
            "llm call %s/%s took %.3fs (tools=%d, tokens in/out=%s/%s)",
            self._provider_name, self._model or "default", elapsed, len(tools),
            getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
        )
        if calls_logger.isEnabledFor(logging.INFO):
            calls_logger.info(
                "llm response [%x] %s/%s (%.3fs, finish_reason=%s, tokens in/out=%s/%s):\n%s",
                call_id, self._provider_name, self._model or "default", elapsed,
                getattr(result, "finish_reason", None),
                getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
                _safe_json(result.content),
            )
        return result

    async def _create_and_trace(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                                 extra_create_args={}, cancellation_token=None) -> CreateResult:
        langfuse = _get_langfuse()
        if langfuse is None:
            return await self._inner.create(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            )

        start = time.monotonic()
        # Langfuse v4 unified start_span/start_generation into start_observation(as_type=...).
        # See https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4 -
        # `as_type="generation"` is the direct replacement for the old start_generation() call.
        generation = langfuse.start_observation(
            name=f"{self._provider_name}.create",
            as_type="generation",
            model=self._model or self._provider_name,
            input=_serialize_messages(messages),
            metadata={"provider": self._provider_name, "tool_count": len(tools)},
        )
        try:
            result = await self._inner.create(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            )
        except Exception as exc:
            generation.update(level="ERROR", status_message=str(exc))
            raise
        else:
            usage = getattr(result, "usage", None)
            generation.update(
                output=result.content,
                usage_details=(
                    {
                        "input": getattr(usage, "prompt_tokens", None),
                        "output": getattr(usage, "completion_tokens", None),
                    }
                    if usage is not None
                    else None
                ),
                metadata={
                    "finish_reason": getattr(result, "finish_reason", None),
                    "cached": getattr(result, "cached", None),
                    "latency_s": round(time.monotonic() - start, 3),
                },
            )
            return result
        finally:
            generation.end()

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                             extra_create_args={}, cancellation_token=None):
        # Not traced in Langfuse - the tool-calling agent loop (orchestrator/tabular/
        # document agents) uses .create(), not streaming; add Langfuse tracing here if
        # create_stream starts seeing real use. Duration IS logged locally either way, same
        # as .create() above, so a streaming caller isn't left with zero timing visibility.
        start = time.monotonic()
        call_id = id(messages)
        if calls_logger.isEnabledFor(logging.INFO):
            calls_logger.info(
                "llm stream request [%x] %s/%s - %d message(s), %d tool(s), tool_choice=%s:\n%s",
                call_id, self._provider_name, self._model or "default", len(messages), len(tools), tool_choice,
                _safe_json({
                    "messages": _serialize_messages(messages),
                    "tools": _serialize_tools(tools),
                }),
            )
        chunk_count = 0
        try:
            async for chunk in self._inner.create_stream(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            ):
                chunk_count += 1
                yield chunk
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning(
                "llm stream %s/%s failed after %.3fs (%d chunks): %s",
                self._provider_name, self._model or "default", elapsed, chunk_count, exc,
            )
            calls_logger.warning("llm stream [%x] failed after %.3fs (%d chunks): %s", call_id, elapsed, chunk_count, exc)
            raise
        else:
            elapsed = time.monotonic() - start
            logger.info(
                "llm stream %s/%s took %.3fs (%d chunks)",
                self._provider_name, self._model or "default", elapsed, chunk_count,
            )
            calls_logger.info("llm stream response [%x] completed in %.3fs (%d chunks)", call_id, elapsed, chunk_count)

    async def close(self) -> None:
        await self._inner.close()

    def actual_usage(self) -> RequestUsage:
        return self._inner.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self._inner.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._inner.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._inner.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._inner.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._inner.model_info
