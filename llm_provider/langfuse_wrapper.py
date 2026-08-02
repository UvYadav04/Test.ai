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

from shared.observability import get_langfuse_client, get_meter

logger = logging.getLogger("llm_provider.langfuse")

calls_logger = logging.getLogger("llm_provider.calls")
calls_logger.setLevel(logging.WARNING)

# Plain OTel token-usage counter, independent of whether Langfuse is configured - so "token
# usage" (per the observability spec's metrics list) shows up in Grafana Cloud even for anyone
# who only wired up the Alloy/Grafana side and skipped Langfuse Cloud.
_meter = get_meter("llm_provider")
_token_usage = _meter.create_counter(
    "llm.tokens", unit="tokens", description="LLM tokens consumed, keyed by provider/model/direction",
)
_llm_call_duration = _meter.create_histogram(
    "llm.call.duration_ms", unit="ms", description="LLM call wall time, keyed by provider/model/outcome",
)


def _record_tokens(provider: str, model: str, usage) -> None:
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is not None:
        _token_usage.add(prompt, {"provider": provider, "model": model or "default", "direction": "input"})
    if completion is not None:
        _token_usage.add(completion, {"provider": provider, "model": model or "default", "direction": "output"})


def _get_langfuse():
    # Delegates to shared/observability.py so there's exactly one place that constructs the
    # Langfuse client (via get_client(), reading LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY from the
    # environment - Langfuse Cloud by default, no self-hosted base_url hardcoded anywhere).
    return get_langfuse_client()


def _serialize_messages(messages):
    out = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or type(m).__name__
        content = getattr(m, "content", None)
        if not isinstance(content, (str, list, dict, type(None))):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def _serialize_tools(tools):
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
    def __init__(self, inner: ChatCompletionClient, provider_name: str, model: str = None):
        self._inner = inner
        self._provider_name = provider_name
        self._model = model

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                      extra_create_args={}, cancellation_token=None) -> CreateResult:
        start = time.monotonic()
        call_id = id(messages)
        if calls_logger.    isEnabledFor(logging.INFO):
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
            _llm_call_duration.record(
                elapsed * 1000, {"provider": self._provider_name, "model": self._model or "default", "outcome": "error"},
            )
            raise
        elapsed = time.monotonic() - start
        usage = getattr(result, "usage", None)
        _record_tokens(self._provider_name, self._model, usage)
        _llm_call_duration.record(
            elapsed * 1000, {"provider": self._provider_name, "model": self._model or "default", "outcome": "ok"},
        )
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
        generation = langfuse.start_observation(
            name=f"{self._provider_name}.create",
            as_type="generation",
            model=self._model or self._provider_name,
            model_parameters={
                "temperature": (extra_create_args or {}).get("temperature"),
                "tool_choice": str(tool_choice),
            },
            input=_serialize_messages(messages),
            metadata={"provider": self._provider_name, "tool_count": len(tools), "tools": _serialize_tools(tools)},
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
            tool_calls = [
                c for c in (result.content if isinstance(result.content, list) else [])
                if hasattr(c, "name") or hasattr(c, "arguments")
            ]
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
                    "tool_calls": _safe_json(tool_calls) if tool_calls else None,
                },
            )
            return result
        finally:
            generation.end()

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                             extra_create_args={}, cancellation_token=None):
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
        last_chunk = None
        langfuse = _get_langfuse()
        generation = None
        if langfuse is not None:
            generation = langfuse.start_observation(
                name=f"{self._provider_name}.create_stream",
                as_type="generation",
                model=self._model or self._provider_name,
                model_parameters={"tool_choice": str(tool_choice)},
                input=_serialize_messages(messages),
                metadata={"provider": self._provider_name, "tool_count": len(tools), "streaming": True},
            )
        try:
            async for chunk in self._inner.create_stream(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            ):
                chunk_count += 1
                last_chunk = chunk
                yield chunk
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning(
                "llm stream %s/%s failed after %.3fs (%d chunks): %s",
                self._provider_name, self._model or "default", elapsed, chunk_count, exc,
            )
            calls_logger.warning("llm stream [%x] failed after %.3fs (%d chunks): %s", call_id, elapsed, chunk_count, exc)
            _llm_call_duration.record(
                elapsed * 1000, {"provider": self._provider_name, "model": self._model or "default", "outcome": "error"},
            )
            if generation is not None:
                generation.update(level="ERROR", status_message=str(exc))
            raise
        else:
            elapsed = time.monotonic() - start
            logger.info(
                "llm stream %s/%s took %.3fs (%d chunks)",
                self._provider_name, self._model or "default", elapsed, chunk_count,
            )
            calls_logger.info("llm stream response [%x] completed in %.3fs (%d chunks)", call_id, elapsed, chunk_count)
            usage = getattr(last_chunk, "usage", None)
            _record_tokens(self._provider_name, self._model, usage)
            _llm_call_duration.record(
                elapsed * 1000, {"provider": self._provider_name, "model": self._model or "default", "outcome": "ok"},
            )
            if generation is not None:
                # autogen's create_stream yields str chunks and finishes with a CreateResult -
                # that final chunk is the only one with usage/content worth recording.
                generation.update(
                    output=getattr(last_chunk, "content", last_chunk),
                    usage_details=(
                        {
                            "input": getattr(usage, "prompt_tokens", None),
                            "output": getattr(usage, "completion_tokens", None),
                        }
                        if usage is not None
                        else None
                    ),
                    metadata={"finish_reason": getattr(last_chunk, "finish_reason", None), "latency_s": round(elapsed, 3)},
                )
        finally:
            if generation is not None:
                generation.end()

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
