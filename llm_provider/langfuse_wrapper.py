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

calls_logger = logging.getLogger("llm_provider.calls")
calls_logger.setLevel(logging.WARNING)

_langfuse_client = None
_langfuse_checked = False


def _get_langfuse():
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
