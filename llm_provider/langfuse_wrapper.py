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
    fixed shape that might not hold across every message type."""
    out = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or type(m).__name__
        content = getattr(m, "content", None)
        if not isinstance(content, (str, list, dict, type(None))):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


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
        langfuse = _get_langfuse()
        if langfuse is None:
            return await self._inner.create(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            )

        start = time.monotonic()
        generation = langfuse.start_generation(
            name=f"{self._provider_name}.create",
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
        # Not traced yet - the tool-calling agent loop (orchestrator/tabular/
        # document agents) uses .create(), not streaming. Add tracing here if
        # create_stream starts seeing real use.
        async for chunk in self._inner.create_stream(
            messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
            extra_create_args=extra_create_args, cancellation_token=cancellation_token,
        ):
            yield chunk

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
