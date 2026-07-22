"""Wraps a ChatCompletionClient with bounded retry-with-backoff for genuinely transient errors
(connection blips, provider 5xx) - deliberately NOT for rate limits, which are frequently a
daily/monthly quota rather than a per-second burst, so retrying blindly can just hold a worker
slot for however long that window turns out to be (see errors.py's classify_llm_error).

Composed in provider.py between the raw/fallback client and LangfuseTracedChatCompletionClient,
so a Langfuse generation reflects the final outcome (with attempt count in its metadata) rather
than one entry per retry.
"""
import asyncio
import logging
import random
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

from llm_provider.errors import classify_llm_error

logger = logging.getLogger("llm_provider.retry")


class RetryingChatCompletionClient(ChatCompletionClient):
    def __init__(self, inner: ChatCompletionClient, max_attempts: int = 3, base_delay_s: float = 0.5):
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                      extra_create_args={}, cancellation_token=None) -> CreateResult:
        last_exc = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._inner.create(
                    messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                    extra_create_args=extra_create_args, cancellation_token=cancellation_token,
                )
            except Exception as exc:
                last_exc = exc
                info = classify_llm_error(exc)
                if not info.retryable or attempt == self._max_attempts:
                    if info.retryable:
                        logger.warning(
                            "llm call failed (%s), giving up after %d attempt(s): %s",
                            info.kind, attempt, exc,
                        )
                    raise
                delay = self._base_delay_s * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                logger.warning(
                    "llm call failed (%s) on attempt %d/%d, retrying in %.2fs: %s",
                    info.kind, attempt, self._max_attempts, delay, exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # unreachable - loop always either returns or raises above

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                             extra_create_args={}, cancellation_token=None):
        # Not retried - once streaming has started and the caller may have already consumed/acted
        # on partial chunks, transparently replaying the call from scratch isn't safe.
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
