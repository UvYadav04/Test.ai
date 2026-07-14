import logging
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

logger = logging.getLogger("llm_provider.fallback")


class FallbackChatCompletionClient(ChatCompletionClient):
    """Wraps a primary and a fallback ChatCompletionClient. Every call tries the primary
    client first; if it raises, the same call is retried against the fallback client instead
    of failing the whole agent run. Usage/token accounting and model_info/capabilities are
    reported from whichever client most recently handled a call."""

    def __init__(self, primary: ChatCompletionClient, fallback: ChatCompletionClient):
        self._primary = primary
        self._fallback = fallback
        self._active = primary

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                      extra_create_args={}, cancellation_token=None) -> CreateResult:
        try:
            result = await self._primary.create(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            )
            self._active = self._primary
            return result
        except Exception as exc:
            logger.warning("primary model client failed (%s), falling back", exc)
            result = await self._fallback.create(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            )
            self._active = self._fallback
            return result

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                             extra_create_args={}, cancellation_token=None):
        try:
            async for chunk in self._primary.create_stream(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            ):
                self._active = self._primary
                yield chunk
        except Exception as exc:
            logger.warning("primary model client failed mid-stream (%s), falling back", exc)
            self._active = self._fallback
            async for chunk in self._fallback.create_stream(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            ):
                yield chunk

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()

    def actual_usage(self) -> RequestUsage:
        return self._active.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self._active.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._active.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._active.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._active.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._active.model_info
