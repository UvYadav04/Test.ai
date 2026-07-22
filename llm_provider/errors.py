"""Classifies exceptions raised by ChatCompletionClient.create() (openai/anthropic/google-genai/
groq SDK errors, all reached through autogen_ext's model clients) into a retry decision plus a
short, non-leaky message safe to show a user.

The raw exception text from these SDKs routinely includes internal detail that has no business
ending up in a chat message - e.g. the Groq 429 this was written for included the exact daily
token quota, tokens used, the org id, and a billing upsell link. investigation.py used to
interpolate str(exc) straight into the assistant's reply; classify_llm_error() gives it something
safe to show instead.

Duck-typed rather than importing every provider SDK's exception classes directly -
openai_client.py, anthropic_client.py, gemini_client.py, and groq_client.py (which uses openai's
SDK against Groq's OpenAI-compatible endpoint) each raise their own SDK's exception types, and
this module shouldn't have to import all four just to tell a 429 apart from a timeout. Every
mainstream provider SDK's HTTP-backed errors expose a `status_code` (or `code`) attribute and/or
have a class name that says what they are - that's enough to classify on.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMErrorInfo:
    kind: str  # "rate_limit" | "auth" | "connection" | "server" | "unknown"
    retryable: bool
    user_message: str
    retry_after_s: Optional[float] = None


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def classify_llm_error(exc: Exception) -> LLMErrorInfo:
    name = type(exc).__name__
    status = _status_code(exc)

    if "RateLimit" in name or status == 429:
        return LLMErrorInfo(
            kind="rate_limit",
            # Rate limits are often a DAILY/monthly quota, not a per-second burst (the Groq error
            # this was written for said "try again in 29m33s") - blindly retrying here would just
            # hold a worker slot for however long that turns out to be. NOT auto-retried for that
            # reason. If a genuinely different provider is configured as fallback (see
            # llm_provider/provider.py), FallbackChatCompletionClient already tried it BEFORE this
            # classification is ever reached - this only fires once every configured option is
            # exhausted.
            retryable=False,
            user_message=(
                "The AI provider is temporarily rate-limited. This usually clears within a few "
                "minutes to half an hour - please try again shortly."
            ),
            retry_after_s=_retry_after_seconds(exc),
        )

    if "Authentication" in name or "PermissionDenied" in name or status in (401, 403):
        return LLMErrorInfo(
            kind="auth",
            retryable=False,
            user_message=(
                "The AI provider rejected the request due to a credentials or permissions issue. "
                "This needs attention from an administrator, not a retry."
            ),
        )

    if "Connection" in name or "Timeout" in name or status in (502, 503, 504):
        return LLMErrorInfo(
            kind="connection",
            retryable=True,
            user_message="Had trouble reaching the AI provider - retrying automatically.",
        )

    if status is not None and status >= 500:
        return LLMErrorInfo(
            kind="server",
            retryable=True,
            user_message="The AI provider had a temporary server error - retrying automatically.",
        )

    return LLMErrorInfo(
        kind="unknown",
        retryable=False,
        user_message="Something went wrong talking to the AI provider.",
    )
