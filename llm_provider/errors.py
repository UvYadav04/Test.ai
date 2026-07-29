from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMErrorInfo:
    kind: str
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
