"""Tiny shared helper - no tokenizer dependency needed for either caller (document_processor's
batching, metadata's estimated_token_count). ~4 characters per token is the standard
rule-of-thumb approximation for English text; both use cases only need a stable, bounded
estimate, not an exact count.
"""

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // _CHARS_PER_TOKEN)
