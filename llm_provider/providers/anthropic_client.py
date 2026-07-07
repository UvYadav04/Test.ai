from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from config import get_settings


def build_client(model: str = None):
    settings = get_settings()
    return AnthropicChatCompletionClient(
        model=model or "claude-3-5-sonnet-20241022",
        api_key=settings.ANTHROPIC_API_KEY,
    )
