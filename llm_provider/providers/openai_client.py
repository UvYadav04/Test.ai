from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings


def build_client(model: str = None):
    settings = get_settings()
    return OpenAIChatCompletionClient(
        model=model or settings.get("DEFAULT_MODEL", "gpt-4o-mini"),
        api_key=settings.OPENAI_API_KEY,
        parallel_tool_calls=False,
    )
