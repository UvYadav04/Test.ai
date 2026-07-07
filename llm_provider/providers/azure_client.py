from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from config import get_settings


def build_client(model: str = None):
    settings = get_settings()
    return AzureOpenAIChatCompletionClient(
        model=model or settings.get("DEFAULT_MODEL", "gpt-4o-mini"),
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    )
