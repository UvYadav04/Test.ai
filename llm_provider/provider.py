from llm_provider import registry
from config import get_settings


class LLMProvider:
    def __init__(self, provider_name: str = None):
        settings = get_settings()
        self.provider_name = provider_name or settings.get("DEFAULT_LLM_PROVIDER", "groq")

    def get_client(self, model: str = None):
        builder = registry.get_builder(self.provider_name)
        return builder(model)
