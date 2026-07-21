from llm_provider import registry
from llm_provider.fallback_client import FallbackChatCompletionClient
from llm_provider.langfuse_wrapper import LangfuseTracedChatCompletionClient
from config import get_settings


class LLMProvider:
    def __init__(self, provider_name: str = None, fallback_provider: str = None):
        settings = get_settings()
        self.provider_name = provider_name or settings.get("DEFAULT_LLM_PROVIDER", "groq")
        self.fallback_provider = fallback_provider

    def get_client(self, model: str = None):
        builder = registry.get_builder(self.provider_name)

        if not self.fallback_provider or self.fallback_provider == self.provider_name:
            client = builder(model)
            return LangfuseTracedChatCompletionClient(client, self.provider_name, model)

        fallback_builder = registry.get_builder(self.fallback_provider)

        try:
            primary = builder(model)
        except Exception:
            # primary client couldn't even be constructed (e.g. missing credentials) - there's
            # nothing to wrap, so just use the fallback directly instead of crashing.
            fallback_client = fallback_builder(None)
            return LangfuseTracedChatCompletionClient(fallback_client, self.fallback_provider, None)

        wrapped = FallbackChatCompletionClient(primary, fallback_builder(None))
        # Wrap the OUTERMOST client so a fallback-triggered call still
        # produces exactly one Langfuse generation per .create(), regardless
        # of which of the two underlying clients actually served it.
        return LangfuseTracedChatCompletionClient(wrapped, self.provider_name, model)
