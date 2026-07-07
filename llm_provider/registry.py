from llm_provider.providers import anthropic_client, azure_client, openai_client

PROVIDER_REGISTRY = {
    "openai": openai_client.build_client,
    "anthropic": anthropic_client.build_client,
    "azure": azure_client.build_client,
}


def get_builder(provider_name: str):
    builder = PROVIDER_REGISTRY.get(provider_name)
    if builder is None:
        supported = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unsupported provider '{provider_name}'. Supported: {supported}")
    return builder
