# LLM Provider

Builds autogen model client instances for different LLM providers. Supported out of the box: `openai`, `anthropic`, `azure`, `groq`, `gemini`.

```python
from llm_provider import LLMProvider

provider = LLMProvider("groq")
client = provider.get_client()
```

Leave out the provider name to use `DEFAULT_LLM_PROVIDER` from `.env` (defaults to `groq`, model `openai/gpt-oss-120b`). `groq` and `gemini` both go through Groq's/Google's OpenAI-compatible endpoints, so they use `OpenAIChatCompletionClient` with a custom `base_url` under the hood.

## Add a new provider

Create `providers/<name>_client.py` with a `build_client(model=None)` function returning an autogen model client. Register it in `registry.py`:

```python
PROVIDER_REGISTRY["groq"] = groq_client.build_client
```

## Config

`config.py` loads every key/value pair from `.env` and exposes `get_settings()`. Adding a new key to `.env` doesn't need any code change - read it anywhere with `get_settings().YOUR_KEY` or `get_settings().get("YOUR_KEY", "default")`. Copy `.env.example` to `.env` and fill in your keys.
