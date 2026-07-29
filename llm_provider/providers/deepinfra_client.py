from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}

DEFAULT_MAX_TOKENS = 8192

DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def build_client(model: str = None):
    settings = get_settings()
    return OpenAIChatCompletionClient(
        model=model or "Qwen/Qwen3-32B",
        api_key=settings.DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
        model_info=MODEL_INFO,
        parallel_tool_calls=False,
        max_tokens=DEFAULT_MAX_TOKENS,
        extra_body=DISABLE_THINKING,
    )
