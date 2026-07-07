from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}


def build_client(model: str = None):
    settings = get_settings()
    return OpenAIChatCompletionClient(
        model=model or "openai/gpt-oss-120b",
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_info=MODEL_INFO,
    )
