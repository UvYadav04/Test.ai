from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}


def build_client(model: str = None):
    settings = get_settings()
    return OpenAIChatCompletionClient(
        model=model or "gemini-2.0-flash",
        api_key=settings.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_info=MODEL_INFO,
    )
