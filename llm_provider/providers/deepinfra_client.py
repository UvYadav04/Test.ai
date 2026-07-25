from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}

# DeepInfra serves Qwen/Qwen3-32B with max_model_len=40960 (prompt + completion combined). When
# a request omits max_tokens the server falls back to the model's own generation_config default
# (65536), which exceeds that and gets rejected outright ("max_tokens=65536 cannot be greater
# than max_model_len=40960"). An explicit, safely-below-40960 max_tokens on every request avoids
# that - leaves plenty of headroom for prompt/tool-call context on top of this.
DEFAULT_MAX_TOKENS = 8192

# Qwen3 is a hybrid reasoning model - by default every response is wrapped in a <think>...
# </think> reasoning block ahead of the actual answer (see
# https://huggingface.co/Qwen/Qwen3-32B). Autogen just takes the raw completion text as the
# message content, so with thinking left on that <think> block rides along through the whole
# tool-calling loop and can leak straight into the orchestrator's final user-facing answer
# (observed: a chat reply that opened with a literal "<think> Okay, the user asked...").
# vLLM (what DeepInfra serves this on) supports turning it off per-request via
# chat_template_kwargs - see https://qwen.readthedocs.io - which also means every tool-call
# turn spends its max_tokens budget on the actual decision instead of competing with a chain of
# thought first.
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
