import asyncio

from autogen_core.models import UserMessage


async def ask_llm_async(client, prompt: str) -> str:
    """The async-native version of ask_llm below - use this from callers that
    are themselves already running on an event loop (e.g. worker_service's
    arq job functions). asyncio.run() inside ask_llm would raise there
    ("cannot run event loop while another loop is running")."""
    result = await client.create(messages=[UserMessage(content=prompt, source="user")])
    await client.close()
    return result.content


def ask_llm(client, prompt: str) -> str:
    """Sync convenience wrapper for callers that are themselves plain sync
    functions (e.g. tool methods AutoGen invokes off the event loop thread) -
    see ask_llm_async's docstring for why this can't be used from async code."""
    return asyncio.run(ask_llm_async(client, prompt))
