import asyncio

from autogen_core.models import UserMessage


async def ask_llm_async(client, prompt: str) -> str:
    result = await client.create(messages=[UserMessage(content=prompt, source="user")])
    await client.close()
    return result.content


def ask_llm(client, prompt: str) -> str:
    return asyncio.run(ask_llm_async(client, prompt))
