import asyncio

from autogen_core.models import UserMessage


def ask_llm(client, prompt: str) -> str:
    async def _run():
        result = await client.create(messages=[UserMessage(content=prompt, source="user")])
        return result.content

    return asyncio.run(_run())
