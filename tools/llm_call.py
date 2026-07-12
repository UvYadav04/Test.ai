import asyncio

from autogen_core.models import UserMessage


def ask_llm(client, prompt: str) -> str:
    async def _run():
        result = await client.create(messages=[UserMessage(content=prompt, source="user")])
        await client.close()
        return result.content

    return asyncio.run(_run())
