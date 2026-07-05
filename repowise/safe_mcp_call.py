import asyncio
from functools import wraps

def safe_mcp_call(fn=None, *, max_retries=3, timeout=10):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    async with asyncio.timeout(timeout):
                        return await func(*args, **kwargs)

                except asyncio.TimeoutError:
                    if attempt == max_retries - 1:
                        raise TimeoutError(
                            f"{func.__name__} timed out after {timeout}s"
                        )

                except asyncio.CancelledError:
                    raise

                except Exception:
                    if attempt == max_retries - 1:
                        raise

                await asyncio.sleep(2 ** attempt)

        return wrapper

    if fn is None:
        return decorator

    return decorator(fn)