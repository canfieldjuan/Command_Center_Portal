# AI Portal Utility Decorators
# Extracted from main.py v26.2.0 - COMPLETE ORIGINAL FUNCTIONALITY

import asyncio
from functools import wraps
from typing import Callable, Any, TypeVar
import aiohttp
import structlog

logger = structlog.get_logger()
F = TypeVar('F', bound=Callable[..., Any])

def async_retry_with_backoff(retries: int = 4, initial_delay: float = 1.0, backoff: float = 2.0):
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    if e.status in [429, 500, 502, 503, 504] and i < retries - 1:
                        logger.warning("API call failed, retrying...", 
                                     attempt=i + 1, delay=delay, status=e.status)
                        await asyncio.sleep(delay)
                        delay *= backoff
                    else: 
                        raise
                except Exception: 
                    raise
        return wrapper
    return decorator
