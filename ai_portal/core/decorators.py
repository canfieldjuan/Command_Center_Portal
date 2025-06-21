"""
Utility decorators for AI Portal application
"""

import asyncio
import aiohttp
import structlog
from functools import wraps
from typing import Callable, TypeVar, Any

logger = structlog.get_logger()

F = TypeVar('F', bound=Callable[..., Any])

def async_retry_with_backoff(retries: int = 4, initial_delay: float = 1.0, backoff: float = 2.0):
    """
    Async retry decorator with exponential backoff for resilient API calls
    
    Args:
        retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff: Multiplier for delay on each retry
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    last_exception = e
                    # Retry on specific HTTP status codes
                    if e.status in [429, 500, 502, 503, 504] and attempt < retries - 1:
                        logger.warning("API call failed, retrying...", 
                                     attempt=attempt + 1, 
                                     delay=delay, 
                                     status=e.status,
                                     function=func.__name__)
                        await asyncio.sleep(delay)
                        delay *= backoff
                    else: 
                        logger.error("API call failed permanently", 
                                   attempt=attempt + 1,
                                   status=e.status,
                                   function=func.__name__)
                        raise
                except asyncio.TimeoutError as e:
                    last_exception = e
                    if attempt < retries - 1:
                        logger.warning("API call timed out, retrying...", 
                                     attempt=attempt + 1, 
                                     delay=delay,
                                     function=func.__name__)
                        await asyncio.sleep(delay)
                        delay *= backoff
                    else:
                        logger.error("API call timed out permanently", 
                                   attempt=attempt + 1,
                                   function=func.__name__)
                        raise
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error("API call failed with unexpected error", 
                               attempt=attempt + 1,
                               error=str(e),
                               function=func.__name__)
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator

def rate_limit(calls_per_second: float = 1.0):
    """
    Rate limiting decorator for API calls
    
    Args:
        calls_per_second: Maximum calls per second allowed
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = asyncio.get_event_loop().time()
            time_since_last = now - last_called[0]
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.debug("Rate limiting API call", 
                           sleep_time=sleep_time,
                           function=func.__name__)
                await asyncio.sleep(sleep_time)
            
            last_called[0] = asyncio.get_event_loop().time()
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def log_execution_time(log_level: str = "info"):
    """
    Decorator to log function execution time
    
    Args:
        log_level: Logging level for timing information
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            try:
                result = await func(*args, **kwargs)
                execution_time = asyncio.get_event_loop().time() - start_time
                
                log_func = getattr(logger, log_level, logger.info)
                log_func("Function execution completed", 
                        function=func.__name__,
                        execution_time=round(execution_time, 3))
                
                return result
            except Exception as e:
                execution_time = asyncio.get_event_loop().time() - start_time
                logger.error("Function execution failed", 
                           function=func.__name__,
                           execution_time=round(execution_time, 3),
                           error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log_func = getattr(logger, log_level, logger.info)
                log_func("Function execution completed", 
                        function=func.__name__,
                        execution_time=round(execution_time, 3))
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error("Function execution failed", 
                           function=func.__name__,
                           execution_time=round(execution_time, 3),
                           error=str(e))
                raise
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator