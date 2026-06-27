"""
retry.py — Retry utilities for retrylib.

This module provides utilities to retry function calls upon failure. It includes
decorators and functions to automatically retry a function a specified number of
times, optionally with exponential backoff. These utilities are useful for
handling transient errors in network calls, file operations, or any other
operations that may fail intermittently.
"""

import time


def retry(max_attempts=3, exceptions=(Exception,)):
    """
    Decorator that retries a function on failure.

    Args:
        max_attempts: Maximum number of times to attempt the function.
        exceptions: Tuple of exception types to catch and retry on.

    Usage:
        @retry(max_attempts=3)
        def flaky_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts - 1):  # BUG: off-by-one — should be range(max_attempts)
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    time.sleep(0.1)
            raise last_error
        return wrapper
    return decorator


def retry_with_backoff(func, max_attempts=3, base_delay=1.0):
    """
    Retry a callable with exponential backoff.

    Args:
        func: The callable to retry.
        max_attempts: Maximum number of attempts.
        base_delay: Initial delay in seconds. Doubles each attempt.

    Returns:
        The return value of func on success.

    Raises:
        The last exception raised after all attempts are exhausted.
    """
    pass  # BUG: not implemented — should retry func with exponential backoff
