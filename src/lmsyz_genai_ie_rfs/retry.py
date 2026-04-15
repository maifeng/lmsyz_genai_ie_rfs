"""Retry decorator: tenacity-based retry logic for LLM API calls.

Wraps callables with exponential backoff and retries on transient API errors
from both OpenAI and Anthropic SDKs.

Input: any callable that may raise openai.RateLimitError, openai.APIError,
       or anthropic.RateLimitError.
Output: the wrapped callable with automatic retry behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import anthropic
import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

_RETRYABLE = (
    openai.RateLimitError,
    openai.APIError,
    anthropic.RateLimitError,
)


def retry_api_call(func: F) -> F:
    """Decorate a function with exponential-backoff retry for LLM API errors.

    Retries up to 5 times with exponential backoff between 2 s and 30 s.
    Logs a warning before each sleep so callers can observe retry activity.

    Retryable exceptions:
        - openai.RateLimitError
        - openai.APIError (covers 5xx transients)
        - anthropic.RateLimitError

    Args:
        func: The callable to wrap.

    Returns:
        The wrapped callable with retry logic applied.

    Example:
        @retry_api_call
        def call_api(client, messages):
            return client.chat.completions.create(...)
    """
    decorated: F = retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )(func)
    return decorated
