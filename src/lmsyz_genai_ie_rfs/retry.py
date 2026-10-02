"""Add bounded retries to callables that raise transient provider API errors.

Input: a provider request callable. Output: a callable with at most five outer
attempts. SDK retries still apply within each attempt. OpenAI connection errors,
408/409/429 responses (except exhausted quota), and 5xx responses are retried.
Anthropic rate limits retain outer retries; other transients use SDK retries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import anthropic
import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


def _is_retryable(error: BaseException) -> bool:
    """Identify transient errors without retrying permanent OpenAI failures.

    Args:
        error: Exception raised by the wrapped callable.

    Returns:
        Whether another outer attempt is appropriate.
    """
    if isinstance(error, anthropic.RateLimitError):
        return True
    if isinstance(error, openai.APIConnectionError):
        return True
    if not isinstance(error, openai.APIStatusError):
        return False
    if error.status_code == 429:
        body = error.body if isinstance(error.body, dict) else {}
        details = body.get("error", body)
        if not isinstance(details, dict):
            details = body
        codes = {error.code, details.get("code"), details.get("type")}
        if codes & {"insufficient_quota", "billing_hard_limit_reached", "credits_exhausted", "credit_balance_exhausted"}:
            return False
    return error.status_code in {408, 409, 429} or error.status_code >= 500


def retry_api_call(func: Callable[P, R]) -> Callable[P, R]:
    """Retry transient API failures with bounded exponential backoff.

    Uses up to five outer attempts, waiting between 2 and 30 seconds before
    retries. Provider SDKs may also retry internally. Permanent OpenAI errors,
    including authentication, bad requests, and exhausted quota, receive no
    outer retry. Anthropic server errors rely on the SDK's own retry policy.

    Args:
        func: Provider request callable.

    Returns:
        Callable preserving the input signature and result type.
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )(func)
