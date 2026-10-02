"""Verify provider retry policy using synthetic errors without requests or sleeps.

Input: SDK exception instances. Output: assertions on retry decisions and actual
outer attempt counts for a wrapped callable.
"""

from __future__ import annotations

from typing import Any

import anthropic
import httpx
import openai
import pytest
from tenacity import wait_none

from lmsyz_genai_ie_rfs.retry import _is_retryable, retry_api_call


def _openai_error(status: int, body: dict[str, Any] | None = None) -> openai.APIStatusError:
    """Build an OpenAI status error without network access.

    Args:
        status: HTTP status code.
        body: Optional error response JSON.

    Returns:
        SDK exception with the requested status and details.
    """
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    return openai.APIStatusError("test failure", response=response, body=body)


@pytest.mark.parametrize("status,expected", [(400, False), (401, False), (403, False), (404, False),
                                            (408, True), (409, True), (422, False), (429, True),
                                            (500, True), (503, True)])
def test_openai_status_policy(status: int, expected: bool) -> None:
    """Retry only transient HTTP statuses.

    Args:
        status: Simulated status code.
        expected: Expected outer retry decision.
    """
    assert _is_retryable(_openai_error(status)) is expected


@pytest.mark.parametrize("body", [
    {"code": "insufficient_quota"},
    {"type": "insufficient_quota"},
    {"error": {"code": "insufficient_quota"}},
    {"code": "billing_hard_limit_reached"},
    {"code": "credits_exhausted"},
    {"code": "credit_balance_exhausted"},
])
def test_quota_failures_are_not_retried(body: dict[str, Any]) -> None:
    """Distinguish exhausted credits from temporary rate limits.

    Args:
        body: Permanent billing failure details.
    """
    assert not _is_retryable(_openai_error(429, body))


def test_connection_errors_are_retryable() -> None:
    """Keep connection and timeout retries enabled."""
    request = httpx.Request("POST", "https://example.invalid")
    assert _is_retryable(openai.APIConnectionError(request=request))
    assert _is_retryable(openai.APITimeoutError(request=request))
    assert not _is_retryable(ValueError("invalid JSON"))


def test_anthropic_server_errors_rely_on_sdk_retries() -> None:
    """Retain Anthropic rate-limit policy without adding server retry layers."""
    request = httpx.Request("POST", "https://example.invalid")
    limited = anthropic.RateLimitError("limit", response=httpx.Response(429, request=request), body=None)
    server = anthropic.InternalServerError("server", response=httpx.Response(500, request=request), body=None)
    assert _is_retryable(limited)
    assert not _is_retryable(server)


@pytest.mark.parametrize("status,body,expected_attempts", [
    (401, None, 1), (400, None, 1), (429, {"code": "insufficient_quota"}, 1),
    (429, {"code": "rate_limit_exceeded"}, 5), (503, None, 5),
])
def test_outer_attempt_budget(status: int, body: dict[str, Any] | None, expected_attempts: int) -> None:
    """Verify the decorator enforces the retry predicate and five-attempt limit.

    Args:
        status: Simulated HTTP status.
        body: Optional error details.
        expected_attempts: Number of callable invocations expected.
    """
    attempts = 0

    @retry_api_call
    def fail() -> None:
        """Raise the selected SDK exception and count attempts.

        Raises:
            openai.APIStatusError: Synthetic provider error.
        """
        nonlocal attempts
        attempts += 1
        raise _openai_error(status, body)

    wrapped: Any = fail
    wrapped.retry.wait = wait_none()
    with pytest.raises(openai.APIStatusError):
        wrapped()
    assert attempts == expected_attempts
