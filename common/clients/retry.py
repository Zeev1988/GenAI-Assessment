"""Shared tenacity retry policy for Azure client calls."""

from __future__ import annotations

from typing import Any

import httpx
from azure.core.exceptions import (
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from common.logging_config import get_logger

_log = get_logger(__name__)

# Exception types that are worth retrying on. 4xx client errors (auth, bad
# request) are intentionally excluded so that we fail fast.
_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    ServiceRequestError,
    ServiceResponseError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    if isinstance(exc, HttpResponseError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    _log.warning(
        "azure.retry",
        attempt=state.attempt_number,
        next_sleep_s=round(state.next_action.sleep, 3) if state.next_action else None,
        exc_type=type(exc).__name__ if exc else None,
        exc_message=str(exc) if exc else None,
    )


def azure_retry(*, max_attempts: int = 5, max_wait_s: float = 20.0) -> AsyncRetrying:
    """Return a configured :class:`AsyncRetrying` for Azure service calls."""
    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=0.5, max=max_wait_s, jitter=0.5),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
    )


async def run_with_retry(
    func: Any,
    *args: Any,
    max_attempts: int = 5,
    max_wait_s: float = 20.0,
    **kwargs: Any,
) -> Any:
    """Invoke an async callable under the shared retry policy."""
    async for attempt in azure_retry(max_attempts=max_attempts, max_wait_s=max_wait_s):
        with attempt:
            return await func(*args, **kwargs)
    raise RuntimeError("azure_retry exited without producing a result")
