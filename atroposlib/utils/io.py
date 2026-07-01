import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def parse_http_response(
    resp: Any, logger: Optional[logging.Logger] = None
) -> Any:
    """
    Parse an HTTP response with proper error handling and logging.

    Args:
        resp: The HTTP response object (must have raise_for_status() and json() methods)
        logger: Optional logger instance. If not provided, uses the default module logger.

    Returns:
        The parsed JSON response

    Raises:
        Exception: Re-raises any exceptions that occur during parsing
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Raise an exception for bad status codes (4xx or 5xx)
        resp.raise_for_status()
        # Attempt to parse the response as JSON
        return await resp.json()
    except Exception as e:
        # Handle HTTP errors (raised by raise_for_status)
        try:
            # Reading the body for logging can itself fail: raise_for_status()
            # releases the connection before raising, so on a large/streamed or
            # interrupted error response resp.text() raises (e.g.
            # ClientPayloadError/ClientConnectionError). Left unguarded it runs
            # before logger.error and replaces the original error, dropping the
            # log line and hiding the HTTP status from status-based retries.
            error_text = await resp.text()
        except Exception:
            error_text = "<response body unavailable>"
        logger.error(
            f"Error fetching from server. Status: {getattr(e, 'status', 'unknown')}, "
            f"Message: {getattr(e, 'message', str(e))}. Response: {error_text}"
        )
        # Re-raise the original exception to allow retry decorators to handle it
        raise
