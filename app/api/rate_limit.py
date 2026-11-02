
from collections import deque
from math import ceil
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request


# Maximum requests allowed per client within the time window.
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

# Store request timestamps for each client IP.
_request_timestamps: dict[str, deque[float]] = {}

# Protect shared state when multiple requests arrive concurrently.
_rate_limit_lock = Lock()


def enforce_ask_rate_limit(request: Request) -> None:
    """
    Enforce a rolling-window rate limit for the /ask endpoint.
    """

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    current_time = monotonic()

    with _rate_limit_lock:

        # Remove expired timestamps and inactive client entries.
        for ip, timestamps in list(_request_timestamps.items()):
            while (
                timestamps
                and current_time - timestamps[0]
                >= RATE_LIMIT_WINDOW_SECONDS
            ):
                timestamps.popleft()

            if not timestamps:
                del _request_timestamps[ip]

        timestamps = _request_timestamps.setdefault(
            client_ip,
            deque(),
        )

        # Reject requests exceeding the configured limit.
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:

            retry_after = max(
                1,
                ceil(
                    RATE_LIMIT_WINDOW_SECONDS
                    - (current_time - timestamps[0])
                ),
            )

            raise HTTPException(
                status_code=429,
                detail=(
                    "Too many requests. Please wait before "
                    "submitting another database question."
                ),
                headers={
                    "Retry-After": str(retry_after)
                },
            )

        # Record the accepted request.
        timestamps.append(current_time)
