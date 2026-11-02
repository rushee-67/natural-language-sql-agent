
import groq


class LLMProviderError(Exception):
    """
    Application-level exception representing an LLM provider failure.

    Contains safe information for API responses without exposing
    the original provider error to the client.
    """

    def __init__(
        self,
        *,
        category: str,
        safe_message: str,
        http_status_code: int,
        provider_exception_type: str,
        provider_status_code: int | None = None,
    ):
        super().__init__(safe_message)

        self.category = category
        self.safe_message = safe_message
        self.http_status_code = http_status_code
        self.provider_exception_type = provider_exception_type
        self.provider_status_code = provider_status_code


def classify_groq_exception(
    exc: Exception,
) -> LLMProviderError | None:
    """
    Convert known Groq SDK exceptions into application-level errors.

    Returns None for exceptions that are not Groq provider errors.
    """

    exception_type = type(exc).__name__

    # A timeout is also a connection error, so check it first.
    if isinstance(exc, groq.APITimeoutError):
        return LLMProviderError(
            category="provider_timeout",
            safe_message=(
                "The language model provider did not respond "
                "in time. Please try again."
            ),
            http_status_code=504,
            provider_exception_type=exception_type,
        )

    # Invalid API key or insufficient provider permissions.
    if isinstance(
        exc,
        (
            groq.AuthenticationError,
            groq.PermissionDeniedError,
        ),
    ):
        return LLMProviderError(
            category="provider_configuration",
            safe_message=(
                "The language model provider is not configured "
                "correctly. Please contact the application administrator."
            ),
            http_status_code=500,
            provider_exception_type=exception_type,
            provider_status_code=getattr(exc, "status_code", None),
        )

    # Provider rate limiting.
    if isinstance(exc, groq.RateLimitError):
        return LLMProviderError(
            category="provider_rate_limit",
            safe_message=(
                "The language model service is temporarily "
                "rate-limited. Please try again shortly."
            ),
            http_status_code=503,
            provider_exception_type=exception_type,
            provider_status_code=getattr(exc, "status_code", 429),
        )

    # Network or connectivity failure.
    if isinstance(exc, groq.APIConnectionError):
        return LLMProviderError(
            category="provider_connection",
            safe_message=(
                "Unable to connect to the language model service. "
                "Please try again."
            ),
            http_status_code=502,
            provider_exception_type=exception_type,
        )

    # Handle remaining HTTP status errors.
    if isinstance(exc, groq.APIStatusError):
        status_code = exc.status_code

        if status_code == 408:
            category = "provider_timeout"
            message = (
                "The language model provider did not respond "
                "in time. Please try again."
            )
            http_status = 504

        elif status_code == 409:
            category = "provider_unavailable"
            message = (
                "The language model service is temporarily "
                "unavailable. Please try again shortly."
            )
            http_status = 503

        elif status_code >= 500:
            category = "provider_unavailable"
            message = (
                "The language model service is temporarily "
                "unavailable. Please try again later."
            )
            http_status = 503

        else:
            category = "provider_request_rejected"
            message = (
                "The language model provider rejected the request. "
                "Please contact the application administrator."
            )
            http_status = 502

        return LLMProviderError(
            category=category,
            safe_message=message,
            http_status_code=http_status,
            provider_exception_type=exception_type,
            provider_status_code=status_code,
        )

    # Handle other Groq SDK errors, such as invalid provider responses.
    if isinstance(exc, groq.APIError):
        return LLMProviderError(
            category="provider_error",
            safe_message=(
                "The language model service returned an unexpected "
                "error. Please try again."
            ),
            http_status_code=502,
            provider_exception_type=exception_type,
        )

    # Do not convert unrelated application exceptions.
    return None
