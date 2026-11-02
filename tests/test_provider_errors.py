
import groq
import httpx
import pytest

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

import app.agent.graph as graph_module
import app.api.routes as routes

from app.agent.provider_errors import (
    LLMProviderError,
    classify_groq_exception,
)
from app.main import app


client = TestClient(app)


def make_request():
    return httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )


def make_status_error(error_class, status_code):
    request = make_request()

    response = httpx.Response(
        status_code=status_code,
        request=request,
    )

    return error_class(
        "private provider error details",
        response=response,
        body={"detail": "sensitive internal information"},
    )


@pytest.mark.parametrize(
    "error_class,status_code,expected_category,expected_http",
    [
        (
            groq.AuthenticationError,
            401,
            "provider_configuration",
            500,
        ),
        (
            groq.PermissionDeniedError,
            403,
            "provider_configuration",
            500,
        ),
        (
            groq.RateLimitError,
            429,
            "provider_rate_limit",
            503,
        ),
        (
            groq.APIStatusError,
            408,
            "provider_timeout",
            504,
        ),
        (
            groq.APIStatusError,
            409,
            "provider_unavailable",
            503,
        ),
        (
            groq.APIStatusError,
            500,
            "provider_unavailable",
            503,
        ),
        (
            groq.APIStatusError,
            400,
            "provider_request_rejected",
            502,
        ),
    ],
)
def test_classify_groq_status_errors(
    error_class,
    status_code,
    expected_category,
    expected_http,
):
    original_error = make_status_error(
        error_class,
        status_code,
    )

    result = classify_groq_exception(original_error)

    assert isinstance(result, LLMProviderError)
    assert result.category == expected_category
    assert result.http_status_code == expected_http
    assert result.provider_status_code == status_code

    # Original provider details must not enter the safe message.
    assert "private provider error details" not in result.safe_message
    assert "sensitive internal information" not in result.safe_message


def test_classify_groq_timeout_error():
    original_error = groq.APITimeoutError(
        request=make_request()
    )

    result = classify_groq_exception(original_error)

    assert result.category == "provider_timeout"
    assert result.http_status_code == 504


def test_classify_groq_connection_error():
    original_error = groq.APIConnectionError(
        request=make_request()
    )

    result = classify_groq_exception(original_error)

    assert result.category == "provider_connection"
    assert result.http_status_code == 502


def test_unrelated_exception_is_not_classified():
    result = classify_groq_exception(
        ValueError("Application error")
    )

    assert result is None


def test_graph_wraps_groq_provider_timeout(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeModel:
        def invoke(self, messages):
            raise groq.APITimeoutError(
                request=make_request()
            )

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            return FakeModel()

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        FakeChatGroq,
    )

    agent = graph_module.build_graph()

    with pytest.raises(LLMProviderError) as captured:
        agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Show the top products."
                    )
                ]
            },
            config={
                "recursion_limit": (
                    graph_module.MAX_GRAPH_STEPS_PER_REQUEST
                )
            },
        )

    assert captured.value.category == "provider_timeout"
    assert captured.value.http_status_code == 504
    assert isinstance(
        captured.value.__cause__,
        groq.APITimeoutError,
    )


@pytest.mark.parametrize(
    "category,message,status_code",
    [
        (
            "provider_timeout",
            "The model provider timed out.",
            504,
        ),
        (
            "provider_rate_limit",
            "The model provider is rate-limited.",
            503,
        ),
        (
            "provider_connection",
            "The model provider cannot be reached.",
            502,
        ),
        (
            "provider_configuration",
            "The provider is not configured correctly.",
            500,
        ),
    ],
)
def test_api_returns_structured_provider_error(
    monkeypatch,
    category,
    message,
    status_code,
):
    class FakeAgent:
        def invoke(self, state, config=None):
            raise LLMProviderError(
                category=category,
                safe_message=message,
                http_status_code=status_code,
                provider_exception_type="MockProviderError",
            )

    monkeypatch.setattr(
        routes,
        "get_agent",
        lambda: FakeAgent(),
    )

    response = client.post(
        "/api/ask",
        json={"question": "Show the top five products."},
    )

    assert response.status_code == status_code
    assert response.json()["detail"] == message
