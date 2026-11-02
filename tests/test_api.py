import app.database.models
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from langchain_core.messages import AIMessage, ToolMessage

from app.api.routes import AskRequest
from app.main import app
import app.api.routes as routes
import app.api.rate_limit as rate_limit


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_schema_endpoint():
    response = client.get("/api/schema")

    assert response.status_code == 200

    schema = response.json()

    assert schema["database"] == "sqlite"
    assert set(schema["tables"]) == {
        "customers",
        "products",
        "orders",
        "order_items",
    }


def test_request_strips_surrounding_whitespace():
    request = AskRequest(
        question="  Show me the customers  "
    )

    assert request.question == "Show me the customers"


def test_request_rejects_blank_question():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_request_rejects_question_over_limit():
    with pytest.raises(ValidationError):
        AskRequest(question="a" * 2001)

def test_ask_returns_answer_and_query_results(monkeypatch):
    query = (
        "SELECT product_name, revenue "
        "FROM product_revenue ORDER BY revenue DESC LIMIT 1"
    )

    execution_payload = {
        "success": True,
        "columns": ["product_name", "revenue"],
        "rows": [{"product_name": "Keyboard", "revenue": 500}],
        "row_count": 1,
        "truncated": False,
    }

    class FakeAgent:
        def invoke(self, state, config=None):
            assert config["recursion_limit"] == (
                routes.MAX_GRAPH_STEPS_PER_REQUEST
            )
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "execute_sql",
                                "args": {"query": query},
                                "id": "execute-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content=json.dumps(execution_payload),
                        tool_call_id="execute-1",
                        name="execute_sql",
                    ),
                    AIMessage(
                        content="The Keyboard generated revenue of 500."
                    ),
                ]
            }

    monkeypatch.setattr(routes, "get_agent", lambda: FakeAgent())

    response = client.post(
        "/api/ask",
        json={"question": "Which product generated the most revenue?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "The Keyboard generated revenue of 500."
    assert data["sql"] == query
    assert data["columns"] == ["product_name", "revenue"]
    assert data["rows"] == [
        {"product_name": "Keyboard", "revenue": 500}
    ]
    assert data["row_count"] == 1
    assert data["truncated"] is False


def test_ask_rejects_blank_question():
    response = client.post(
        "/api/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_returns_502_when_agent_raises(monkeypatch):
    class BrokenAgent:
        def invoke(self, state, config=None):
            raise RuntimeError("Internal provider error")

    monkeypatch.setattr(routes, "get_agent", lambda: BrokenAgent())

    response = client.post(
        "/api/ask",
        json={"question": "Show the top five products"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The SQL agent could not process the question. Please try again."
    )

def test_frontend_homepage_is_served():
    response = client.get("/")

    assert response.status_code == 200
    assert "Data Assistant" in response.text


def test_ask_without_sql_execution_returns_success_false(monkeypatch):

    class FakeAgent:
        def invoke(self, state, config=None):
            return {
                "messages": [
                    AIMessage(
                        content="The database contains four tables."
                    )
                ]
            }

    monkeypatch.setattr(
        routes,
        "get_agent",
        lambda: FakeAgent(),
    )

    response = client.post(
        "/api/ask",
        json={"question": "What tables are available?"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is False


def test_ask_tool_budget_exhaustion_returns_success_false(monkeypatch):

    execution_payload = {
        "success": True,
        "columns": ["product_name"],
        "rows": [{"product_name": "Keyboard"}],
        "row_count": 1,
        "truncated": False,
    }

    class FakeAgent:
        def invoke(self, state, config=None):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "execute_sql",
                                "args": {"query": "SELECT 1"},
                                "id": "execute-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content=json.dumps(execution_payload),
                        tool_call_id="execute-1",
                        name="execute_sql",
                    ),
                    AIMessage(
                        content="The tool-call budget was exceeded."
                    ),
                ],
                "tool_budget_exhausted": True,
            }

    monkeypatch.setattr(
        routes,
        "get_agent",
        lambda: FakeAgent(),
    )

    response = client.post(
        "/api/ask",
        json={"question": "Show the top products."},
    )

    assert response.status_code == 200
    assert response.json()["success"] is False


def test_ask_rate_limit_blocks_excessive_requests(monkeypatch):
    # Configure a small limit for this test.
    monkeypatch.setattr(
        rate_limit,
        "RATE_LIMIT_MAX_REQUESTS",
        1,
    )

    monkeypatch.setattr(
        rate_limit,
        "RATE_LIMIT_WINDOW_SECONDS",
        60,
    )

    # Clear previous test request history.
    rate_limit._request_timestamps.clear()

    execution_payload = {
        "success": True,
        "columns": ["value"],
        "rows": [{"value": 1}],
        "row_count": 1,
        "truncated": False,
    }

    class FakeAgent:
        def invoke(self, state, config=None):
            return {
                "messages": [
                    ToolMessage(
                        content=json.dumps(execution_payload),
                        tool_call_id="execute-1",
                        name="execute_sql",
                    ),
                    AIMessage(content="The query returned one row."),
                ]
            }

    monkeypatch.setattr(
        routes,
        "get_agent",
        lambda: FakeAgent(),
    )

    try:
        # First request should succeed.
        first_response = client.post(
            "/api/ask",
            json={"question": "Show one result."},
        )

        assert first_response.status_code == 200

        # Second request should be rejected.
        second_response = client.post(
            "/api/ask",
            json={"question": "Show another result."},
        )

        assert second_response.status_code == 429

        assert "Retry-After" in second_response.headers

        assert "Too many requests" in (
            second_response.json()["detail"]
        )

    finally:
        rate_limit._request_timestamps.clear()
