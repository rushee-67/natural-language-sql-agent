from langchain_core import messages
from sqlalchemy import update
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END

from app.agent.graph import (
    MAX_TOOL_CALLS_PER_REQUEST,
    check_tool_budget,
    route_after_agent,
    route_after_tool_budget,
)

import app.agent.graph as graph_module
from app.agent.recovery import inspect_tool_results


class FakeModelWithTools:
    """Simulates LLM responses for a deterministic graph test."""

    def __init__(self):
        query = (
            "SELECT name FROM customers "
            "ORDER BY customer_id LIMIT 2"
        )

        self.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_schema",
                        "args": {},
                        "id": "test_schema",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validate_sql",
                        "args": {"query": query},
                        "id": "test_validate",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_sql",
                        "args": {"query": query},
                        "id": "test_execute",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="The query returned two customer names."
            ),
        ]

    def invoke(self, messages):
        return self.responses.pop(0)


class FakeChatGroq:
    """Replaces the real LLM client during automated testing."""

    def __init__(self, **kwargs):
        pass

    def bind_tools(self, tools):
        return FakeModelWithTools()


def test_agent_executes_tools_and_returns_final_answer(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        FakeChatGroq,
    )

    agent = graph_module.build_graph()

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Show me the first two customers."
                )
            ]
        }
    )

    messages = result["messages"]

    # Check the final answer.
    assert messages[-1].content == (
        "The query returned two customer names."
    )

    # Verify that all three tools actually executed.
    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    tool_results = {
        message.name: message.content
        for message in tool_messages
    }

    assert set(tool_results) == {
        "get_schema",
        "validate_sql",
        "execute_sql",
    }

    # Verify the actual outputs returned by the tools.
    assert '"customers"' in tool_results["get_schema"]

    validation = json.loads(tool_results["validate_sql"])
    assert validation["valid"] is True

    execution = json.loads(tool_results["execute_sql"])
    assert execution["success"] is True
    assert execution["row_count"] == 2


def make_tool_state(
    tool_name: str,
    result: dict,
    retry_count: int = 0,
) -> dict:
    """Create a small state containing one tool result."""

    return {
        "messages": [
            HumanMessage(content="Test database question."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {},
                        "id": "test-call",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=json.dumps(result),
                tool_call_id="test-call",
                name=tool_name,
            ),
        ],
        "retry_count": retry_count,
    }


def test_inspector_records_validation_failure():
    state = make_tool_state(
        "validate_sql",
        {
            "valid": False,
            "message": "Only SELECT queries are allowed.",
        },
    )

    update = inspect_tool_results(state)

    assert update["retry_count"] == 1
    assert update["recovery_exhausted"] is False
    assert "validation failed" in update["last_error"].lower()


def test_inspector_records_execution_failure():
    state = make_tool_state(
        "execute_sql",
        {
            "success": False,
            "stage": "execution",
            "error": "no such column: customer_name",
        },
    )

    update = inspect_tool_results(state)

    assert update["retry_count"] == 1
    assert "customer_name" in update["last_error"]


def test_inspector_ignores_successful_execution():
    state = make_tool_state(
        "execute_sql",
        {
            "success": True,
            "columns": ["name"],
            "rows": [{"name": "Rahul"}],
            "row_count": 1,
            "truncated": False,
        },
    )

    update = inspect_tool_results(state)

    assert update == {}


def test_inspector_stops_after_recovery_limit():
    state = make_tool_state(
        "execute_sql",
        {
            "success": False,
            "stage": "execution",
            "error": "no such column: customer_name",
        },
        retry_count=2,
    )

    update = inspect_tool_results(state)

    assert "retry_count" not in update
    assert update["recovery_exhausted"] is True
    assert "customer_name" in update["last_error"]

def test_agent_stops_after_two_recovery_attempts(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class RepeatedInvalidQueryModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            # Deliberately request invalid SQL every time.
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validate_sql",
                        "args": {
                            "query": "DELETE FROM customers"
                        },
                        "id": f"invalid_{self.calls}",
                        "type": "tool_call",
                    }
                ],
            )

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            return RepeatedInvalidQueryModel()

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        FakeChatGroq,
    )

    agent = graph_module.build_graph()

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Show me the customer list."
                )
            ]
        },
        config={
    "recursion_limit": graph_module.MAX_GRAPH_STEPS_PER_REQUEST
},
    )

    validation_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == "validate_sql"
    ]

    # One initial failure plus two recovery attempts.
    assert len(validation_messages) == 3

    assert result["retry_count"] == 2
    assert result["recovery_exhausted"] is True

    assert "two recovery attempts" in (
        result["messages"][-1].content
    )

def test_successful_validation_preserves_execution_error():
    state = make_tool_state(
        "validate_sql",
        {
            "valid": True,
            "message": "SQL passed the initial validation checks.",
            "normalized_sql": (
                "SELECT name FROM customers LIMIT 1"
            ),
        },
        retry_count=1,
    )

    state["last_error"] = (
        "SQL execution failed: no such column: customer_name"
    )

    update = inspect_tool_results(state)

    assert update == {}
    assert state["retry_count"] == 1
    assert "customer_name" in state["last_error"]

def test_agent_recovers_from_execution_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    bad_query = (
        "SELECT customer_name FROM customers LIMIT 2"
    )
    corrected_query = (
        "SELECT name FROM customers "
        "ORDER BY customer_id LIMIT 2"
    )

    def tool_call(name, args, call_id):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )

    class FakeRecoveryModel:
        def __init__(self):
            self.responses = [
                tool_call("get_schema", {}, "schema_1"),

                tool_call(
                    "validate_sql",
                    {"query": bad_query},
                    "validate_bad",
                ),

                tool_call(
                    "execute_sql",
                    {"query": bad_query},
                    "execute_bad",
                ),

                tool_call(
                    "validate_sql",
                    {"query": corrected_query},
                    "validate_corrected",
                ),

                tool_call(
                    "execute_sql",
                    {"query": corrected_query},
                    "execute_corrected",
                ),

                AIMessage(
                    content=(
                        "I corrected the column name and "
                        "successfully retrieved two customers."
                    )
                ),
            ]

        def invoke(self, messages):
            # Verify the LLM receives a sanitized, actionable
            # database error before attempting correction.
            if len(self.responses) == 3:
                assert any(
                    isinstance(message, ToolMessage)
                    and message.name == "execute_sql"
                    and "column that does not exist"
                    in message.content
                    for message in messages
                )

            if not self.responses:
                raise AssertionError(
                    "The graph requested more model responses "
                    "than expected."
                )

            return self.responses.pop(0)

    class FakeChatGroq:
        def __init__(self, **kwargs):
            self.fake_model = FakeRecoveryModel()

        def bind_tools(self, tools):
            return self.fake_model

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        FakeChatGroq,
    )

    agent = graph_module.build_graph()

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Show me two customers."
                )
            ]
        },
        config={
    "recursion_limit": graph_module.MAX_GRAPH_STEPS_PER_REQUEST
},
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    execution_results = [
        json.loads(message.content)
        for message in tool_messages
        if message.name == "execute_sql"
    ]

    # First execution fails because the column does not exist.
    assert len(execution_results) == 2
    assert execution_results[0]["success"] is False
    assert (
        execution_results[0]["stage"] == "execution"
    )

    # The corrected query succeeds against the real database.
    assert execution_results[1]["success"] is True
    assert execution_results[1]["row_count"] == 2

    # Exactly one recovery attempt was used.
    assert result["retry_count"] == 1
    assert result["last_error"] == ""
    assert result["recovery_exhausted"] is False

    # The graph returns a final answer.
    assert "successfully retrieved two customers" in (
        result["messages"][-1].content
    )

def test_successful_execution_clears_stale_error():
    state = make_tool_state(
        "execute_sql",
        {
            "success": True,
            "columns": ["name"],
            "rows": [{"name": "Rahul"}],
            "row_count": 1,
            "truncated": False,
        },
        retry_count=1,
    )

    state["last_error"] = "Previous SQL execution failed."
    state["recovery_exhausted"] = True

    update = inspect_tool_results(state)

    assert update["last_error"] == ""
    assert update["recovery_exhausted"] is False
    assert "retry_count" not in update


def make_tool_calls(count):
    """Create mock tool calls for testing."""

    return [
        {
            "name": "get_schema",
            "args": {},
            "id": f"tool-call-{i}",
            "type": "tool_call",
        }
        for i in range(count)
    ]


def test_tool_budget_counts_requested_calls():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=make_tool_calls(3),
            )
        ]
    }

    update = check_tool_budget(state)

    assert update["tool_call_count"] == 3
    assert update["tool_budget_exhausted"] is False


def test_tool_budget_accumulates_across_iterations():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=make_tool_calls(2),
            )
        ],
        "tool_call_count": 5,
    }

    update = check_tool_budget(state)

    assert update["tool_call_count"] == 7
    assert update["tool_budget_exhausted"] is False


def test_tool_budget_allows_exact_limit():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=make_tool_calls(2),
            )
        ],
        "tool_call_count": MAX_TOOL_CALLS_PER_REQUEST - 2,
    }

    update = check_tool_budget(state)

    assert update["tool_call_count"] == MAX_TOOL_CALLS_PER_REQUEST
    assert update["tool_budget_exhausted"] is False


def test_tool_budget_rejects_calls_exceeding_limit():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=make_tool_calls(2),
            )
        ],
        "tool_call_count": MAX_TOOL_CALLS_PER_REQUEST - 1,
    }

    update = check_tool_budget(state)

    assert update["tool_budget_exhausted"] is True

    # No additional calls are reserved.
    assert "tool_call_count" not in update

    updated_state = {**state, **update}

    assert route_after_tool_budget(updated_state) == "limit_exceeded"


def test_tool_budget_allows_valid_execution():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=make_tool_calls(2),
            )
        ],
        "tool_call_count": 2,
    }

    update = check_tool_budget(state)

    updated_state = {**state, **update}

    assert route_after_tool_budget(updated_state) == "tools"


def test_agent_routes_tool_requests_to_budget_check():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=make_tool_calls(1),
            )
        ]
    }

    assert route_after_agent(state) == "check_tool_budget"


def test_agent_routes_final_answers_to_end():
    state = {
        "messages": [
            AIMessage(content="The query returned five products.")
        ]
    }

    assert route_after_agent(state) == END


def test_agent_stops_after_tool_call_budget_is_exhausted(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    execution_count = {"value": 0}

    @tool
    def get_schema():
        """Return a minimal database schema."""
        execution_count["value"] += 1

        return json.dumps({
            "tables": {}
        })

    class FakeModel:
        def __init__(self):
            self.invocation_count = 0

        def invoke(self, messages):
            self.invocation_count += 1

            # Always request another schema inspection.
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_schema",
                        "args": {},
                        "id": f"schema-{self.invocation_count}",
                        "type": "tool_call",
                    }
                ],
            )

    fake_model = FakeModel()

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            return fake_model

    # Replace the actual tools and LLM with test doubles.
    monkeypatch.setattr(
        graph_module,
        "TOOLS",
        [get_schema],
    )

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        FakeChatGroq,
    )

    agent = graph_module.build_graph()

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Inspect the database schema."
                )
            ]
        },
        config={
            "recursion_limit": (
                graph_module.MAX_GRAPH_STEPS_PER_REQUEST
            )
        },
    )

    # Exactly 12 tool calls should have executed.
    assert execution_count["value"] == (
        graph_module.MAX_TOOL_CALLS_PER_REQUEST
    )

    # The model requested one additional call.
    assert fake_model.invocation_count == (
        graph_module.MAX_TOOL_CALLS_PER_REQUEST + 1
    )

    # The state records the exhausted budget.
    assert result["tool_call_count"] == (
        graph_module.MAX_TOOL_CALLS_PER_REQUEST
    )

    assert result["tool_budget_exhausted"] is True

    # Verify that the agent returned a controlled response.
    final_message = result["messages"][-1]

    assert isinstance(final_message, AIMessage)
    assert "maximum tool-call limit" in final_message.content


def test_chatgroq_receives_timeout_and_retry_configuration(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_REQUEST_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "2")

    captured = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        FakeChatGroq,
    )

    graph_module.build_graph()

    assert captured["request_timeout"] == 30.0
    assert captured["max_retries"] == 2
    assert captured["temperature"] == 0


def test_chatgroq_rejects_invalid_timeout(monkeypatch):
    import pytest

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_REQUEST_TIMEOUT_SECONDS", "-1")

    with pytest.raises(
        RuntimeError,
        match="timeout must be positive",
    ):
        graph_module.build_graph()


def test_chatgroq_rejects_negative_retries(monkeypatch):
    import pytest

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "-1")

    with pytest.raises(
        RuntimeError,
        match="timeout must be positive",
    ):
        graph_module.build_graph()
