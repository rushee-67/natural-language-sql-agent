
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.prompts import SQL_AGENT_SYSTEM_PROMPT
from app.agent.state import SQLAgentState
from app.tools.schema_tool import get_schema
from app.tools.validation_tool import validate_sql
from app.tools.sql_tool import execute_sql
from app.agent.recovery import inspect_tool_results
from app.agent.provider_errors import classify_groq_exception


MAX_TOOL_CALLS_PER_REQUEST = 12

# Maximum LangGraph node executions per request.
MAX_GRAPH_STEPS_PER_REQUEST = 60

DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_GROQ_MAX_RETRIES = 2

# All tools available to the LLM.
TOOLS = [
    get_schema,
    validate_sql,
    execute_sql,
]


def route_after_agent(state: SQLAgentState) -> str:
    """
    Decide whether the agent requested tools or produced
    its final response.
    """

    messages = state.get("messages", [])

    if not messages:
        return END

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "check_tool_budget"

    return END


def check_tool_budget(state: SQLAgentState) -> dict:
    """
    Check and reserve tool-call budget before ToolNode executes.

    The complete requested batch is either allowed or rejected.
    """

    messages = state.get("messages", [])

    if not messages:
        return {}

    last_message = messages[-1]
    requested_calls = len(
        getattr(last_message, "tool_calls", [])
    )

    current_count = state.get("tool_call_count", 0)

    # No tool calls to execute.
    if requested_calls == 0:
        return {}

    proposed_count = current_count + requested_calls

    # Reject the complete batch if it exceeds the budget.
    if proposed_count > MAX_TOOL_CALLS_PER_REQUEST:
        return {
            "tool_budget_exhausted": True,
        }

    # Reserve calls before they reach ToolNode.
    return {
        "tool_call_count": proposed_count,
        "tool_budget_exhausted": False,
    }


def route_after_tool_budget(state: SQLAgentState) -> str:
    """Choose whether to execute tools or stop the workflow."""

    if state.get("tool_budget_exhausted", False):
        return "limit_exceeded"

    return "tools"


def tool_budget_failed(state: SQLAgentState) -> dict:
    """Return a final response when the tool-call budget is exhausted."""

    return {
        "messages": [
            AIMessage(
                content=(
                    "I've reached the maximum tool-call limit of "
                    f"{MAX_TOOL_CALLS_PER_REQUEST} for this request. "
                    "The latest batch of tool calls was not executed. "
                    "Please narrow your question and try again."
                )
            )
        ]
    }


def route_after_inspection(state: SQLAgentState) -> str:
    """Decide whether to continue or stop after inspecting tool results."""

    if state.get("recovery_exhausted", False):
        return "failure"

    return "agent"


def recovery_failed(state: SQLAgentState) -> dict:
    """Return a final response after recovery attempts are exhausted."""

    return {
        "messages": [
            AIMessage(
                content=(
                    "I couldn't complete the database query after "
                    "the initial attempt and two recovery attempts. "
                    "No final query result is available. Please "
                    "rephrase the question or check the database schema."
                )
            )
        ]
    }


def build_graph():
    """Create and compile the SQL Agent workflow."""

    # Load configuration from the local .env file.
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Configure it in .env."
        )

    # Read LLM reliability configuration.
    try:
        request_timeout = float(
            os.getenv(
                "GROQ_REQUEST_TIMEOUT_SECONDS",
                str(DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS),
            )
        )

        max_retries = int(
            os.getenv(
                "GROQ_MAX_RETRIES",
                str(DEFAULT_GROQ_MAX_RETRIES),
            )
        )

    except ValueError as exc:
        raise RuntimeError(
            "Invalid Groq timeout or retry configuration."
        ) from exc

    if request_timeout <= 0 or max_retries < 0:
        raise RuntimeError(
            "Groq timeout must be positive and retries cannot be negative."
        )

    # Initialize the LLM with reliability settings.
    model = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
        request_timeout=request_timeout,
        max_retries=max_retries,
        )

    # Make the tool definitions available to the LLM.
    model_with_tools = model.bind_tools(TOOLS)

    # Agent node: call the LLM using the current message history.
    def call_model(state: SQLAgentState) -> dict:

        messages = [
            SystemMessage(content=SQL_AGENT_SYSTEM_PROMPT),
            *state["messages"],
        ]

        try:
            response = model_with_tools.invoke(messages)

        except Exception as exc:
            provider_error = classify_groq_exception(exc)

            # Preserve normal handling for unrelated application errors.
            if provider_error is None:
                raise

            # Raise our structured exception while preserving
            # the original provider exception as its cause.
            raise provider_error from exc

        return {"messages": [response]}

    # Create the graph.
    builder = StateGraph(SQLAgentState)

    # Register nodes.
    builder.add_node("agent", call_model)
    builder.add_node("check_tool_budget", check_tool_budget)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("inspect_tools", inspect_tool_results)
    builder.add_node("tool_budget_failed", tool_budget_failed)
    builder.add_node("recovery_failed", recovery_failed)

    # Begin with the agent.
    builder.add_edge(START, "agent")

    # Check whether the model requested tools.
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "check_tool_budget": "check_tool_budget",
            END: END,
        },
    )

    # Enforce the tool-call budget before execution.
    builder.add_conditional_edges(
        "check_tool_budget",
        route_after_tool_budget,
        {
            "tools": "tools",
            "limit_exceeded": "tool_budget_failed",
        },
    )

    # Stop when the tool-call budget is exceeded.
    builder.add_edge("tool_budget_failed", END)

    # Execute tools and inspect their results.
    builder.add_edge("tools", "inspect_tools")

    # Recover from tool failures when attempts remain.
    builder.add_conditional_edges(
        "inspect_tools",
        route_after_inspection,
        {
            "agent": "agent",
            "failure": "recovery_failed",
        },
    )

    builder.add_edge("recovery_failed", END)

    return builder.compile()
