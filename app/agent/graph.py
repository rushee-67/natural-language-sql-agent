import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


from app.agent.prompts import SQL_AGENT_SYSTEM_PROMPT
from app.agent.state import SQLAgentState
from app.tools.schema_tool import get_schema
from app.tools.validation_tool import validate_sql
from app.tools.sql_tool import execute_sql
from app.agent.recovery import inspect_tool_results


# All tools available to the LLM.
TOOLS = [
    get_schema,
    validate_sql,
    execute_sql,
]

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

    # Initialize the LLM.
    model = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )

    # Make the tool definitions available to the LLM.
    model_with_tools = model.bind_tools(TOOLS)

    # Agent node: call the LLM using the current message history.
    def call_model(state: SQLAgentState) -> dict:
        messages = [
            SystemMessage(content=SQL_AGENT_SYSTEM_PROMPT),
            *state["messages"],
        ]

        response = model_with_tools.invoke(messages)

        # Append the AI response to the graph state.
        return {"messages": [response]}

    # Create the graph using our state definition.
    builder = StateGraph(SQLAgentState)

    # Register the two nodes.
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("inspect_tools", inspect_tool_results)
    builder.add_node("recovery_failed", recovery_failed)

    # Every execution begins with the agent node.
    builder.add_edge(START, "agent")

    # Route to the tools node if the model requested a tool.
    # Otherwise, finish the workflow.
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After executing tools, return to the agent.
    # Inspect every batch of tool results.
    builder.add_edge("tools", "inspect_tools")

    # Continue if recovery is allowed; otherwise stop.
    builder.add_conditional_edges(
        "inspect_tools",
        route_after_inspection,
        {
            "agent": "agent",
            "failure": "recovery_failed",
        },
    )

    # The failure node terminates the workflow.
    builder.add_edge("recovery_failed", END)

    # Compile the graph into an executable workflow.
    return builder.compile()