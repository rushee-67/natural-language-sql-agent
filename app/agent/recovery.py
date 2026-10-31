import json

from langchain_core.messages import ToolMessage

from app.agent.state import SQLAgentState


MAX_RECOVERY_ATTEMPTS = 2

KNOWN_TOOLS = {
    "get_schema",
    "validate_sql",
    "execute_sql",
}


def get_tool_error(message: ToolMessage) -> str | None:
    """
    Return an error description if the tool result indicates failure.
    Return None when the tool result indicates success.
    """

    tool_name = message.name or "unknown_tool"
    content = message.content

    # Handle exceptions reported by LangChain's tool execution layer.
    if getattr(message, "status", None) == "error":
        return f"{tool_name} failed: {content}"

    # Our three tools are expected to return JSON strings.
    if not isinstance(content, str):
        if tool_name in KNOWN_TOOLS:
            return f"{tool_name} returned an unexpected output format."
        return None

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        if tool_name in KNOWN_TOOLS:
            return f"{tool_name} returned invalid JSON."
        return None

    if not isinstance(result, dict):
        if tool_name in KNOWN_TOOLS:
            return f"{tool_name} returned an unexpected JSON structure."
        return None

    # Interpret validation-tool results.
    if tool_name == "validate_sql":
        if result.get("valid") is not True:
            reason = result.get(
                "message",
                "SQL validation failed.",
            )
            return f"SQL validation failed: {reason}"

    # Interpret execution-tool results.
    elif tool_name == "execute_sql":
        if result.get("success") is not True:
            reason = result.get(
                "error",
                "SQL execution failed.",
            )
            stage = result.get("stage", "execution")
            return f"SQL {stage} failed: {reason}"

    # Validate the schema tool's expected output structure.
    elif tool_name == "get_schema":
        if not isinstance(result.get("tables"), dict):
            return "Schema tool returned an invalid schema structure."

    return None


def inspect_tool_results(state: SQLAgentState) -> dict:
    """
    Inspect the most recent batch of tool results and update
    recovery state when a tool reports an error.
    """

    messages = state.get("messages", [])
    latest_tool_messages = []

    # Inspect only the consecutive ToolMessages at the end of
    # the conversation. Earlier tool failures must not be counted
    # again during a later graph iteration.
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            latest_tool_messages.append(message)
        else:
            break

    if not latest_tool_messages:
        return {}

    latest_tool_messages.reverse()

    errors = []

    for message in latest_tool_messages:
        error = get_tool_error(message)

        if error is not None:
            errors.append(error)

    # There is no new failure to record.
    if not errors:
    # A successful validation does not mean SQL execution succeeded.
        successful_execution = any(
            message.name == "execute_sql"
            for message in latest_tool_messages
        )

        # Clear the old error only after successful query execution.
        if (
            successful_execution
            and (
                state.get("last_error")
                or state.get("recovery_exhausted", False)
            )
        ):
            return {
                "last_error": "",
                "recovery_exhausted": False,
            }

        # Preserve previous error information after intermediate successes.
        return {}

    last_error = "; ".join(errors)[:2000]
    retry_count = state.get("retry_count", 0)

    # All allowed recovery attempts have already been granted.
    if retry_count >= MAX_RECOVERY_ATTEMPTS:
        return {
            "last_error": last_error,
            "recovery_exhausted": True,
        }

    # Grant one recovery attempt. Routing will be added in Phase 4.3.
    return {
        "last_error": last_error,
        "retry_count": retry_count + 1,
        "recovery_exhausted": False,
    }