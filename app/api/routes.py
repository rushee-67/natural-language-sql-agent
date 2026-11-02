
import json
import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator

from app.agent.graph import build_graph, MAX_GRAPH_STEPS_PER_REQUEST
from app.tools.schema_tool import get_schema
from app.agent.provider_errors import LLMProviderError



router = APIRouter()
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="A question to answer using the database.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be blank.")

        return value


class AskResponse(BaseModel):
    success: bool
    answer: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)    ##default_factory=list creates a new list for each model instance instead of sharing one mutable list across instances
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False


@lru_cache(maxsize=1)
def get_agent():
    """Build the LangGraph agent once and reuse it across requests."""
    return build_graph()


def _content_to_text(content: Any) -> str:
    """Convert an assistant message's content into readable text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))

        return "\n".join(part for part in parts if part).strip()

    return ""


def _parse_tool_payload(content: Any) -> dict[str, Any]:
    """Parse a tool's JSON output into a dictionary."""
    if isinstance(content, dict):
        return content

    if not isinstance(content, str):
        return {}

    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _extract_sql(messages: list[Any]) -> str | None:
    """Find the query passed to the execute_sql tool."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue

        for tool_call in reversed(message.tool_calls or []):
            if tool_call.get("name") == "execute_sql":
                arguments = tool_call.get("args", {})

                if isinstance(arguments, dict):
                    query = arguments.get("query") or arguments.get("sql")

                    if isinstance(query, str):
                        return query

    return None


def _extract_answer(messages: list[Any]) -> str:
    """Find the final assistant response, ignoring intermediate tool calls."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            answer = _content_to_text(message.content)

            if answer:
                return answer

    return "The agent did not produce a final answer."


def _extract_execution_result(
    messages: list[Any],
) -> dict[str, Any] | None:
    """Find the most recent execute_sql result."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "execute_sql":
            return _parse_tool_payload(message.content)

    return None


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/schema")
def read_schema():
    result = get_schema.invoke({})

    if isinstance(result, str):
        return json.loads(result)

    return result


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a natural-language question using the SQL agent."""
    try:
        result = get_agent().invoke(
            {
                "messages": [
                    HumanMessage(content=request.question)
                ]
            },
            config={
                "recursion_limit": MAX_GRAPH_STEPS_PER_REQUEST
            },
        )
    except LLMProviderError as exc:
        # Log detailed diagnostic information internally.
        logger.exception(
            "LLM provider failure: category=%s, "
            "provider_exception=%s, upstream_status=%s",
            exc.category,
            exc.provider_exception_type,
            exc.provider_status_code,
        )

        # Return only the safe, categorized error to the client.
        raise HTTPException(
            status_code=exc.http_status_code,
            detail=exc.safe_message,
        ) from exc

    except Exception as exc:
        # Handle unexpected non-provider application errors.
        logger.exception(
            "The SQL agent failed while handling a question."
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The SQL agent could not process the question. "
                "Please try again."
            ),
        ) from exc

    messages = result.get("messages", [])
    execution = _extract_execution_result(messages)
    answer = _extract_answer(messages)
    sql = _extract_sql(messages)

    if execution:
        sql = (
            execution.get("query")
            or execution.get("sql")
            or execution.get("normalized_sql")
            or sql
        )

    raw_columns = execution.get("columns", []) if execution else []
    raw_rows = execution.get("rows", []) if execution else []

    columns = (
        [str(column) for column in raw_columns]
        if isinstance(raw_columns, list)
        else []
    )
    rows = (
        [row for row in raw_rows if isinstance(row, dict)]
        if isinstance(raw_rows, list)
        else []
    )

    try:
        row_count = int(execution.get("row_count", len(rows))) if execution else 0
    except (TypeError, ValueError):
        row_count = len(rows)

    # Check whether the agent exhausted its recovery attempts.
    recovery_exhausted = bool(
        result.get("recovery_exhausted", False)
    )

    # Check whether the agent exceeded its tool-call budget.
    tool_budget_exhausted = bool(
        result.get("tool_budget_exhausted", False)
    )

    # SQL execution must explicitly report success.
    execution_succeeded = (
        execution is not None
        and execution.get("success") is True
    )

    # Fail-closed success policy.
    success = (
        execution_succeeded
        and not recovery_exhausted
        and not tool_budget_exhausted
    )

    return AskResponse(
        success=success,
        answer=answer,
        sql=sql,
        columns=columns,
        rows=rows,
        row_count=row_count,
        truncated=bool(execution.get("truncated", False)) if execution else False,
    )
