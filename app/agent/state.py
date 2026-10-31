from typing import Annotated

from typing_extensions import NotRequired, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SQLAgentState(TypedDict):
    """
    State shared between nodes in the SQL agent graph.
    """

    # Conversation history, including AI and tool messages.
    messages: Annotated[list[AnyMessage], add_messages]

    # Number of recovery attempts already granted.
    retry_count: NotRequired[int]

    # Most recent validation or execution error.
    last_error: NotRequired[str]

    # Whether the recovery limit has been reached.
    recovery_exhausted: NotRequired[bool]