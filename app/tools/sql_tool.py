import json

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.database.connection import engine
from app.tools.validation_tool import validate_sql_query


MAX_RESULT_ROWS = 100


def _execute_read_query(query: str) -> dict:
    """Validate and execute one read-only SQL query."""

    # Defense in depth: never trust the agent to validate the query.
    validation = validate_sql_query(query)

    if not validation["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "error": validation["message"],
        }

    sql = validation["normalized_sql"]

    try:
        with engine.connect() as connection:
            try:
                # Prevent database modifications on this connection.
                connection.exec_driver_sql("PRAGMA query_only = ON")

                result = connection.execute(text(sql))

                columns = list(result.keys())

                # Fetch one extra row to detect truncated results.
                fetched_rows = result.fetchmany(MAX_RESULT_ROWS + 1)

                truncated = len(fetched_rows) > MAX_RESULT_ROWS
                fetched_rows = fetched_rows[:MAX_RESULT_ROWS]

                rows = [
                    dict(zip(columns, row))
                    for row in fetched_rows
                ]

                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": truncated,
                }

            finally:
                # Reset connection state before returning it to the pool.
                try:
                    connection.rollback()
                    connection.exec_driver_sql("PRAGMA query_only = OFF")
                except SQLAlchemyError:
                    # Discard a connection that cannot be safely reset.
                    connection.invalidate()

    except SQLAlchemyError as exc:
        return {
            "success": False,
            "stage": "execution",
            "error": str(getattr(exc, "orig", exc)),
        }


@tool
def execute_sql(query: str) -> str:
    """
    Execute one validated, read-only SQL query against the SQLite
    database and return structured JSON containing results or errors.
    """
    result = _execute_read_query(query)
    return json.dumps(result, default=str)