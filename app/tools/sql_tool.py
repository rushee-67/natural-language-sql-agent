
import json
import sqlite3
import time
import logging

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


from app.database.connection import engine
from app.tools.validation_tool import validate_sql_query

logger = logging.getLogger(__name__)

# Maximum number of rows returned to the application.
MAX_RESULT_ROWS = 100

# Maximum SQLite query execution time.
MAX_QUERY_EXECUTION_SECONDS = 5.0

# Invoke the progress handler every 1,000 SQLite VM instructions.
SQLITE_PROGRESS_HANDLER_STEPS = 1000

# Functions that must not be executed by the SQL agent.
BLOCKED_SQLITE_FUNCTIONS = {
    "load_extension",
    "writefile",
    "readfile",
}

# SQLite operations explicitly permitted by the authorizer.
ALLOWED_SQLITE_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}

# SQLite runtime resource limits.
SQLITE_RUNTIME_LIMITS = {
    "SQLITE_LIMIT_LENGTH": 1_000_000,
    "SQLITE_LIMIT_SQL_LENGTH": 10_000,
    "SQLITE_LIMIT_COLUMN": 50,
    "SQLITE_LIMIT_EXPR_DEPTH": 100,
    "SQLITE_LIMIT_COMPOUND_SELECT": 10,
    "SQLITE_LIMIT_FUNCTION_ARG": 20,
    "SQLITE_LIMIT_ATTACHED": 0,
    "SQLITE_LIMIT_LIKE_PATTERN_LENGTH": 500,
    "SQLITE_LIMIT_VARIABLE_NUMBER": 500,
    "SQLITE_LIMIT_WORKER_THREADS": 0,
}


def _sqlite_authorizer(
    action,
    arg1,
    arg2,
    database_name,
    trigger_name,
):
    """Allow only approved read operations."""

    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or "").lower()

        if function_name in BLOCKED_SQLITE_FUNCTIONS:
            return sqlite3.SQLITE_DENY

        return sqlite3.SQLITE_OK

    if action in ALLOWED_SQLITE_ACTIONS:
        return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY


def _restore_sqlite_runtime_limits(
    dbapi_connection,
    previous_limits: dict,
) -> None:
    """Restore the original SQLite runtime limits."""

    for category, previous_value in previous_limits.items():
        dbapi_connection.setlimit(
            category,
            previous_value,
        )


def _apply_sqlite_runtime_limits(dbapi_connection) -> dict:
    """
    Apply SQLite runtime limits and return previous values.

    Never increase an existing, stricter SQLite limit.
    """

    if not callable(getattr(dbapi_connection, "setlimit", None)):
        raise RuntimeError(
            "SQLite runtime limits require Python 3.11 or newer."
        )

    if not callable(getattr(dbapi_connection, "getlimit", None)):
        raise RuntimeError(
            "SQLite getlimit() is unavailable."
        )

    previous_limits = {}

    try:
        for name, maximum in SQLITE_RUNTIME_LIMITS.items():
            category = getattr(sqlite3, name, None)

            # Skip limits unavailable in this SQLite build.
            if category is None:
                continue

            previous = dbapi_connection.getlimit(category)

            # SQLite may be compiled without an expression-depth
            # limit. In that case, runtime lowering is unavailable.
            if (
                name == "SQLITE_LIMIT_EXPR_DEPTH"
                and previous == 0
            ):
                continue

            # Preserve a stricter pre-existing limit.
            target_limit = min(previous, maximum)

            previous_limits[category] = previous

            dbapi_connection.setlimit(
                category,
                target_limit,
            )

            actual = dbapi_connection.getlimit(category)

            if actual > target_limit:
                raise RuntimeError(
                    f"Could not enforce SQLite limit: {name}."
                )

    except Exception as exc:
        # Attempt to restore every limit changed so far.
        try:
            _restore_sqlite_runtime_limits(
                dbapi_connection,
                previous_limits,
            )
        except Exception as restore_exc:
            raise RuntimeError(
                "SQLite limits could not be applied or restored."
            ) from restore_exc

        raise RuntimeError(
            "SQLite runtime limit configuration failed."
        ) from exc

    return previous_limits


def _sanitize_execution_error(exc: Exception) -> str:
    """
    Convert raw SQLite errors into safe, actionable messages.

    Detailed exceptions are retained in application logs.
    Raw database error messages are not returned to the agent.
    """

    original_error = getattr(exc, "orig", exc)
    message = str(original_error).lower()

    if "no such table" in message:
        return (
            "The query references a table or view that does not "
            "exist. Inspect the database schema."
        )

    if "no such column" in message:
        return (
            "The query references a column that does not exist. "
            "Inspect the schema and verify column names."
        )

    if "ambiguous column name" in message:
        return (
            "A column reference is ambiguous. Qualify the column "
            "using its table name or alias."
        )

    if "no such function" in message:
        return (
            "The query uses a SQL function that is unavailable "
            "in this database."
        )

    if "syntax error" in message or "incomplete input" in message:
        return "The generated SQL contains a syntax error."

    if "not authorized" in message or "authorization denied" in message:
        return (
            "The query attempted a database operation that is "
            "not permitted."
        )

    if "too many columns" in message:
        return "The query exceeds the configured column limit."

    if "too many sql variables" in message:
        return "The query exceeds the configured SQL parameter limit."

    if "string or blob too big" in message:
        return "The query exceeds the configured data-size limit."

    if "like or glob pattern too complex" in message:
        return "The query exceeds the configured pattern-length limit."

    if "too many terms in compound select" in message:
        return "The query exceeds the configured compound SELECT limit."

    if "too many arguments on function" in message:
        return "The query exceeds the configured function argument limit."

    if "expression tree is too large" in message:
        return "The query exceeds the configured expression-depth limit."

    if "database is locked" in message or "database is busy" in message:
        return "The database is temporarily busy. Please retry."

    return (
        "The query could not be executed. It may violate SQL "
        "validation or database resource limits."
    )


def _execute_read_query(
    query: str,
    timeout_seconds: float = MAX_QUERY_EXECUTION_SECONDS,
) -> dict:
    """Validate and execute one read-only SQL query."""

    # Defense in depth: independently validate the query.
    validation = validate_sql_query(query)

    if not validation["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "error": validation["message"],
        }

    sql = validation["normalized_sql"]

    if (
        not isinstance(timeout_seconds, (int, float))
        or timeout_seconds < 0
    ):
        return {
            "success": False,
            "stage": "configuration",
            "error": "Invalid query execution timeout.",
        }

    # Track whether SQLite interrupted the query due to timeout.
    timed_out = False
    stage = "connection"

    try:
        with engine.connect() as connection:
            stage = "configuration"

            dbapi_connection = (
                connection.connection.driver_connection
            )

            previous_limits = {}
            result = None
            connection_invalidated = False

            # Calculate the execution deadline.
            deadline = time.monotonic() + timeout_seconds

            def progress_handler() -> int:
                """
                Interrupt SQLite when the deadline expires.

                Returning nonzero instructs SQLite to abort
                the current statement.
                """
                nonlocal timed_out

                if time.monotonic() >= deadline:
                    timed_out = True
                    return 1

                return 0

            try:
                # Apply SQLite runtime limits.
                try:
                    previous_limits = (
                        _apply_sqlite_runtime_limits(
                            dbapi_connection
                        )
                    )
                except Exception:
                    # Never reuse a connection whose security
                    # configuration could not be established.
                    connection.invalidate()
                    connection_invalidated = True
                    raise

                # Enable read-only protection before installing
                # the authorizer because PRAGMA is not permitted
                # by our authorizer.
                connection.exec_driver_sql(
                    "PRAGMA query_only = ON"
                )

                # Install SQLite authorization.
                dbapi_connection.set_authorizer(
                    _sqlite_authorizer
                )

                # Install execution timeout.
                dbapi_connection.set_progress_handler(
                    progress_handler,
                    SQLITE_PROGRESS_HANDLER_STEPS,
                )

                # All security configuration is complete.
                stage = "execution"

                # Execute the validated query.
                result = connection.execute(text(sql))

                columns = list(result.keys())

                # Fetch one extra row to detect truncation.
                fetched_rows = result.fetchmany(
                    MAX_RESULT_ROWS + 1
                )

                truncated = (
                    len(fetched_rows) > MAX_RESULT_ROWS
                )

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
                # Do not attempt to reuse an invalidated connection.
                if not connection_invalidated:
                    try:
                        # Close the result cursor.
                        if result is not None:
                            result.close()

                        # Remove temporary SQLite callbacks.
                        dbapi_connection.set_progress_handler(
                            None,
                            0,
                        )

                        dbapi_connection.set_authorizer(None)

                        # Roll back the read transaction.
                        connection.rollback()

                        # Reset read-only connection configuration.
                        connection.exec_driver_sql(
                            "PRAGMA query_only = OFF"
                        )

                        # Restore the original SQLite limits.
                        _restore_sqlite_runtime_limits(
                            dbapi_connection,
                            previous_limits,
                        )

                    except Exception:
                        # Never return a connection with uncertain
                        # security state to the connection pool.
                        connection.invalidate()

    
    except (RuntimeError, sqlite3.Error, SQLAlchemyError) as exc:

        original_error = getattr(exc, "orig", exc)

        # The progress handler interrupted the query.
        if timed_out:
            logger.warning(
                "SQL query exceeded its execution deadline."
            )

            return {
                "success": False,
                "stage": "execution",
                "timed_out": True,
                "error": (
                    "Query execution exceeded the maximum "
                    f"allowed time of {timeout_seconds:g} seconds."
                ),
            }

        # Preserve detailed diagnostic information in server logs.
        logger.warning(
            "SQL tool failed during %s. Exception: %s: %s",
            stage,
            type(original_error).__name__,
            str(original_error),
        )

        if stage == "configuration":
            return {
                "success": False,
                "stage": "configuration",
                "error": (
                    "SQLite security configuration could not "
                    "be applied safely."
                ),
            }

        if stage == "connection":
            return {
                "success": False,
                "stage": "connection",
                "error": (
                    "The database connection is temporarily unavailable."
                ),
            }

        return {
            "success": False,
            "stage": "execution",
            "error": _sanitize_execution_error(exc),
        }



@tool
def execute_sql(query: str) -> str:
    """
    Execute one validated, read-only SQL query against SQLite.

    Returns structured JSON containing query results or errors.
    """
    result = _execute_read_query(query)

    return json.dumps(result, default=str)
