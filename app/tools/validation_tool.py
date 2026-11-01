
import json

import sqlglot
from langchain_core.tools import tool
from sqlglot import exp
from sqlglot.errors import ParseError


# Maximum SQL statement length.
MAX_SQL_LENGTH = 10_000

# Query complexity limits.
MAX_JOIN_COUNT = 5
MAX_SUBQUERY_COUNT = 3
MAX_CTE_COUNT = 5
MAX_AST_NODES = 1000

# Potentially dangerous functions.
BLOCKED_FUNCTIONS = {
    "load_extension",
    "writefile",
    "readfile",
}


def _validate_query_complexity(statement) -> str | None:
    """
    Check structural complexity of a parsed SQL statement.

    Returns an error message when a limit is exceeded.
    Otherwise, returns None.
    """

    join_count = sum(
        1 for _ in statement.find_all(exp.Join)
    )

    if join_count > MAX_JOIN_COUNT:
        return (
            f"Query contains too many JOIN operations. "
            f"Maximum allowed: {MAX_JOIN_COUNT}."
        )

    subquery_count = sum(
        1 for _ in statement.find_all(exp.Subquery)
    )

    if subquery_count > MAX_SUBQUERY_COUNT:
        return (
            f"Query contains too many nested subqueries. "
            f"Maximum allowed: {MAX_SUBQUERY_COUNT}."
        )

    cte_count = sum(
        1 for _ in statement.find_all(exp.CTE)
    )

    if cte_count > MAX_CTE_COUNT:
        return (
            f"Query contains too many CTEs. "
            f"Maximum allowed: {MAX_CTE_COUNT}."
        )

    ast_node_count = sum(
        1 for _ in statement.walk()
    )

    if ast_node_count > MAX_AST_NODES:
        return (
            f"Query structure is too complex. "
            f"Maximum allowed AST nodes: {MAX_AST_NODES}."
        )

    return None


def validate_sql_query(query: str) -> dict:
    """Validate SQL using deterministic application rules."""

    if not isinstance(query, str) or not query.strip():
        return {
            "valid": False,
            "message": "SQL query cannot be empty.",
        }

    query = query.strip()

    if len(query) > MAX_SQL_LENGTH:
        return {
            "valid": False,
            "message": "SQL query exceeds the maximum allowed length.",
        }

    try:
        statements = [
            statement
            for statement in sqlglot.parse(query, read="sqlite")
            if statement is not None
        ]
    except ParseError:
        return {
            "valid": False,
            "message": "SQL could not be parsed for SQLite.",
        }

    # Reject empty input and multiple SQL statements.
    if len(statements) != 1:
        return {
            "valid": False,
            "message": "Exactly one SQL statement is required.",
        }

    statement = statements[0]

    # Allow SELECT statements, including SELECT with a CTE.
    if not isinstance(statement, exp.Select):
        return {
            "valid": False,
            "message": "Only SELECT queries are allowed.",
        }

    # Reject potentially dangerous functions.
    for function in statement.find_all(exp.Anonymous):
        if function.name.lower() in BLOCKED_FUNCTIONS:
            return {
                "valid": False,
                "message": (
                    f"Function '{function.name}' is not permitted."
                ),
            }

    # Reject queries that exceed structural complexity limits.
    complexity_error = _validate_query_complexity(statement)

    if complexity_error:
        return {
            "valid": False,
            "message": complexity_error,
        }

    return {
        "valid": True,
        "message": "SQL passed all validation checks.",
        "normalized_sql": statement.sql(dialect="sqlite"),
    }


@tool
def validate_sql(query: str) -> str:
    """
    Validate a candidate SQL query.

    Allows one SELECT statement and SELECT queries using CTEs.
    Rejects write statements, multiple statements, blocked
    functions and overly complex query structures.
    """
    result = validate_sql_query(query)
    return json.dumps(result)
