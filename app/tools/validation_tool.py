import json

import sqlglot
from langchain_core.tools import tool
from sqlglot import exp
from sqlglot.errors import ParseError


MAX_SQL_LENGTH = 10_000

# Block potentially dangerous functions explicitly.
BLOCKED_FUNCTIONS = {
    "load_extension",
    "writefile",
    "readfile",
}


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

    # Reject selected functions that are unnecessary for this project
    # and may expose filesystem or extension capabilities.
    for function in statement.find_all(exp.Anonymous):
        if function.name.lower() in BLOCKED_FUNCTIONS:
            return {
                "valid": False,
                "message": (
                    f"Function '{function.name}' is not permitted."
                ),
            }

    return {
        "valid": True,
        "message": "SQL passed the initial validation checks.",
        "normalized_sql": statement.sql(dialect="sqlite"),
    }


@tool
def validate_sql(query: str) -> str:
    """
    Validate a candidate SQL query.

    Allows one SELECT statement and SELECT queries using CTEs.
    Rejects write statements, multiple statements and blocked functions.
    """
    result = validate_sql_query(query)
    return json.dumps(result)