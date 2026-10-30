import json

import app.database.models
from sqlalchemy import inspect

from app.database.connection import Base, engine
from app.tools.schema_tool import get_schema
from app.tools.validation_tool import validate_sql
from app.tools.sql_tool import execute_sql
import app.tools.sql_tool as sql_tool_module
from sqlalchemy import text


# Ensure the schema exists, even in a fresh test environment.
Base.metadata.create_all(bind=engine)


def test_get_schema_lists_expected_tables():
    result = get_schema.invoke({})
    schema = json.loads(result)

    assert schema["database"] == "sqlite"

    assert set(schema["tables"]) == {
        "customers",
        "products",
        "orders",
        "order_items",
    }


def test_get_schema_includes_column_metadata():
    schema = json.loads(get_schema.invoke({}))

    customers = {
        column["name"]: column
        for column in schema["tables"]["customers"]["columns"]
    }

    assert customers["customer_id"]["primary_key"] is True
    assert customers["name"]["type"] == "VARCHAR(100)"
    assert customers["email"]["nullable"] is False


def test_get_schema_includes_foreign_keys():
    schema = json.loads(get_schema.invoke({}))

    orders = {
        column["name"]: column
        for column in schema["tables"]["orders"]["columns"]
    }

    assert orders["customer_id"]["foreign_key"] == (
        "customers.customer_id"
    )





def validate(query: str) -> dict:
    result = validate_sql.invoke({"query": query})
    return json.loads(result)


def test_validate_allows_select():
    result = validate("SELECT name FROM customers")

    assert result["valid"] is True
    assert "normalized_sql" in result


def test_validate_allows_cte():
    query = """
        WITH customer_counts AS (
            SELECT city, COUNT(*) AS total
            FROM customers
            GROUP BY city
        )
        SELECT * FROM customer_counts
    """

    assert validate(query)["valid"] is True


def test_validate_rejects_modification_statements():
    queries = [
        "INSERT INTO customers VALUES (1, 'A', 'B', 'a@b.com')",
        "UPDATE customers SET city = 'Delhi'",
        "DELETE FROM customers",
        "DROP TABLE customers",
    ]

    for query in queries:
        assert validate(query)["valid"] is False


def test_validate_rejects_multiple_statements():
    query = "SELECT 1; SELECT 2"

    assert validate(query)["valid"] is False


def test_validate_rejects_empty_or_invalid_input():
    assert validate("")["valid"] is False
    assert validate("NOT A QUERY")["valid"] is False


def test_validate_rejects_blocked_function():
    result = validate("SELECT load_extension('x')")

    assert result["valid"] is False

def execute(query: str) -> dict:
    result = execute_sql.invoke({"query": query})
    return json.loads(result)


def test_execute_sql_returns_query_results():
    result = execute(
        "SELECT name FROM customers ORDER BY customer_id LIMIT 3"
    )

    assert result["success"] is True
    assert result["row_count"] == 3
    assert result["columns"] == ["name"]
    assert len(result["rows"]) == 3
    assert result["truncated"] is False


def test_execute_sql_rejects_unsafe_query():
    result = execute("DELETE FROM customers")

    assert result["success"] is False
    assert result["stage"] == "validation"


def test_execute_sql_returns_database_errors():
    result = execute(
        "SELECT missing_column FROM customers"
    )

    assert result["success"] is False
    assert result["stage"] == "execution"
    assert result["error"]


def test_execute_sql_limits_large_results():
    result = execute(
        "SELECT order_item_id "
        "FROM order_items ORDER BY order_item_id"
    )

    assert result["success"] is True
    assert result["row_count"] == MAX_ROWS
    assert result["truncated"] is True

def test_execution_blocks_write_if_validator_is_bypassed(monkeypatch):
    # Record the customer count before the test.
    with engine.connect() as connection:
        before = connection.execute(
            text("SELECT COUNT(*) FROM customers")
        ).scalar_one()

    # Simulate a broken validator that incorrectly accepts DELETE.
    monkeypatch.setattr(
        sql_tool_module,
        "validate_sql_query",
        lambda query: {
            "valid": True,
            "normalized_sql": query,
        },
    )

    result = json.loads(
        sql_tool_module.execute_sql.invoke(
            {"query": "DELETE FROM customers"}
        )
    )

    # SQLite's query-only restriction should block the write.
    assert result["success"] is False
    assert result["stage"] == "execution"

    # The database contents must remain unchanged.
    with engine.connect() as connection:
        after = connection.execute(
            text("SELECT COUNT(*) FROM customers")
        ).scalar_one()

    assert after == before


# Test constant: keep synchronized with sql_tool.py.
MAX_ROWS = 100