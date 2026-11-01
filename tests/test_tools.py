import json

import app.database.models
from sqlalchemy import inspect

import sqlite3
import app.tools.sql_tool as sql_tool

from app.database.connection import Base, engine
from app.tools.schema_tool import get_schema
from app.tools.validation_tool import validate_sql
from app.tools.sql_tool import execute_sql
import app.tools.sql_tool as sql_tool_module
from sqlalchemy import text
from app.tools.validation_tool import validate_sql_query


# Ensure the schema exists, even in a fresh test environment.
Base.metadata.create_all(bind=engine)

# Test constant: keep synchronized with sql_tool.py.
MAX_ROWS = 100

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


def test_sqlite_authorizer_allows_safe_read_operations():
    authorizer = sql_tool._sqlite_authorizer

    assert authorizer(
        sqlite3.SQLITE_SELECT, None, None, "main", None
    ) == sqlite3.SQLITE_OK

    assert authorizer(
        sqlite3.SQLITE_READ, "customers", "name", "main", None
    ) == sqlite3.SQLITE_OK

    assert authorizer(
        sqlite3.SQLITE_FUNCTION, None, "count", "main", None
    ) == sqlite3.SQLITE_OK


def test_sqlite_authorizer_blocks_write_operations():
    authorizer = sql_tool._sqlite_authorizer

    blocked_actions = [
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
    ]

    for action in blocked_actions:
        assert authorizer(
            action, None, None, "main", None
        ) == sqlite3.SQLITE_DENY


def test_sqlite_authorizer_blocks_dangerous_functions():
    authorizer = sql_tool._sqlite_authorizer

    for function_name in ["load_extension", "readfile", "writefile"]:
        assert authorizer(
            sqlite3.SQLITE_FUNCTION,
            None,
            function_name,
            "main",
            None,
        ) == sqlite3.SQLITE_DENY


def test_query_execution_timeout_and_connection_cleanup(monkeypatch):
    # Make the progress callback execute very frequently.
    monkeypatch.setattr(
        sql_tool,
        "SQLITE_PROGRESS_HANDLER_STEPS",
        1,
    )

    expensive_query = """
        WITH RECURSIVE counter(value) AS (
            SELECT 1
            UNION ALL
            SELECT value + 1
            FROM counter
            WHERE value < 1000000
        )
        SELECT value FROM counter
    """

    # Zero-second deadline ensures the first progress callback
    # interrupts the expensive query.
    result = sql_tool._execute_read_query(
        expensive_query,
        timeout_seconds=0,
    )

    assert result["success"] is False
    assert result["stage"] == "execution"
    assert result["timed_out"] is True
    assert "maximum allowed time" in result["error"]

    # Verify the pooled connection is clean after interruption.
    follow_up_result = sql_tool._execute_read_query(
        "SELECT 1 AS value"
    )

    assert follow_up_result["success"] is True
    assert follow_up_result["rows"] == [{"value": 1}]


def test_validation_rejects_too_many_joins():
    query = """
        SELECT c.customer_id
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN customers c2 ON c2.customer_id = c.customer_id
        JOIN orders o2 ON o2.customer_id = c2.customer_id
        JOIN products p2 ON p2.product_id = oi.product_id
    """

    result = validate_sql_query(query)

    assert result["valid"] is False
    assert "too many JOIN operations" in result["message"]


def test_validation_rejects_too_many_subqueries():
    query = """
        SELECT *
        FROM (
            SELECT *
            FROM (
                SELECT *
                FROM (
                    SELECT *
                    FROM (
                        SELECT 1 AS value
                    ) AS a
                ) AS b
            ) AS c
        ) AS d
    """

    result = validate_sql_query(query)

    assert result["valid"] is False
    assert "nested subqueries" in result["message"]


def test_validation_rejects_too_many_ctes():
    ctes = [
        f"c{i} AS (SELECT {i} AS value)"
        for i in range(6)
    ]

    query = (
        "WITH "
        + ", ".join(ctes)
        + " SELECT value FROM c5"
    )

    result = validate_sql_query(query)

    assert result["valid"] is False
    assert "too many CTEs" in result["message"]


def test_validation_rejects_excessive_ast_nodes():
    expressions = " + ".join(["1"] * 600)
    query = f"SELECT {expressions}"

    result = validate_sql_query(query)

    assert result["valid"] is False
    assert "too complex" in result["message"]


def test_validation_allows_normal_analytics_query():
    query = """
        SELECT
            p.product_id,
            p.product_name,
            SUM(oi.quantity * oi.unit_price) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.status IN ('shipped', 'delivered')
        GROUP BY p.product_id, p.product_name
        ORDER BY revenue DESC
        LIMIT 5
    """

    result = validate_sql_query(query)

    assert result["valid"] is True
    assert "normalized_sql" in result


def test_sqlite_runtime_limits_are_applied_and_restored():
    db = sqlite3.connect(":memory:")

    try:
        # Capture original settings.
        original_limits = {
            getattr(sqlite3, name): db.getlimit(
                getattr(sqlite3, name)
            )
            for name in sql_tool.SQLITE_RUNTIME_LIMITS
            if hasattr(sqlite3, name)
        }

        previous_limits = (
            sql_tool._apply_sqlite_runtime_limits(db)
        )

        # Verify configured limits.
        for name, maximum in sql_tool.SQLITE_RUNTIME_LIMITS.items():
            category = getattr(sqlite3, name, None)

            if category is None:
                continue

            actual = db.getlimit(category)

            # An expression-depth limit of zero means
            # SQLite was compiled without that limit.
            if (
                name == "SQLITE_LIMIT_EXPR_DEPTH"
                and original_limits[category] == 0
            ):
                assert actual == 0
                continue

            assert actual <= maximum

        # Restore original settings.
        sql_tool._restore_sqlite_runtime_limits(
            db,
            previous_limits,
        )

        for category, original_value in original_limits.items():
            assert db.getlimit(category) == original_value

    finally:
        db.close()


def test_analytics_query_works_with_sqlite_runtime_limits():
    query = """
        SELECT
            p.product_name,
            SUM(oi.quantity * oi.unit_price) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.status IN ('shipped', 'delivered')
        GROUP BY p.product_name
        ORDER BY revenue DESC
        LIMIT 5
    """

    result = sql_tool._execute_read_query(query)

    assert result["success"] is True
    assert len(result["rows"]) == 5
    assert result["truncated"] is False


def test_sql_error_sanitization_hides_table_name():
    error = sqlite3.OperationalError(
        "no such table: confidential_customer_data"
    )

    result = sql_tool._sanitize_execution_error(error)

    assert "does not exist" in result
    assert "confidential_customer_data" not in result


def test_sql_error_sanitization_handles_missing_columns():
    error = sqlite3.OperationalError(
        "no such column: secret_salary"
    )

    result = sql_tool._sanitize_execution_error(error)

    assert "column that does not exist" in result
    assert "secret_salary" not in result


def test_sql_error_sanitization_handles_ambiguous_columns():
    error = sqlite3.OperationalError(
        "ambiguous column name: customer_id"
    )

    result = sql_tool._sanitize_execution_error(error)

    assert "ambiguous" in result


def test_sql_error_sanitization_handles_authorization_errors():
    error = sqlite3.DatabaseError("not authorized")

    result = sql_tool._sanitize_execution_error(error)

    assert "not permitted" in result


def test_sql_error_sanitization_handles_unknown_errors():
    error = sqlite3.DatabaseError("some unexpected internal error")

    result = sql_tool._sanitize_execution_error(error)

    assert "could not be executed" in result
    assert "unexpected internal error" not in result

