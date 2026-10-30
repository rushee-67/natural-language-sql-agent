import json

from langchain_core.tools import tool
from sqlalchemy import inspect

from app.database.connection import engine


@tool
def get_schema() -> str:
    """
    Inspect the database and return its tables, columns,
    data types, primary keys, and foreign-key relationships.
    """
    inspector = inspect(engine)

    tables = {}

    # Discover all tables in the database.
    for table_name in inspector.get_table_names():

        # Identify the primary-key columns.
        primary_key = inspector.get_pk_constraint(table_name)
        primary_key_columns = set(
            primary_key.get("constrained_columns") or []
        )

        # Map foreign-key columns to their referenced columns.
        foreign_keys = {}

        for fk in inspector.get_foreign_keys(table_name):
            local_columns = fk.get("constrained_columns") or []
            remote_columns = fk.get("referred_columns") or []
            remote_table = fk.get("referred_table")

            for local_column, remote_column in zip(
                local_columns, remote_columns
            ):
                foreign_keys[local_column] = (
                    f"{remote_table}.{remote_column}"
                )

        # Collect metadata for each column.
        columns = []

        for column in inspector.get_columns(table_name):
            column_name = column["name"]

            columns.append({
                "name": column_name,
                "type": str(column["type"]),
                "nullable": column["nullable"],
                "primary_key": (
                    column_name in primary_key_columns
                ),
                "foreign_key": foreign_keys.get(column_name),
            })

        tables[table_name] = {
            "columns": columns
        }

    schema = {
        "database": engine.dialect.name,
        "tables": tables,
    }

    # Return structured, readable information to the agent.
    return json.dumps(schema, indent=2)