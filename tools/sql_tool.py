from pathlib import Path

import duckdb
import pandas as pd


# --------------------------------------------------
# Database configuration
# --------------------------------------------------

DATABASE_PATH = Path("database/olist.duckdb")


# --------------------------------------------------
# Connection helper
# --------------------------------------------------

def get_connection():
    """
    Create and return a DuckDB connection.
    """
    return duckdb.connect(str(DATABASE_PATH))


# --------------------------------------------------
# Schema tool
# --------------------------------------------------

def get_table_names():
    """
    Return all table names in the DuckDB database.
    """
    connection = get_connection()

    try:
        tables = connection.execute(
            """
            SHOW TABLES
            """
        ).fetchall()

        return [table[0] for table in tables]

    finally:
        connection.close()


# --------------------------------------------------
# Table schema tool
# --------------------------------------------------

def get_table_schema(table_name):
    """
    Return the schema of a specific table.
    """
    connection = get_connection()

    try:
        schema = connection.execute(
            f"""
            DESCRIBE {table_name}
            """
        ).fetchdf()

        return schema

    finally:
        connection.close()


# --------------------------------------------------
# SQL execution tool
# --------------------------------------------------

def execute_sql(query):
    """
    Execute a read-only SQL query and return the result
    as a Pandas DataFrame.
    """

    query_clean = query.strip().lower()

    # Basic safety check
    blocked_keywords = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "truncate ",
        "create ",
        "replace ",
    ]

    for keyword in blocked_keywords:
        if keyword in query_clean:
            raise ValueError(
                f"Blocked SQL operation detected: {keyword.strip()}"
            )

    connection = get_connection()

    try:
        result = connection.execute(query).fetchdf()
        return result

    finally:
        connection.close()


# --------------------------------------------------
# Sample rows tool
# --------------------------------------------------

def get_sample_rows(table_name, limit=5):
    """
    Return sample rows from a table.
    """

    query = f"""
    SELECT *
    FROM {table_name}
    LIMIT {limit}
    """

    return execute_sql(query)


# --------------------------------------------------
# Row count tool
# --------------------------------------------------

def get_row_count(table_name):
    """
    Return the number of rows in a table.
    """

    query = f"""
    SELECT COUNT(*) AS row_count
    FROM {table_name}
    """

    result = execute_sql(query)

    return int(result.iloc[0]["row_count"])


# --------------------------------------------------
# Basic test
# --------------------------------------------------

if __name__ == "__main__":

    print("Testing SQL Tool")
    print("=" * 60)

    print("\nAvailable tables:")

    for table in get_table_names():
        print(f"  {table}")

    print("\nOrders table schema:")
    print(get_table_schema("orders"))

    print("\nSample orders:")
    print(get_sample_rows("orders", 5))

    print("\nOrders row count:")
    print(get_row_count("orders"))

    print("\nSQL Tool test completed successfully.")