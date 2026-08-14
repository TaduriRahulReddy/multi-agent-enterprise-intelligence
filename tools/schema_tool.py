from tools.sql_tool import execute_sql


def get_database_schema() -> str:
    """
    Retrieve the actual DuckDB schema and convert it into
    text that can be provided to the LLM.
    """

    tables_df = execute_sql("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
    """)

    schema_parts = []

    for table_name in tables_df["table_name"]:

        columns_df = execute_sql(f"""
            SELECT
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name = '{table_name}'
            ORDER BY ordinal_position
        """)

        schema_parts.append(f"TABLE: {table_name}")

        for _, row in columns_df.iterrows():
            schema_parts.append(
                f"  - {row['column_name']} ({row['data_type']})"
            )

        schema_parts.append("")

    return "\n".join(schema_parts)


if __name__ == "__main__":

    print("Testing Database Schema Tool")
    print("=" * 60)

    schema = get_database_schema()

    print(schema)

    print("=" * 60)
    print("Schema Tool test completed successfully.")