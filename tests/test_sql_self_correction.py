from tools.sql_tool import execute_sql
from tools.schema_tool import get_database_schema
from agents.data_analyst_agent_v3 import correct_sql


def test_self_correction():
    """
    Intentionally execute invalid SQL,
    capture the database error,
    ask Qwen to repair the SQL,
    and verify that the corrected query works.
    """

    print("Testing SQL Self-Correction")
    print("=" * 70)

    question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    schema = get_database_schema()

    # Intentionally incorrect SQL.
    # 'quantity' does not exist in order_items.
    bad_sql = """
    SELECT
        p.product_category_name,
        SUM(oi.quantity * oi.price) AS total_revenue
    FROM order_items oi
    JOIN products p
        ON oi.product_id = p.product_id
    GROUP BY p.product_category_name
    ORDER BY total_revenue DESC
    LIMIT 5
    """

    print("\nIntentionally Bad SQL:")
    print("-" * 70)
    print(bad_sql)
    print("-" * 70)

    try:
        execute_sql(bad_sql)

        print(
            "\nUnexpected result: "
            "the bad SQL executed successfully."
        )

    except Exception as error:

        print("\nExpected SQL Error:")
        print("-" * 70)
        print(error)
        print("-" * 70)

        print(
            "\nSending failed SQL and error "
            "to Qwen for correction..."
        )

        corrected_sql = correct_sql(
            question=question,
            schema=schema,
            failed_sql=bad_sql,
            error_message=str(error),
        )

        print("\nCorrected SQL:")
        print("-" * 70)
        print(corrected_sql)
        print("-" * 70)

        print("\nExecuting corrected SQL...")

        result = execute_sql(corrected_sql)

        print("\nCorrected Query Result:")
        print("-" * 70)
        print(result)
        print("-" * 70)

        print(
            "\nSelf-correction test completed successfully."
        )


if __name__ == "__main__":
    test_self_correction()