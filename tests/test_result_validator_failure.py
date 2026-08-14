from tools.sql_tool import execute_sql
from tools.result_validator import validate_result


def test_invalid_result():

    print("Testing Result Validator - Invalid Result")
    print("=" * 70)

    question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    # Intentionally creates duplicate amplification.
    # Joining products inside the category CTE creates
    # multiple rows for the same category.
    bad_sql = """
    WITH translated_categories AS (
        SELECT
            t.product_category_name,
            p.product_category_name
                AS product_category_name_english
        FROM products p
        JOIN category_translation t
            ON p.product_category_name =
               t.product_category_name
    )
    SELECT
        pc.product_category_name_english AS category,
        SUM(oi.price) AS total_revenue
    FROM order_items oi
    JOIN products p
        ON oi.product_id = p.product_id
    JOIN translated_categories pc
        ON p.product_category_name =
           pc.product_category_name
    GROUP BY
        pc.product_category_name_english
    ORDER BY
        total_revenue DESC
    LIMIT 5
    """

    print("\nExecuting intentionally flawed SQL...")

    result = execute_sql(bad_sql)

    print("\nQuery Result:")
    print("-" * 70)
    print(result)
    print("-" * 70)

    validation = validate_result(
        question=question,
        sql=bad_sql,
        result=result,
    )

    print("\nValidation Result:")
    print(validation)

    assert validation["is_valid"] is False

    print(
        "\nSUCCESS: Validator correctly detected "
        "the logically invalid result."
    )


if __name__ == "__main__":
    test_invalid_result()