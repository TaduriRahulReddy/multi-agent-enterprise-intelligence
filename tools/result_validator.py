from tools.sql_tool import execute_sql
from tools.llm_tool import ask_llm


def validate_result(
    question: str,
    sql: str,
    result,
) -> dict:
    """
    Validate whether a SQL result is logically plausible.

    Returns a dictionary like:
    {
        "is_valid": True/False,
        "reason": "...",
    }
    """

    # --------------------------------------------------
    # Rule-based checks first
    # --------------------------------------------------

    if result is None:
        return {
            "is_valid": False,
            "reason": "Query returned no result."
        }

    if result.empty:
        return {
            "is_valid": False,
            "reason": "Query returned an empty result."
        }

    # --------------------------------------------------
    # Known business sanity check
    # --------------------------------------------------

    total_revenue_df = execute_sql(
        """
        SELECT SUM(price) AS total_revenue
        FROM order_items
        """
    )

    total_revenue = float(
        total_revenue_df.iloc[0]["total_revenue"]
    )

    # If the result contains a revenue-like column,
    # make sure no category exceeds total company revenue.
    revenue_columns = [
        column
        for column in result.columns
        if "revenue" in column.lower()
    ]

    for column in revenue_columns:

        max_value = result[column].max()

        if max_value > total_revenue:

            return {
                "is_valid": False,
                "reason": (
                    f"Result is logically inconsistent. "
                    f"Maximum {column} value "
                    f"({max_value:,.2f}) exceeds total "
                    f"product revenue "
                    f"({total_revenue:,.2f})."
                )
            }

    # --------------------------------------------------
    # LLM-based semantic validation
    # --------------------------------------------------

    prompt = f"""
You are a senior data analyst validating SQL results.

BUSINESS QUESTION:
{question}

GENERATED SQL:
{sql}

QUERY RESULT:
{result.to_string(index=False)}

VALIDATION TASK:

Determine whether the result is logically consistent
with the business question and SQL.

Check for:

1. Duplicate amplification caused by incorrect joins.
2. Implausibly large values.
3. Incorrect aggregation.
4. Wrong category mappings.
5. Missing GROUP BY logic.
6. Results that do not answer the business question.

Return exactly one of these formats:

VALID: <short reason>

or

INVALID: <short reason>
"""

    response = ask_llm(prompt).strip()

    if response.upper().startswith("VALID:"):

        return {
            "is_valid": True,
            "reason": response
        }

    return {
        "is_valid": False,
        "reason": response
    }


if __name__ == "__main__":

    print("Testing Result Validator")
    print("=" * 70)

    question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    sql = """
    SELECT
        p.product_category_name,
        SUM(oi.price) AS total_revenue
    FROM order_items oi
    JOIN products p
        ON oi.product_id = p.product_id
    GROUP BY p.product_category_name
    ORDER BY total_revenue DESC
    LIMIT 5
    """

    result = execute_sql(sql)

    validation = validate_result(
        question=question,
        sql=sql,
        result=result,
    )

    print("\nValidation Result:")
    print(validation)

    print("\nResult Validator test completed.")