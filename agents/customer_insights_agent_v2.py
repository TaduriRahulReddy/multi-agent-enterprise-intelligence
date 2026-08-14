from tools.sql_tool import execute_sql
from tools.llm_tool import ask_llm
from tools.schema_tool import get_database_schema
from tools.result_validator import validate_result


# ============================================================
# CUSTOMER INSIGHTS AGENT - VERSION 2
# RESULT-SHAPE AWARE CUSTOMER ANALYTICS
# ============================================================


MAX_ATTEMPTS = 3


def build_customer_sql_prompt(
    question: str,
    schema: str,
) -> str:
    """
    Build a customer-focused SQL prompt that also considers
    the expected shape of the answer.
    """

    return f"""
You are a Customer Insights Agent working with DuckDB.

Your job is to answer customer-related business questions
using the enterprise database.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

IMPORTANT CUSTOMER IDENTITY RULE:

customer_unique_id is NOT in the orders table.

To use customer_unique_id, join:

orders.customer_id = customers.customer_id

Then use:

customers.customer_unique_id

CUSTOMER ANALYTICS GUIDANCE:

1. customers.customer_id is the order-level customer identifier.

2. customers.customer_unique_id is the stable customer identity
   across multiple orders.

3. orders.customer_id joins to:
   customers.customer_id

4. reviews.order_id joins to:
   orders.order_id

5. payments.order_id joins to:
   orders.order_id

6. Customer geography is available through:
   customers.customer_city
   customers.customer_state

7. Do not invent customer demographic fields.

RESULT-SHAPE RULES:

1. If the user asks:
   "How many..."
   "What is the count..."
   "How many repeat customers..."
   then return a single aggregated count.

2. Do NOT return thousands of customer IDs when the user
   only asked for a count.

3. For repeat-customer count, use logic equivalent to:

WITH repeat_customers AS (
    SELECT
        c.customer_unique_id
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
    HAVING COUNT(DISTINCT o.order_id) > 1
)
SELECT
    COUNT(*) AS repeat_customer_count
FROM repeat_customers

4. If the user asks to list repeat customers, then returning
   customer IDs is appropriate.

5. Match the SQL result shape to the business question.

GENERAL RULES:

1. Use only tables and columns present in the schema.
2. Generate DuckDB-compatible SQL.
3. Do not invent tables or columns.
4. Use correct JOIN conditions.
5. Generate only SELECT or WITH queries.
6. Do not modify the database.
7. Return only executable SQL.
8. Do not include explanations.
9. Do not include markdown.
10. Avoid SELECT * unless explicitly requested.
11. Avoid unnecessary joins.
12. Verify every table and column before returning.

Return only SQL.
"""


def build_customer_correction_prompt(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Repair failed customer SQL.
    """

    return f"""
You are an expert DuckDB SQL debugger specializing in
customer analytics.

A customer query failed.

BUSINESS QUESTION:
{question}

DATABASE SCHEMA:
{schema}

FAILED SQL:
{failed_sql}

DUCKDB ERROR:
{error_message}

IMPORTANT:

The DuckDB error is authoritative.

If a column does not exist, do not use it again.

CUSTOMER IDENTITY RULE:

customer_unique_id exists in customers, not orders.

Use:

orders.customer_id = customers.customer_id

Then use:

customers.customer_unique_id

RESULT-SHAPE RULE:

If the business question asks "How many repeat customers",
the corrected SQL must return a SINGLE COUNT row,
not a list of customer IDs.

A correct pattern is:

WITH repeat_customers AS (
    SELECT
        c.customer_unique_id
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
    HAVING COUNT(DISTINCT o.order_id) > 1
)
SELECT
    COUNT(*) AS repeat_customer_count
FROM repeat_customers

RULES:

1. Fix the specific DuckDB error.
2. Preserve the original business intent.
3. Use only valid schema fields.
4. Match result shape to the question.
5. Generate DuckDB-compatible SQL.
6. Generate only SELECT or WITH queries.
7. Return only corrected SQL.
8. Do not include explanations.
9. Do not include markdown.

Return only corrected SQL.
"""


def build_shape_correction_prompt(
    question: str,
    schema: str,
    sql: str,
    result_text: str,
) -> str:
    """
    Fix SQL that executes successfully but returns the wrong
    output shape for the user's question.
    """

    return f"""
You are an expert customer analytics SQL reviewer.

The SQL executed successfully, but the output shape does
not match the user's question.

BUSINESS QUESTION:
{question}

DATABASE SCHEMA:
{schema}

SQL:
{sql}

CURRENT RESULT:
{result_text}

IMPORTANT:

If the user asks "How many", return one aggregated count row.

Do not return a list of individual customer IDs when the
question asks only for a count.

For repeat-customer count, use:

WITH repeat_customers AS (
    SELECT
        c.customer_unique_id
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
    HAVING COUNT(DISTINCT o.order_id) > 1
)
SELECT
    COUNT(*) AS repeat_customer_count
FROM repeat_customers

RULES:

1. Preserve the business question.
2. Use only valid schema fields.
3. Fix only the result-shape problem.
4. Return only executable DuckDB SQL.
5. Do not include explanations.
6. Do not include markdown.

Return only corrected SQL.
"""


def clean_sql_response(response: str) -> str:
    """
    Remove markdown formatting.
    """

    sql = response.strip()

    if sql.startswith("```sql"):
        sql = sql[6:]

    elif sql.startswith("```"):
        sql = sql[3:]

    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


def generate_customer_sql(
    question: str,
    schema: str,
) -> str:
    """
    Generate initial customer SQL.
    """

    prompt = build_customer_sql_prompt(
        question=question,
        schema=schema,
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def correct_customer_sql(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Correct failed customer SQL.
    """

    prompt = build_customer_correction_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        error_message=error_message,
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def correct_result_shape(
    question: str,
    schema: str,
    sql: str,
    result,
) -> str:
    """
    Correct SQL when the output shape does not match
    the user's question.
    """

    prompt = build_shape_correction_prompt(
        question=question,
        schema=schema,
        sql=sql,
        result_text=result.to_string(index=False),
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def result_shape_is_valid(
    question: str,
    result,
) -> bool:
    """
    Basic result-shape validation.

    If the question asks for a count, we expect one row.
    """

    normalized_question = question.lower()

    count_phrases = [
        "how many",
        "what is the count",
        "count of",
        "number of",
    ]

    asks_for_count = any(
        phrase in normalized_question
        for phrase in count_phrases
    )

    if asks_for_count:
        return len(result) == 1

    return True


def analyze_customer_question(question: str):
    """
    Customer Insights Agent V2 workflow.

    Question
        ↓
    Schema
        ↓
    Qwen SQL
        ↓
    DuckDB
        ↓
    SQL correction if needed
        ↓
    Result-shape validation
        ↓
    Logical result validation
        ↓
    Final validated result
    """

    print("\n" + "=" * 70)
    print("CUSTOMER INSIGHTS AGENT - VERSION 2")
    print("RESULT-SHAPE AWARE CUSTOMER ANALYTICS")
    print("=" * 70)

    print("\nCustomer Question:")
    print(question)

    print("\nRetrieving database schema...")

    schema = get_database_schema()

    print(
        "Generating customer SQL using "
        "Qwen 2.5 7B..."
    )

    sql = generate_customer_sql(
        question=question,
        schema=schema,
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print(f"\nSQL Attempt {attempt}:")
        print("-" * 70)
        print(sql)
        print("-" * 70)

        try:
            result = execute_sql(sql)

            print("\nQuery Result:")
            print("-" * 70)
            print(result)
            print("-" * 70)

            # ------------------------------------------------
            # Result-shape validation
            # ------------------------------------------------

            print("\nChecking result shape...")

            if not result_shape_is_valid(
                question=question,
                result=result,
            ):

                print(
                    "\nResult shape does not match "
                    "the user's question."
                )

                if attempt == MAX_ATTEMPTS:
                    print(
                        "\nMaximum SQL attempts reached."
                    )
                    return None

                print(
                    "\nSending result-shape issue "
                    "back to Qwen..."
                )

                sql = correct_result_shape(
                    question=question,
                    schema=schema,
                    sql=sql,
                    result=result,
                )

                continue

            print("Result shape is valid.")

            # ------------------------------------------------
            # Logical result validation
            # ------------------------------------------------

            print("\nValidating customer result...")

            validation = validate_result(
                question=question,
                sql=sql,
                result=result,
            )

            print("\nValidation Result:")
            print(validation["reason"])

            if validation["is_valid"]:

                print(
                    f"\nCustomer query validated "
                    f"successfully on attempt {attempt}."
                )

                return result

            print("\nCustomer result validation failed.")

            return None

        except Exception as error:

            print("\nSQL execution failed.")

            print("\nDuckDB Error:")
            print(error)

            if attempt == MAX_ATTEMPTS:

                print(
                    "\nMaximum SQL attempts reached."
                )

                return None

            print(
                "\nSending failed SQL and DuckDB "
                "error back to Qwen..."
            )

            sql = correct_customer_sql(
                question=question,
                schema=schema,
                failed_sql=sql,
                error_message=str(error),
            )

    return None


# ============================================================
# TEST CUSTOMER INSIGHTS AGENT VERSION 2
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Customer Insights Agent Version 2"
    )

    test_question = (
        "How many repeat customers do we have?"
    )

    result = analyze_customer_question(
        test_question
    )

    if result is not None:

        print("\nFinal Result:")
        print(result)

        print(
            "\nCustomer Insights Agent Version 2 "
            "test completed successfully."
        )

    else:

        print(
            "\nCustomer Insights Agent Version 2 "
            "test failed."
        )