from tools.sql_tool import execute_sql
from tools.llm_tool import ask_llm
from tools.schema_tool import get_database_schema
from tools.result_validator import validate_result


# ============================================================
# CUSTOMER INSIGHTS AGENT - VERSION 1
# CUSTOMER-SPECIFIC SQL + SELF-CORRECTION
# ============================================================


MAX_ATTEMPTS = 3


def build_customer_sql_prompt(
    question: str,
    schema: str,
) -> str:
    """
    Build a customer-focused SQL generation prompt.
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

To use customer_unique_id, you MUST join:

orders.customer_id = customers.customer_id

Then use:

customers.customer_unique_id

For repeat-customer analysis, the correct pattern is:

SELECT
    c.customer_unique_id,
    COUNT(DISTINCT o.order_id) AS order_count
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
GROUP BY c.customer_unique_id
HAVING COUNT(DISTINCT o.order_id) > 1

CUSTOMER ANALYTICS GUIDANCE:

1. customers.customer_id is the order-level customer identifier.

2. customers.customer_unique_id is the stable customer identity
   that should be used across multiple orders.

3. orders.customer_id joins to:
   customers.customer_id

4. customer_unique_id must always come from:
   customers.customer_unique_id

5. For repeat-customer analysis:
   - join orders to customers
   - group by customers.customer_unique_id
   - count distinct orders
   - keep customers with more than one order

6. Reviews connect through:
   reviews.order_id = orders.order_id

7. Payments connect through:
   payments.order_id = orders.order_id

8. Customer geography is available through:
   customers.customer_city
   customers.customer_state

9. Do not invent:
   - churn labels
   - customer lifetime value fields
   - demographic fields
   - income
   - age
   - gender
   - customer segments
   unless they are explicitly present in the schema.

10. Avoid unnecessary joins that can duplicate rows.

RULES:

1. Use only tables and columns present in DATABASE SCHEMA.
2. Generate DuckDB-compatible SQL.
3. Do not invent tables or columns.
4. Use correct JOIN conditions.
5. Generate only SELECT or WITH queries.
6. Do not modify the database.
7. Return only executable SQL.
8. Do not include explanations.
9. Do not include markdown.
10. Avoid SELECT * unless explicitly requested.
11. Verify every table and column before returning.
12. If using customer_unique_id, it MUST come from customers.
13. Never reference orders.customer_unique_id because it does not exist.

Return only SQL.
"""


def build_customer_correction_prompt(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Repair customer SQL after a DuckDB execution error.
    """

    return f"""
You are an expert DuckDB SQL debugger specializing in
customer analytics.

A customer analytics query failed during execution.

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

If DuckDB says a column does not exist,
DO NOT use that column from that table again.

CRITICAL CUSTOMER IDENTITY RULE:

customer_unique_id does NOT exist in orders.

The correct relationship is:

orders.customer_id
    =
customers.customer_id

Then use:

customers.customer_unique_id

For repeat-customer analysis, use this logic:

SELECT
    c.customer_unique_id,
    COUNT(DISTINCT o.order_id) AS order_count
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
GROUP BY c.customer_unique_id
HAVING COUNT(DISTINCT o.order_id) > 1

If the failed SQL contains:

orders.customer_unique_id

or references customer_unique_id directly from orders,

you MUST replace that logic by joining customers.

CUSTOMER RELATIONSHIPS:

orders.customer_id
    -> customers.customer_id

reviews.order_id
    -> orders.order_id

payments.order_id
    -> orders.order_id

RULES:

1. Fix the exact DuckDB error.
2. Preserve the original customer business question.
3. Use only valid schema fields.
4. Do not invent replacement columns.
5. Generate DuckDB-compatible SQL.
6. Generate only SELECT or WITH queries.
7. Do not modify the database.
8. Return only corrected SQL.
9. Do not include explanations.
10. Do not include markdown.
11. Verify every JOIN before returning.
12. Verify every column against the supplied schema.
13. customer_unique_id must come from customers.
14. Do not return the same invalid SQL again.

Return only corrected SQL.
"""


def build_customer_validation_correction_prompt(
    question: str,
    schema: str,
    failed_sql: str,
    validation_reason: str,
) -> str:
    """
    Repair customer SQL that executed but produced
    a logically invalid result.
    """

    return f"""
You are an expert customer analytics SQL debugger.

The SQL below executed successfully, but the result
validator determined that the output is logically incorrect.

BUSINESS QUESTION:
{question}

DATABASE SCHEMA:
{schema}

SQL QUERY:
{failed_sql}

VALIDATION FAILURE:
{validation_reason}

CUSTOMER IDENTITY RULE:

customer_unique_id exists only in customers.

Use:

orders.customer_id = customers.customer_id

Then reference:

customers.customer_unique_id

For repeat customers:

1. Join orders to customers.
2. Group by customers.customer_unique_id.
3. Count DISTINCT orders.order_id.
4. Keep customers with more than one order.
5. Count those unique customers.

RULES:

1. Preserve the original business question.
2. Use only tables and columns present in the schema.
3. Fix incorrect JOIN or aggregation logic.
4. Avoid duplicate row amplification.
5. Do not invent customer fields.
6. Generate DuckDB-compatible SQL.
7. Generate only SELECT or WITH queries.
8. Do not modify the database.
9. Return only executable SQL.
10. Do not include explanations.
11. Do not include markdown.
12. Verify the corrected result logic before returning.

Return only corrected SQL.
"""


def clean_sql_response(response: str) -> str:
    """
    Remove markdown formatting from an LLM SQL response.
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
    Generate initial customer-focused SQL.
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
    Correct SQL that failed during DuckDB execution.
    """

    prompt = build_customer_correction_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        error_message=error_message,
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def correct_invalid_customer_result(
    question: str,
    schema: str,
    failed_sql: str,
    validation_reason: str,
) -> str:
    """
    Correct SQL that executed successfully but
    produced a logically invalid customer result.
    """

    prompt = build_customer_validation_correction_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        validation_reason=validation_reason,
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def analyze_customer_question(question: str):
    """
    Complete Customer Insights Agent workflow.

    Customer Question
        ↓
    Retrieve Schema
        ↓
    Qwen Generates Customer SQL
        ↓
    DuckDB Execution
        ↓

    If SQL execution fails:
        DuckDB Error
            ↓
        Qwen SQL Correction
            ↓
        Retry

    If SQL execution succeeds:
        Result Validator
            ↓

        Valid:
            Return Result

        Invalid:
            Validation Feedback
                ↓
            Qwen Logical Correction
                ↓
            Retry
    """

    print("\n" + "=" * 70)
    print("CUSTOMER INSIGHTS AGENT - VERSION 1")
    print("SELF-CORRECTING CUSTOMER ANALYTICS AGENT")
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

        print("\nExecuting customer SQL against DuckDB...")

        try:
            # ------------------------------------------------
            # Execute SQL
            # ------------------------------------------------

            result = execute_sql(sql)

            print("\nQuery Result:")
            print("-" * 70)
            print(result)
            print("-" * 70)

            # ------------------------------------------------
            # Validate result
            # ------------------------------------------------

            print("\nValidating customer query result...")

            validation = validate_result(
                question=question,
                sql=sql,
                result=result,
            )

            print("\nValidation Result:")
            print(validation["reason"])

            # ------------------------------------------------
            # Valid result
            # ------------------------------------------------

            if validation["is_valid"]:

                print(
                    f"\nCustomer query and result "
                    f"validated successfully "
                    f"on attempt {attempt}."
                )

                return result

            # ------------------------------------------------
            # Logically invalid result
            # ------------------------------------------------

            print(
                "\nCustomer result validation failed."
            )

            if attempt == MAX_ATTEMPTS:

                print(
                    "\nMaximum SQL attempts reached."
                )

                print(
                    "The Customer Insights Agent could "
                    "not produce a valid result."
                )

                return None

            print(
                "\nSending SQL and validation failure "
                "back to Qwen for correction..."
            )

            sql = correct_invalid_customer_result(
                question=question,
                schema=schema,
                failed_sql=sql,
                validation_reason=validation["reason"],
            )

            continue

        except Exception as error:

            # ------------------------------------------------
            # DuckDB execution error
            # ------------------------------------------------

            print("\nSQL execution failed.")

            print("\nDuckDB Error:")
            print(error)

            if attempt == MAX_ATTEMPTS:

                print(
                    "\nMaximum SQL attempts reached."
                )

                print(
                    "The Customer Insights Agent could "
                    "not produce executable SQL."
                )

                return None

            print(
                "\nSending failed customer SQL and "
                "DuckDB error back to Qwen "
                "for correction..."
            )

            sql = correct_customer_sql(
                question=question,
                schema=schema,
                failed_sql=sql,
                error_message=str(error),
            )

    return None


# ============================================================
# TEST CUSTOMER INSIGHTS AGENT
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Customer Insights Agent Version 1"
    )

    test_question = (
        "How many repeat customers do we have?"
    )

    result = analyze_customer_question(
        test_question
    )

    if result is not None:

        print(
            "\nCustomer Insights Agent Version 1 "
            "test completed successfully."
        )

    else:

        print(
            "\nCustomer Insights Agent Version 1 "
            "test failed."
        )