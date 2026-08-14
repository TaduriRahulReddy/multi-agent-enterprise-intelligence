from tools.sql_tool import execute_sql
from tools.llm_tool import ask_llm
from tools.schema_tool import get_database_schema
from tools.result_validator import validate_result


# ============================================================
# DATA ANALYST AGENT - VERSION 4
# STRONGER LOGICAL SQL CORRECTION
# ============================================================


MAX_ATTEMPTS = 3


def build_sql_prompt(
    question: str,
    schema: str,
) -> str:
    """
    Build the initial SQL-generation prompt.
    """

    return f"""
You are an expert data analyst working with DuckDB.

Convert the user's business question into a valid
DuckDB SQL query.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

IMPORTANT DATABASE RULES:

1. Use only tables and columns present in the schema.

2. For product revenue, use:

   SUM(order_items.price)

3. order_items does NOT contain a quantity column.

4. Product category is stored in:

   products.product_category_name

5. English product category is stored in:

   category_translation.product_category_name_english

6. The correct translation relationship is:

   products.product_category_name =
   category_translation.product_category_name

7. When analyzing product-category revenue,
   join category_translation DIRECTLY to products.

8. Do NOT create a category-translation CTE by joining
   products into a mapping and then joining that mapping
   back to products.

9. Avoid joins that can multiply order_items rows.

10. Each order_items row should contribute its price
    exactly once to product revenue.

GENERAL RULES:

1. Generate DuckDB-compatible SQL.
2. Do not invent tables or columns.
3. Use explicit JOIN conditions.
4. Generate only SELECT or WITH queries.
5. Do not modify the database.
6. Do not include explanations.
7. Do not include markdown.
8. Return only executable SQL.
9. Avoid SELECT * unless explicitly requested.
10. Verify all joins and aggregations before returning.

Return only SQL.
"""


def build_execution_correction_prompt(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Repair SQL after a DuckDB execution error.
    """

    return f"""
You are an expert DuckDB SQL debugger.

A SQL query failed during execution.

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

If a column or table does not exist,
do not use it again.

For product-category revenue:

- Use order_items.price.
- Do not use quantity.
- Join order_items to products using product_id.
- Join category_translation directly to products using
  product_category_name.

Correct pattern:

SELECT
    ct.product_category_name_english,
    SUM(oi.price) AS total_revenue
FROM order_items oi
JOIN products p
    ON oi.product_id = p.product_id
LEFT JOIN category_translation ct
    ON p.product_category_name =
       ct.product_category_name
GROUP BY
    ct.product_category_name_english
ORDER BY
    total_revenue DESC

RULES:

1. Fix the specific error.
2. Preserve the original business intent.
3. Use only schema fields.
4. Do not invent columns.
5. Generate DuckDB-compatible SQL.
6. Generate only SELECT or WITH queries.
7. Do not modify the database.
8. Return only corrected SQL.
9. Do not include explanations.
10. Do not include markdown.

Return only corrected SQL.
"""


def build_logical_correction_prompt(
    question: str,
    schema: str,
    failed_sql: str,
    validation_reason: str,
) -> str:
    """
    Repair SQL that executed successfully but produced
    a logically invalid analytical result.
    """

    return f"""
You are an expert data analyst debugging a LOGICALLY
INCORRECT DuckDB query.

The SQL query executed successfully, but the result
validator rejected the output.

BUSINESS QUESTION:
{question}

DATABASE SCHEMA:
{schema}

FAILED SQL:
{failed_sql}

VALIDATION FAILURE:
{validation_reason}

CRITICAL INSTRUCTION:

The previous SQL likely caused duplicate row amplification.

You MUST change the query structure.

DO NOT simply return the same SQL again.

For product-category revenue:

1. Start from order_items.

2. Join products exactly once:

   order_items.product_id = products.product_id

3. Join category_translation exactly once:

   products.product_category_name =
   category_translation.product_category_name

4. Do NOT create a translation CTE that joins products
   into the category mapping.

5. Do NOT join a product-derived category mapping back
   to products by category.

6. Every order_items row must contribute its price
   exactly once.

7. Revenue must be:

   SUM(order_items.price)

CORRECT STRUCTURE:

SELECT
    ct.product_category_name_english AS category,
    SUM(oi.price) AS total_revenue
FROM order_items oi
JOIN products p
    ON oi.product_id = p.product_id
LEFT JOIN category_translation ct
    ON p.product_category_name =
       ct.product_category_name
GROUP BY
    ct.product_category_name_english
ORDER BY
    total_revenue DESC
LIMIT 5

IMPORTANT:

If the validator says a category revenue exceeds total
product revenue, the query is mathematically impossible
and usually contains row multiplication.

You MUST remove the source of duplication.

RULES:

1. Preserve the original business question.
2. Use only valid schema fields.
3. Do not return the same failed SQL.
4. Remove unnecessary joins and CTEs.
5. Avoid many-to-many or duplicated joins.
6. Generate DuckDB-compatible SQL.
7. Generate only SELECT or WITH queries.
8. Do not modify the database.
9. Return only corrected SQL.
10. Do not include explanations.
11. Do not include markdown.

Return only corrected SQL.
"""


def clean_sql_response(
    response: str,
) -> str:
    """
    Remove markdown formatting from SQL returned by Qwen.
    """

    sql = response.strip()

    if sql.startswith("```sql"):
        sql = sql[6:]

    elif sql.startswith("```"):
        sql = sql[3:]

    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


def generate_sql(
    question: str,
    schema: str,
) -> str:
    """
    Generate initial SQL.
    """

    prompt = build_sql_prompt(
        question=question,
        schema=schema,
    )

    response = ask_llm(
        prompt
    )

    return clean_sql_response(
        response
    )


def correct_execution_error(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Correct SQL after a DuckDB execution error.
    """

    prompt = build_execution_correction_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        error_message=error_message,
    )

    response = ask_llm(
        prompt
    )

    return clean_sql_response(
        response
    )


def correct_logical_error(
    question: str,
    schema: str,
    failed_sql: str,
    validation_reason: str,
) -> str:
    """
    Correct SQL after logical result validation fails.
    """

    prompt = build_logical_correction_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        validation_reason=validation_reason,
    )

    response = ask_llm(
        prompt
    )

    return clean_sql_response(
        response
    )


def analyze(
    question: str,
):
    """
    Data Analyst V4 workflow.

    Question
        ↓
    Actual Schema
        ↓
    Qwen SQL Generation
        ↓
    DuckDB
        ↓
    Execution Error?
        ├── Yes → Repair SQL → Retry
        └── No
               ↓
         Result Validation
               ↓
         Logically Valid?
        ├── Yes → Return
        └── No → Structural SQL Repair → Retry
    """

    print("\n" + "=" * 70)
    print("DATA ANALYST AGENT - VERSION 4")
    print("STRUCTURAL SQL SELF-CORRECTION")
    print("=" * 70)

    print("\nBusiness Question:")
    print(question)

    print("\nRetrieving database schema...")

    schema = get_database_schema()

    print(
        "Generating initial SQL using "
        "Qwen 2.5 7B..."
    )

    sql = generate_sql(
        question=question,
        schema=schema,
    )

    previous_sql = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        print(
            f"\nSQL Attempt {attempt}:"
        )

        print("-" * 70)
        print(sql)
        print("-" * 70)

        # ----------------------------------------------------
        # Prevent identical retries
        # ----------------------------------------------------

        if (
            previous_sql is not None
            and sql.strip() == previous_sql.strip()
        ):

            print(
                "\nQwen returned the same failed SQL again."
            )

            if attempt == MAX_ATTEMPTS:

                print(
                    "\nMaximum SQL attempts reached."
                )

                return None

        previous_sql = sql

        print(
            "\nExecuting SQL against DuckDB..."
        )

        try:
            result = execute_sql(
                sql
            )

        except Exception as error:

            print(
                "\nSQL execution failed."
            )

            print(
                "\nDuckDB Error:"
            )

            print(
                error
            )

            if attempt == MAX_ATTEMPTS:

                print(
                    "\nMaximum SQL attempts reached."
                )

                return None

            print(
                "\nSending failed SQL and DuckDB error "
                "back to Qwen for correction..."
            )

            sql = correct_execution_error(
                question=question,
                schema=schema,
                failed_sql=sql,
                error_message=str(error),
            )

            continue

        # ----------------------------------------------------
        # SQL executed
        # ----------------------------------------------------

        print(
            "\nQuery Result:"
        )

        print("-" * 70)
        print(result)
        print("-" * 70)

        print(
            "\nValidating query result..."
        )

        validation = validate_result(
            question=question,
            sql=sql,
            result=result,
        )

        print(
            "\nValidation Result:"
        )

        print(
            validation["reason"]
        )

        # ----------------------------------------------------
        # VALID RESULT
        # ----------------------------------------------------

        if validation["is_valid"]:

            print(
                f"\nQuery and result validated "
                f"successfully on attempt {attempt}."
            )

            return result

        # ----------------------------------------------------
        # LOGICAL FAILURE
        # ----------------------------------------------------

        print(
            "\nResult validation failed."
        )

        if attempt == MAX_ATTEMPTS:

            print(
                "\nMaximum SQL attempts reached."
            )

            print(
                "The agent could not produce a "
                "logically valid result."
            )

            return None

        print(
            "\nSending SQL and logical validation "
            "failure back to Qwen..."
        )

        print(
            "Requesting structural SQL correction..."
        )

        sql = correct_logical_error(
            question=question,
            schema=schema,
            failed_sql=sql,
            validation_reason=validation["reason"],
        )

    return None


# ============================================================
# TEST DATA ANALYST AGENT VERSION 4
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Data Analyst Agent Version 4"
    )

    test_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    result = analyze(
        test_question
    )

    if result is not None:

        print("\nFinal Result:")
        print(result)

        print(
            "\nData Analyst Agent Version 4 "
            "test completed successfully."
        )

    else:

        print(
            "\nData Analyst Agent Version 4 "
            "test failed."
        )