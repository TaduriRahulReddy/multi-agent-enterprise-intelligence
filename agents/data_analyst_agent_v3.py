from tools.sql_tool import execute_sql
from tools.llm_tool import ask_llm
from tools.schema_tool import get_database_schema
from tools.result_validator import validate_result


# ============================================================
# DATA ANALYST AGENT - VERSION 3
# SELF-CORRECTING + RESULT-VALIDATING SQL AGENT
# ============================================================


MAX_ATTEMPTS = 3


def build_sql_prompt(question: str, schema: str) -> str:
    """
    Build the initial SQL-generation prompt using
    the real DuckDB schema.
    """

    return f"""
You are an expert data analyst working with DuckDB.

Convert the user's business question into a valid
DuckDB SQL query.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

RULES:

1. Use only tables and columns that exist in the schema.
2. Generate DuckDB-compatible SQL.
3. Do not invent tables or columns.
4. Use correct JOIN conditions.
5. Return only the SQL query.
6. Do not include explanations.
7. Do not include markdown code blocks.
8. Do not modify the database.
9. Generate only SELECT queries or WITH queries.
10. For product revenue, use order_items.price.
11. The order_items table does NOT contain a quantity column.
12. Each row in order_items represents an individual order item.
13. Product categories are stored in:
    products.product_category_name
14. For English category names, join category_translation
    using product_category_name.
15. Use:
    category_translation.product_category_name_english
    for the English category name.
16. Prefer explicit JOIN conditions.
17. Avoid SELECT * unless explicitly requested.
18. Avoid joins that unnecessarily duplicate rows.
19. Before returning, mentally validate every table,
    column, JOIN, and aggregation against the schema.

Return only SQL.
"""


def build_correction_prompt(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Repair SQL that failed during DuckDB execution.
    """

    return f"""
You are an expert DuckDB SQL debugger.

A SQL query failed during execution.

Your task is to fix the SQL using the actual database
schema and DuckDB error.

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

If DuckDB says a table or column does not exist,
DO NOT use that table or column again.

For this database:

order_items contains:
- order_id
- order_item_id
- product_id
- seller_id
- shipping_limit_date
- price
- freight_value

order_items DOES NOT contain a quantity column.

Each order_items row represents one order item.

For product revenue use:

SUM(order_items.price)

RULES:

1. Fix the specific DuckDB error.
2. Preserve the original business intent.
3. Use only valid tables and columns.
4. Do not invent replacement columns.
5. Generate DuckDB-compatible SQL.
6. Generate only SELECT or WITH queries.
7. Do not modify the database.
8. Return only SQL.
9. Do not include explanations.
10. Do not include markdown code blocks.
11. Verify every table and column against the schema
    before returning.

Return only corrected SQL.
"""


def build_validation_correction_prompt(
    question: str,
    schema: str,
    failed_sql: str,
    validation_reason: str,
) -> str:
    """
    Repair SQL that executed successfully but produced
    a logically invalid result.
    """

    return f"""
You are an expert DuckDB data analyst and SQL debugger.

The SQL query below executed successfully, but the
result validator determined that the result is logically
incorrect.

BUSINESS QUESTION:
{question}

DATABASE SCHEMA:
{schema}

SQL QUERY:
{failed_sql}

VALIDATION FAILURE:
{validation_reason}

Your task is to rewrite the SQL so that it correctly
answers the business question.

IMPORTANT RULES:

1. Use only tables and columns present in DATABASE SCHEMA.
2. Preserve the original business intent.
3. Carefully inspect JOIN logic.
4. Avoid duplicate row amplification caused by joins.
5. Carefully inspect aggregation logic.
6. Do not invent tables or columns.
7. Generate DuckDB-compatible SQL.
8. Generate only SELECT or WITH queries.
9. Do not modify the database.
10. Return only executable SQL.
11. Do not include explanations.
12. Do not include markdown code blocks.

IMPORTANT DATABASE KNOWLEDGE:

- order_items does NOT contain a quantity column.
- Each row in order_items represents one order item.
- Product revenue should use:

  SUM(order_items.price)

- Product category name is stored in:

  products.product_category_name

- English category name is stored in:

  category_translation.product_category_name_english

- Correct category translation JOIN:

  products.product_category_name =
  category_translation.product_category_name

- Do NOT create a category translation mapping by
  joining products into a translation CTE because
  multiple products share the same category and this
  can multiply rows.

Before returning the SQL:

1. Verify all columns exist.
2. Verify all JOIN conditions are necessary.
3. Verify JOINs do not create duplicate amplification.
4. Verify aggregation answers the original question.

Return only corrected SQL.
"""


def clean_sql_response(response: str) -> str:
    """
    Remove markdown formatting and extra whitespace
    from the LLM response.
    """

    sql = response.strip()

    if sql.startswith("```sql"):
        sql = sql[6:]

    elif sql.startswith("```"):
        sql = sql[3:]

    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


def generate_sql(question: str, schema: str) -> str:
    """
    Generate the initial SQL query.
    """

    prompt = build_sql_prompt(
        question=question,
        schema=schema,
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def correct_sql(
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """
    Correct SQL that failed during database execution.
    """

    prompt = build_correction_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        error_message=error_message,
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def correct_invalid_result(
    question: str,
    schema: str,
    failed_sql: str,
    validation_reason: str,
) -> str:
    """
    Correct SQL that executed successfully but produced
    a logically invalid result.
    """

    prompt = build_validation_correction_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        validation_reason=validation_reason,
    )

    response = ask_llm(prompt)

    return clean_sql_response(response)


def analyze(question: str):
    """
    Complete Version 3 agent workflow.

    User Question
        ↓
    Retrieve Real Database Schema
        ↓
    Qwen Generates SQL
        ↓
    Execute SQL in DuckDB
        ↓

    If SQL execution fails:
        DuckDB Error
            ↓
        Qwen SQL Correction
            ↓
        Retry

    If SQL executes:
        Result Validator
            ↓

        Valid:
            Return Result

        Invalid:
            Validation Reason
                ↓
            Qwen Logical SQL Correction
                ↓
            Retry
    """

    print("\n" + "=" * 70)
    print("DATA ANALYST AGENT - VERSION 3")
    print("SELF-CORRECTING + RESULT-VALIDATING SQL AGENT")
    print("=" * 70)

    print("\nBusiness Question:")
    print(question)

    print("\nRetrieving database schema...")

    schema = get_database_schema()

    print("Generating initial SQL using Qwen 2.5 7B...")

    sql = generate_sql(
        question=question,
        schema=schema,
    )

    # ========================================================
    # RETRY LOOP
    # ========================================================

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print(f"\nSQL Attempt {attempt}:")
        print("-" * 70)
        print(sql)
        print("-" * 70)

        print("\nExecuting SQL against DuckDB...")

        try:
            # ------------------------------------------------
            # Execute generated SQL
            # ------------------------------------------------

            result = execute_sql(sql)

            print("\nQuery Result:")
            print("-" * 70)
            print(result)
            print("-" * 70)

            # ------------------------------------------------
            # Validate result
            # ------------------------------------------------

            print("\nValidating query result...")

            validation = validate_result(
                question=question,
                sql=sql,
                result=result,
            )

            print("\nValidation Result:")
            print(validation["reason"])

            # ------------------------------------------------
            # Result is valid
            # ------------------------------------------------

            if validation["is_valid"]:

                print(
                    f"\nQuery and result validated "
                    f"successfully on attempt {attempt}."
                )

                return result

            # ------------------------------------------------
            # Result is logically invalid
            # ------------------------------------------------

            print("\nResult validation failed.")

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
                "\nSending SQL and validation failure "
                "back to Qwen for correction..."
            )

            sql = correct_invalid_result(
                question=question,
                schema=schema,
                failed_sql=sql,
                validation_reason=validation["reason"],
            )

            continue

        # ====================================================
        # SQL EXECUTION ERROR
        # ====================================================

        except Exception as error:

            print("\nSQL execution failed.")

            print("\nDuckDB Error:")
            print(error)

            if attempt == MAX_ATTEMPTS:

                print(
                    "\nMaximum SQL attempts reached."
                )

                print(
                    "The agent could not produce "
                    "executable SQL."
                )

                return None

            print(
                "\nSending failed SQL and DuckDB error "
                "back to Qwen for correction..."
            )

            sql = correct_sql(
                question=question,
                schema=schema,
                failed_sql=sql,
                error_message=str(error),
            )

    return None


# ============================================================
# VERSION 3 TEST
# ============================================================

if __name__ == "__main__":

    print("Testing Data Analyst Agent Version 3")

    test_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    result = analyze(test_question)

    if result is not None:

        print(
            "\nData Analyst Agent Version 3 "
            "test completed successfully."
        )

    else:

        print(
            "\nData Analyst Agent Version 3 "
            "test failed."
        )