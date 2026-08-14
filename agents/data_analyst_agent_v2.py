from tools.sql_tool import execute_sql
from tools.llm_tool import ask_llm
from tools.schema_tool import get_database_schema


# ============================================================
# DATA ANALYST AGENT - VERSION 2
# ============================================================


def build_sql_prompt(question: str, schema: str) -> str:
    """
    Build a prompt for the local LLM using the
    actual DuckDB database schema.
    """

    prompt = f"""
You are an expert data analyst working with DuckDB.

Your task is to convert a user's business question into
a valid DuckDB SQL query.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

IMPORTANT RULES:

1. Use only tables and columns that exist in the schema.
2. Generate DuckDB-compatible SQL.
3. Do not invent tables or columns.
4. Use appropriate JOIN conditions.
5. Return only the SQL query.
6. Do not include explanations.
7. Do not include markdown code blocks.
8. Do not modify, delete, insert, update, or drop data.
9. Generate only SELECT queries or WITH queries.
10. When calculating product revenue, use the price column
    from the order_items table.
11. product_category_name is stored in the products table.
12. If English category names are needed, join
    category_translation using product_category_name.

Return only SQL.
"""

    return prompt


def clean_sql_response(response: str) -> str:
    """
    Clean possible markdown formatting returned by the LLM.
    """

    sql = response.strip()

    if sql.startswith("```sql"):
        sql = sql[6:]

    elif sql.startswith("```"):
        sql = sql[3:]

    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


def generate_sql(question: str) -> str:
    """
    Convert a natural-language question into SQL.
    """

    schema = get_database_schema()

    prompt = build_sql_prompt(
        question=question,
        schema=schema
    )

    response = ask_llm(prompt)

    sql = clean_sql_response(response)

    return sql


def analyze(question: str):
    """
    Complete Version 2 workflow.

    User Question
        ↓
    Actual Database Schema
        ↓
    Qwen Local LLM
        ↓
    SQL Generation
        ↓
    SQL Tool
        ↓
    DuckDB
        ↓
    Query Result
    """

    print("\n" + "=" * 70)
    print("DATA ANALYST AGENT - VERSION 2")
    print("=" * 70)

    print("\nBusiness Question:")
    print(question)

    print("\nRetrieving database schema...")

    print("Generating SQL using Qwen 2.5 7B...")

    sql = generate_sql(question)

    print("\nGenerated SQL:")
    print("-" * 70)
    print(sql)
    print("-" * 70)

    print("\nExecuting SQL against DuckDB...")

    try:
        result = execute_sql(sql)

        print("\nQuery Result:")
        print("-" * 70)
        print(result)
        print("-" * 70)

        return result

    except Exception as error:

        print("\nSQL execution failed.")

        print("\nError:")
        print(error)

        return None


# ============================================================
# TEST VERSION 2
# ============================================================

if __name__ == "__main__":

    print("Testing Data Analyst Agent Version 2")

    test_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    result = analyze(test_question)

    if result is not None:

        print(
            "\nData Analyst Agent Version 2 "
            "test completed successfully."
        )