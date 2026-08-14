from tools.llm_tool import ask_llm


# ============================================================
# BUSINESS INSIGHTS AGENT - VERSION 1
# ============================================================


def build_insights_prompt(
    question: str,
    analytical_result: str,
) -> str:
    """
    Build a prompt that converts validated analytical
    output into a business-facing explanation.
    """

    return f"""
You are a Business Insights Agent in a multi-agent
enterprise intelligence system.

Your job is to convert validated analytical results
into a concise business explanation.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

INSTRUCTIONS:

1. Use only the information provided in the analytical result.
2. Do not invent additional numbers.
3. Do not claim causation unless the data supports it.
4. Clearly identify the most important finding.
5. Mention relevant ranking or comparison when available.
6. Add 1-2 practical business implications.
7. Keep the response concise and executive-friendly.
8. Do not mention SQL, DuckDB, schemas, or technical implementation.
9. Do not use markdown tables.
10. Avoid overly generic recommendations.

Return the answer using this structure:

Key Finding:
<concise finding>

Business Interpretation:
<short explanation>

Recommended Actions:
- <action 1>
- <action 2>
"""


def generate_insights(
    question: str,
    analytical_result,
) -> str:
    """
    Convert analytical results into business insights.
    """

    if analytical_result is None:
        return (
            "No validated analytical result was provided, "
            "so business insights cannot be generated."
        )

    result_text = analytical_result.to_string(
        index=False
    )

    prompt = build_insights_prompt(
        question=question,
        analytical_result=result_text,
    )

    response = ask_llm(prompt)

    return response.strip()


# ============================================================
# TEST BUSINESS INSIGHTS AGENT
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    print("Testing Business Insights Agent")
    print("=" * 70)

    test_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    test_result = pd.DataFrame(
        {
            "product_category_name_english": [
                "health_beauty",
                "watches_gifts",
                "bed_bath_table",
                "sports_leisure",
                "computers_accessories",
            ],
            "total_revenue": [
                1258681.34,
                1205005.68,
                1036988.68,
                988048.97,
                911954.32,
            ],
        }
    )

    insights = generate_insights(
        question=test_question,
        analytical_result=test_result,
    )

    print("\nBusiness Insights:")
    print("-" * 70)
    print(insights)
    print("-" * 70)

    print(
        "\nBusiness Insights Agent "
        "Version 1 test completed."
    )