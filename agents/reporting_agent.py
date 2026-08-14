from tools.llm_tool import ask_llm


# ============================================================
# REPORTING AGENT - VERSION 1
# EXECUTIVE REPORT GENERATION
# ============================================================


def build_reporting_prompt(
    question: str,
    analytical_result: str,
    business_insights: str,
) -> str:
    """
    Build a prompt that converts validated analytics and
    business insights into an executive-ready report.
    """

    return f"""
You are a Reporting Agent in a multi-agent enterprise
intelligence system.

Your job is to convert validated analytical results and
business insights into a concise executive report.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

BUSINESS INSIGHTS:
{business_insights}

REPORTING RULES:

1. Use only the information provided above.
2. Do not invent additional metrics or percentages.
3. Do not claim causation unless explicitly supported.
4. Do not mention SQL, DuckDB, database schemas,
   prompts, LLMs, or technical implementation.
5. Clearly answer the original business question.
6. Keep the report concise and executive-friendly.
7. Preserve important numeric results exactly.
8. Distinguish facts from recommendations.
9. Avoid unsupported claims such as:
   - "majority of revenue"
   - "high customer loyalty"
   - "strong market demand"
   unless the supplied data directly proves them.
10. If the available result is only a count,
    do not infer trends, growth, or rates.
11. Recommended actions should be framed as areas
    for further investigation when the data does not
    directly support a decision.

Return the report using EXACTLY this structure:

EXECUTIVE SUMMARY
<2-3 sentence summary>

KEY RESULTS
- <result 1>
- <result 2 if relevant>
- <result 3 if relevant>

BUSINESS IMPLICATION
<short grounded interpretation>

RECOMMENDED NEXT STEPS
- <next step 1>
- <next step 2>
"""


def generate_report(
    question: str,
    analytical_result,
    business_insights: str,
) -> str:
    """
    Generate an executive report from validated
    analytical output and business insights.
    """

    if analytical_result is None:
        return (
            "A report cannot be generated because no "
            "validated analytical result was provided."
        )

    if business_insights is None:
        return (
            "A report cannot be generated because no "
            "business insights were provided."
        )

    # Convert DataFrame result to readable text.
    if hasattr(analytical_result, "to_string"):
        result_text = analytical_result.to_string(
            index=False
        )
    else:
        result_text = str(analytical_result)

    prompt = build_reporting_prompt(
        question=question,
        analytical_result=result_text,
        business_insights=business_insights,
    )

    response = ask_llm(prompt)

    return response.strip()


# ============================================================
# TEST REPORTING AGENT
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    print("Testing Reporting Agent Version 1")
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

    test_insights = """
Key Finding:
Health & Beauty is the highest-revenue category,
followed by Watches & Gifts.

Business Interpretation:
The ranking identifies the categories contributing
the largest product revenue among the categories shown.

Recommended Actions:
- Review inventory and merchandising priorities for
  leading categories.
- Investigate cross-selling opportunities using
  customer-level purchase data.
"""

    report = generate_report(
        question=test_question,
        analytical_result=test_result,
        business_insights=test_insights,
    )

    print("\nExecutive Report:")
    print("-" * 70)
    print(report)
    print("-" * 70)

    print(
        "\nReporting Agent Version 1 "
        "test completed successfully."
    )