from tools.llm_tool import ask_llm
from tools.report_validator import validate_report


# ============================================================
# REPORTING AGENT - VERSION 2
# GROUNDED REPORT GENERATION + AUTOMATIC REWRITE
# ============================================================


MAX_ATTEMPTS = 3


def build_reporting_prompt(
    question: str,
    analytical_result: str,
    business_insights: str,
) -> str:
    """
    Build the initial executive-report prompt.
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

1. Use only information supported by the validated
   analytical result.

2. Do not invent numbers, percentages, trends,
   benchmarks, or comparisons.

3. Do not claim causation unless explicitly supported.

4. Do not make unsupported qualitative claims such as:
   - strong customer loyalty
   - significant portion of customers
   - high demand
   - strong retention
   - strong market performance
   - majority of customers
   - high customer satisfaction

5. If the data only contains an absolute count,
   report the count without describing it as high,
   low, strong, weak, significant, or substantial.

6. If the data only contains a ranking, describe the
   ranking without inferring demand, preference,
   profitability, or market strength.

7. Preserve all important numerical results exactly.

8. Recommendations may suggest additional analysis,
   validation, experimentation, or investigation.

9. Recommendations must not depend on unsupported claims.

10. Do not mention SQL, DuckDB, LLMs, prompts,
    database schemas, or technical implementation.

Return EXACTLY this structure:

EXECUTIVE SUMMARY
<2-3 sentence grounded summary>

KEY RESULTS
- <supported result 1>
- <supported result 2 if relevant>
- <supported result 3 if relevant>

BUSINESS IMPLICATION
<grounded interpretation only>

RECOMMENDED NEXT STEPS
- <next step 1>
- <next step 2>
"""


def build_rewrite_prompt(
    question: str,
    analytical_result: str,
    business_insights: str,
    failed_report: str,
    validation_reason: str,
) -> str:
    """
    Ask the LLM to rewrite an ungrounded report.
    """

    return f"""
You are a senior enterprise reporting editor.

The executive report below failed grounding validation.

Your job is to rewrite the report so that EVERY factual
claim is directly supported by the validated analytical result.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

BUSINESS INSIGHTS:
{business_insights}

FAILED REPORT:
{failed_report}

VALIDATION FAILURE:
{validation_reason}

STRICT REWRITE RULES:

1. Remove every unsupported factual claim.

2. If the analytical result only contains an absolute count,
   state the count only.

3. Do not describe an absolute count as:
   - strong
   - weak
   - high
   - low
   - significant
   - substantial
   - loyal
   - strong retention
   - high demand

4. If a ranking is provided, only state the ranking
   and corresponding supported values.

5. Do not infer:
   - profitability
   - customer preference
   - customer loyalty strength
   - demand strength
   - market share
   - marketing effectiveness

6. Recommendations may suggest further analysis.

7. Recommendations must not rely on unsupported premises.

8. Preserve validated numbers exactly.

9. Do not mention technical implementation.

10. Return only the rewritten report.

Return EXACTLY this structure:

EXECUTIVE SUMMARY
<grounded summary>

KEY RESULTS
- <supported result>

BUSINESS IMPLICATION
<grounded interpretation>

RECOMMENDED NEXT STEPS
- <next step 1>
- <next step 2>
"""


def _result_to_text(analytical_result) -> str:
    """
    Convert analytical output into readable text.
    """

    if hasattr(analytical_result, "to_string"):
        return analytical_result.to_string(
            index=False
        )

    return str(analytical_result)


def generate_initial_report(
    question: str,
    analytical_result,
    business_insights: str,
) -> str:
    """
    Generate the first executive report.
    """

    result_text = _result_to_text(
        analytical_result
    )

    prompt = build_reporting_prompt(
        question=question,
        analytical_result=result_text,
        business_insights=business_insights,
    )

    response = ask_llm(prompt)

    return response.strip()


def rewrite_report(
    question: str,
    analytical_result,
    business_insights: str,
    failed_report: str,
    validation_reason: str,
) -> str:
    """
    Rewrite a report after grounding validation fails.
    """

    result_text = _result_to_text(
        analytical_result
    )

    prompt = build_rewrite_prompt(
        question=question,
        analytical_result=result_text,
        business_insights=business_insights,
        failed_report=failed_report,
        validation_reason=validation_reason,
    )

    response = ask_llm(prompt)

    return response.strip()


def generate_grounded_report(
    question: str,
    analytical_result,
    business_insights: str,
) -> str:
    """
    Generate, validate, and if necessary rewrite
    the executive report.
    """

    if analytical_result is None:

        return (
            "A report cannot be generated because no "
            "validated analytical result was provided."
        )

    if not business_insights:

        return (
            "A report cannot be generated because no "
            "business insights were provided."
        )

    print("\nGenerating initial executive report...")

    report = generate_initial_report(
        question=question,
        analytical_result=analytical_result,
        business_insights=business_insights,
    )

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        print(
            f"\nReport Validation Attempt {attempt}:"
        )

        print("-" * 70)
        print(report)
        print("-" * 70)

        validation = validate_report(
            question=question,
            analytical_result=analytical_result,
            executive_report=report,
        )

        print("\nGrounding Validation:")
        print(validation)

        if validation["is_valid"]:

            print(
                f"\nExecutive report validated "
                f"successfully on attempt {attempt}."
            )

            return report

        if attempt == MAX_ATTEMPTS:

            print(
                "\nMaximum report-rewrite attempts reached."
            )

            return (
                "Unable to produce a fully grounded "
                "executive report."
            )

        print(
            "\nReport is not fully grounded."
        )

        print(
            "Sending validation feedback back to "
            "Qwen for rewrite..."
        )

        report = rewrite_report(
            question=question,
            analytical_result=analytical_result,
            business_insights=business_insights,
            failed_report=report,
            validation_reason=validation["reason"],
        )

    return report


# ============================================================
# TEST REPORTING AGENT VERSION 2
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Reporting Agent Version 2"
    )

    print("=" * 70)

    test_question = (
        "How many repeat customers do we have?"
    )

    test_result = pd.DataFrame(
        {
            "repeat_customer_count": [
                2997
            ]
        }
    )

    # Deliberately includes unsupported statements
    # so we can test the grounding-rewrite loop.
    test_insights = """
Key Finding:
We have 2,997 repeat customers.

Business Interpretation:
This indicates a strong customer loyalty base and
a significant portion of customers returning.

Recommended Actions:
- Expand loyalty programs because retention is strong.
- Analyze repeat-customer purchase frequency.
"""

    final_report = generate_grounded_report(
        question=test_question,
        analytical_result=test_result,
        business_insights=test_insights,
    )

    print("\n" + "=" * 70)
    print("FINAL GROUNDED REPORT")
    print("=" * 70)

    print(final_report)

    print(
        "\nReporting Agent Version 2 "
        "test completed."
    )