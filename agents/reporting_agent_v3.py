from tools.llm_tool import ask_llm
from tools.report_validator_v2 import validate_report


# ============================================================
# REPORTING AGENT - VERSION 3
# STRICT GROUNDED REPORT GENERATION + HYBRID VALIDATION
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

Your job is to convert validated analytical results into
a concise executive report.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

BUSINESS INSIGHTS:
{business_insights}

CRITICAL GROUNDING RULE:

The VALIDATED ANALYTICAL RESULT is the only authoritative
source of factual evidence.

Business insights may contain unsupported interpretations.
Do NOT automatically repeat them.

STRICT REPORTING RULES:

1. Use only claims directly supported by the validated
   analytical result.

2. Do not invent:
   - numbers
   - percentages
   - trends
   - growth
   - benchmarks
   - comparisons
   - causal explanations
   - customer behavior conclusions

3. If the analytical result contains only an absolute count:

   Example:
   repeat_customer_count = 2997

   You MAY say:

   "There are 2,997 repeat customers."

   "The analysis identified 2,997 customers who placed
   more than one order."

   You MUST NOT say:

   - strong customer loyalty
   - high retention
   - positive retention
   - strong customer base
   - significant portion
   - majority of customers
   - beneficial for revenue
   - beneficial for growth
   - strong customer retention
   - loyalty programs are effective

4. An absolute repeat-customer count does NOT prove:
   - loyalty strength
   - retention strength
   - future revenue
   - business growth
   - customer satisfaction
   - marketing effectiveness

5. If the analytical result contains a ranking:

   You MAY state:
   - the ranking
   - the values
   - which item ranks first, second, etc.

   You MUST NOT infer:
   - strong demand
   - customer preference
   - profitability
   - market leadership
   - market share

6. Preserve numerical results exactly.

7. BUSINESS IMPLICATION must remain factual.

   For a simple count, a valid implication is:

   "The analysis confirms the number of customers who
   placed more than one order. Additional context is
   required to determine the repeat-customer rate."

8. RECOMMENDED NEXT STEPS should focus on further analysis.

   Good examples:

   - Compare repeat customers with total unique customers.
   - Calculate the repeat-customer rate.
   - Analyze purchase frequency.
   - Compare repeat versus one-time customer revenue.
   - Analyze customer purchase intervals.

9. Do NOT recommend expanding a loyalty program unless
   the analytical result directly measures program
   performance.

10. Do NOT say a recommendation is justified by strong
    retention, strong loyalty, demand, growth, or revenue
    unless those metrics are explicitly supplied.

11. Do not mention:
    - SQL
    - DuckDB
    - schemas
    - LLMs
    - prompts
    - technical implementation

Return EXACTLY this structure:

EXECUTIVE SUMMARY
<grounded summary>

KEY RESULTS
- <supported result 1>
- <supported result 2 if relevant>

BUSINESS IMPLICATION
<strictly grounded interpretation>

RECOMMENDED NEXT STEPS
- <analysis-oriented next step 1>
- <analysis-oriented next step 2>
"""


def build_rewrite_prompt(
    question: str,
    analytical_result: str,
    failed_report: str,
    validation_reason: str,
) -> str:
    """
    Rewrite a failed report using strict validator feedback.
    """

    return f"""
You are a STRICT enterprise reporting editor.

The report below failed grounding validation.

You must rewrite it so that every factual statement is
directly supported by the validated analytical result.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

FAILED REPORT:
{failed_report}

VALIDATION FAILURE:
{validation_reason}

IMPORTANT:

Ignore unsupported interpretations in the failed report.

Do not try to preserve unsupported business conclusions.

If the analytical result is:

repeat_customer_count = 2997

then a properly grounded report should communicate only
facts such as:

"There are 2,997 repeat customers."

"The analysis identified 2,997 customers who placed
more than one order."

"Additional context is required to determine the
repeat-customer rate."

STRICT REWRITE RULES:

1. Remove any claims about:
   - loyalty strength
   - retention strength
   - strong customer base
   - business growth
   - future revenue
   - customer satisfaction
   - significant customer share
   - marketing effectiveness
   - program effectiveness

2. Remove phrases such as:
   - strong
   - high retention
   - strong retention
   - positive retention
   - strong loyalty
   - significant portion
   - strong base
   - beneficial for growth
   - beneficial for revenue

3. Do NOT recommend expanding loyalty programs based only
   on the repeat-customer count.

4. Recommendations must focus on additional analysis.

5. For repeat-customer count, suitable recommendations are:

   - Compare repeat customers with total unique customers.
   - Calculate the repeat-customer rate.
   - Analyze repeat-customer purchase frequency.
   - Compare revenue from repeat and one-time customers.

6. Preserve validated numerical values exactly.

7. If the report contains only a count, do not add a
   qualitative interpretation.

8. Do not mention technical implementation.

9. Return only the rewritten report.

Return EXACTLY:

EXECUTIVE SUMMARY
<grounded summary>

KEY RESULTS
- <supported result>

BUSINESS IMPLICATION
<strictly grounded interpretation>

RECOMMENDED NEXT STEPS
- <analysis-oriented next step 1>
- <analysis-oriented next step 2>
"""


def result_to_text(
    analytical_result,
) -> str:
    """
    Convert analytical result into readable text.
    """

    if hasattr(
        analytical_result,
        "to_string",
    ):
        return analytical_result.to_string(
            index=False
        )

    return str(
        analytical_result
    )


def generate_initial_report(
    question: str,
    analytical_result,
    business_insights: str,
) -> str:
    """
    Generate the initial executive report.
    """

    result_text = result_to_text(
        analytical_result
    )

    prompt = build_reporting_prompt(
        question=question,
        analytical_result=result_text,
        business_insights=business_insights,
    )

    response = ask_llm(
        prompt
    )

    return response.strip()


def rewrite_report(
    question: str,
    analytical_result,
    failed_report: str,
    validation_reason: str,
) -> str:
    """
    Rewrite an invalid report.
    """

    result_text = result_to_text(
        analytical_result
    )

    prompt = build_rewrite_prompt(
        question=question,
        analytical_result=result_text,
        failed_report=failed_report,
        validation_reason=validation_reason,
    )

    response = ask_llm(
        prompt
    )

    return response.strip()


def generate_grounded_report(
    question: str,
    analytical_result,
    business_insights: str,
) -> str:
    """
    Generate and validate an executive report.

    Workflow:

    Validated Analytical Result
            ↓
       Initial Report
            ↓
     Hybrid Validator V2
            ↓
         Valid?
        /     \
      Yes      No
       |        |
     Return   Rewrite
                |
             Validate Again
    """

    if analytical_result is None:

        return (
            "A report cannot be generated because no "
            "validated analytical result was provided."
        )

    if not business_insights:

        business_insights = (
            "No additional business insights provided."
        )

    print(
        "\nGenerating initial executive report..."
    )

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

        print(
            "\nGrounding Validation:"
        )

        print(
            validation
        )

        # ----------------------------------------------------
        # VALID
        # ----------------------------------------------------

        if validation["is_valid"]:

            print(
                f"\nExecutive report validated "
                f"successfully on attempt {attempt}."
            )

            return report

        # ----------------------------------------------------
        # MAX ATTEMPTS
        # ----------------------------------------------------

        if attempt == MAX_ATTEMPTS:

            print(
                "\nMaximum report-rewrite attempts reached."
            )

            return (
                "Unable to produce a fully grounded "
                "executive report."
            )

        # ----------------------------------------------------
        # REWRITE
        # ----------------------------------------------------

        print(
            "\nReport is not fully grounded."
        )

        print(
            "Validation Source:"
        )

        print(
            validation.get(
                "source",
                "unknown",
            )
        )

        print(
            "\nValidation Reason:"
        )

        print(
            validation["reason"]
        )

        print(
            "\nSending grounding feedback back "
            "to Qwen for strict rewrite..."
        )

        report = rewrite_report(
            question=question,
            analytical_result=analytical_result,
            failed_report=report,
            validation_reason=validation["reason"],
        )

    return report


# ============================================================
# TEST REPORTING AGENT VERSION 3
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Reporting Agent Version 3"
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

    # Intentionally bad insights.
    # Reporting Agent V3 must NOT blindly repeat them.
    test_insights = """
Key Finding:
There are 2,997 repeat customers.

Business Interpretation:
This shows strong customer loyalty and high retention.

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

    print(
        final_report
    )

    print(
        "\nReporting Agent Version 3 "
        "test completed."
    )