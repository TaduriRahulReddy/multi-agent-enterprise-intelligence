from tools.llm_tool import ask_llm
from tools.report_validator_v10 import validate_report


# ============================================================
# REPORTING AGENT - VERSION 7
#
# Changes from V6:
# - Uses Report Validator V10
# - Preserves the same generation / rewrite loop
# - Preserves the same fallback behavior
# - Retests the revenue-ranking integration failure from V6
# - Retests the repeat-customer count integration failure from V6
# - Verifies Markdown headings are parsed correctly
# - Verifies numbered lists do not create fake factual claims
# - Verifies recommendations remain separate from factual claims
# ============================================================


MAX_ATTEMPTS = 3

FALLBACK_REPORT = (
    "Unable to produce a fully grounded executive report."
)


# ============================================================
# RESULT CONVERSION
# ============================================================


def result_to_text(
    analytical_result,
) -> str:
    """
    Convert analytical output into readable text.
    """

    if analytical_result is None:
        return ""

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


# ============================================================
# INITIAL REPORT PROMPT
# ============================================================


def build_reporting_prompt(
    question: str,
    analytical_result,
    business_insights: str = "",
) -> str:
    """
    Build the first-pass executive-report prompt.
    """

    result_text = result_to_text(
        analytical_result
    )

    if not business_insights:
        business_insights = (
            "No additional business insights were provided."
        )

    return f"""
You are an enterprise analytics reporting agent.

Create a concise executive report using ONLY factual information
supported by the validated analytical result.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{result_text}

OPTIONAL BUSINESS INSIGHTS:
{business_insights}

STRICT GROUNDING RULES:

1. The validated analytical result is the authoritative source
   of factual information.

2. Business insights are suggestions only.
   They are NOT automatically factual.

3. Include a business insight only when it is directly
   supported by the analytical result.

4. Never invent or infer unsupported:
   - percentages
   - rates
   - trends
   - causes
   - demand
   - customer interest
   - customer preference
   - popularity
   - loyalty
   - retention strength
   - satisfaction
   - profitability
   - market share
   - market leadership
   - product complementarity
   - marketing effectiveness

5. Revenue-ranking results support factual statements about:
   - revenue values
   - ordering
   - rank position
   - which categories are listed in the result

6. Revenue-ranking results alone do NOT prove:
   - high demand
   - strong demand
   - customer interest
   - customer preference
   - popularity
   - profitability
   - market share
   - market leadership
   - complementary product relationships
   - marketing effectiveness

7. A count of repeat customers does NOT establish:
   - strong customer loyalty
   - strong retention
   - high retention
   - customer satisfaction
   - customer share
   - majority/minority status

8. Recommendations may propose additional analysis.

Valid examples:
   - Compare revenue across periods.
   - Analyze unit volume.
   - Perform basket analysis to evaluate cross-selling.
   - Calculate repeat-customer rate.
   - Analyze purchase frequency.

9. Recommendations must not rely on unsupported premises.

Bad:
"Increase cross-selling because the categories have
complementary customer interests."

Good:
"Use basket analysis to determine whether cross-selling
opportunities exist between the top categories."

10. When the analytical result is insufficient for a broader
    conclusion, state that additional analysis is required.

Use exactly these section headings:

EXECUTIVE SUMMARY

KEY RESULTS

BUSINESS IMPLICATION

RECOMMENDED NEXT STEPS

Keep the report concise, professional, and grounded.
"""


# ============================================================
# REWRITE PROMPT
# ============================================================


def build_rewrite_prompt(
    question: str,
    analytical_result,
    previous_report: str,
    validation_result: dict,
) -> str:
    """
    Build a correction prompt using validator feedback.
    """

    result_text = result_to_text(
        analytical_result
    )

    validation_reason = validation_result.get(
        "reason",
        "The report contains unsupported claims.",
    )

    claim_results = validation_result.get(
        "claim_results",
        [],
    )

    recommendation_results = validation_result.get(
        "recommendation_results",
        [],
    )

    unsupported_claims = []

    for result in claim_results:

        if not result.get(
            "supported",
            True,
        ):
            claim = result.get(
                "claim",
                "",
            )

            if claim:
                unsupported_claims.append(
                    claim
                )

    unsupported_recommendations = []

    for result in recommendation_results:

        if not result.get(
            "is_valid",
            True,
        ):
            recommendation = result.get(
                "recommendation",
                "",
            )

            if recommendation:
                unsupported_recommendations.append(
                    recommendation
                )

    if unsupported_claims:

        unsupported_claim_text = "\n".join(
            f"- {claim}"
            for claim in unsupported_claims
        )

    else:

        unsupported_claim_text = (
            "None explicitly identified."
        )

    if unsupported_recommendations:

        unsupported_recommendation_text = "\n".join(
            f"- {recommendation}"
            for recommendation in unsupported_recommendations
        )

    else:

        unsupported_recommendation_text = (
            "None explicitly identified."
        )

    return f"""
You are correcting an executive analytics report that failed
strict factual-grounding validation.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{result_text}

PREVIOUS REPORT:
{previous_report}

VALIDATION FAILURE:
{validation_reason}

UNSUPPORTED FACTUAL CLAIMS:
{unsupported_claim_text}

UNSUPPORTED RECOMMENDATIONS:
{unsupported_recommendation_text}

Rewrite the ENTIRE report.

STRICT CORRECTION RULES:

1. Use ONLY the validated analytical result as factual evidence.

2. Remove every unsupported factual claim.

3. Do not replace an unsupported claim with another unsupported
   interpretation.

4. Revenue-ranking data supports:
   - category names
   - revenue values
   - ordering
   - rank position

5. Revenue-ranking data alone does NOT establish:
   - demand
   - customer interest
   - customer preference
   - popularity
   - profitability
   - market share
   - market leadership
   - product complementarity
   - marketing effectiveness

6. Repeat-customer counts do NOT establish:
   - loyalty strength
   - retention strength
   - customer satisfaction
   - percentage of total customers

7. Recommendations may propose further analysis.

Valid examples:
   - Compare category revenue across time periods.
   - Analyze category unit volume.
   - Perform basket analysis to evaluate cross-selling.
   - Calculate repeat-customer rate.

8. Recommendations must not assume unproven facts.

Bad:
"Explore cross-selling because these products have
complementary customer interests."

Good:
"Use basket analysis to determine whether cross-selling
opportunities exist between these categories."

9. A phrase such as "top 5 categories" means the result contains
   five ranked categories. The number 5 is NOT a revenue value.

10. If evidence is insufficient for a business conclusion,
    explicitly say that additional analysis is required.

Use exactly these section headings:

EXECUTIVE SUMMARY

KEY RESULTS

BUSINESS IMPLICATION

RECOMMENDED NEXT STEPS

Return only the rewritten report.
"""


# ============================================================
# INITIAL GENERATION
# ============================================================


def generate_initial_report(
    question: str,
    analytical_result,
    business_insights: str = "",
) -> str:
    """
    Generate the initial report.
    """

    prompt = build_reporting_prompt(
        question=question,
        analytical_result=analytical_result,
        business_insights=business_insights,
    )

    response = ask_llm(
        prompt
    )

    return response.strip()


# ============================================================
# REWRITE
# ============================================================


def rewrite_report(
    question: str,
    analytical_result,
    previous_report: str,
    validation_result: dict,
) -> str:
    """
    Rewrite a report after validation failure.
    """

    prompt = build_rewrite_prompt(
        question=question,
        analytical_result=analytical_result,
        previous_report=previous_report,
        validation_result=validation_result,
    )

    response = ask_llm(
        prompt
    )

    return response.strip()


# ============================================================
# GROUNDED REPORTING LOOP
# ============================================================


def generate_grounded_report(
    question: str,
    analytical_result,
    business_insights: str = "",
) -> dict:
    """
    Generate, validate, and when necessary rewrite an executive
    report.

    Flow:

        Analytical Result
              |
        Business Insights
              |
        Reporting Agent V7
              |
         Draft Report
              |
        Validator V10
              |
           Valid?
          /     \
        Yes      No
         |        |
      Return    Rewrite
                  |
              Revalidate
                  |
            MAX 3 attempts
                  |
               Fallback
    """

    if analytical_result is None:

        return {
            "executive_report": FALLBACK_REPORT,
            "validation": {
                "is_valid": False,
                "reason": (
                    "No analytical result was provided."
                ),
                "source": "reporting_agent_v7",
            },
            "attempts": 0,
        }

    report = generate_initial_report(
        question=question,
        analytical_result=analytical_result,
        business_insights=business_insights,
    )

    last_validation = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        print(
            f"\nReporting Agent V7 - "
            f"Validation Attempt {attempt}"
        )

        print(
            "-" * 70
        )

        validation = validate_report(
            question=question,
            analytical_result=analytical_result,
            executive_report=report,
        )

        last_validation = validation

        print(
            validation
        )

        if validation.get(
            "is_valid",
            False,
        ):

            return {
                "executive_report": report,
                "validation": validation,
                "attempts": attempt,
            }

        if attempt < MAX_ATTEMPTS:

            report = rewrite_report(
                question=question,
                analytical_result=analytical_result,
                previous_report=report,
                validation_result=validation,
            )

    return {
        "executive_report": FALLBACK_REPORT,
        "validation": last_validation,
        "attempts": MAX_ATTEMPTS,
    }


# ============================================================
# LOCAL TESTS - REPORTING AGENT VERSION 7
# ============================================================


if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Reporting Agent Version 7"
    )

    print("=" * 70)

    passed_tests = 0
    failed_tests = 0

    # ========================================================
    # TEST 1
    # REVENUE-RANKING REGRESSION
    # ========================================================

    print(
        "\nTEST 1 - Revenue Ranking Regression"
    )

    print(
        "=" * 70
    )

    revenue_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    revenue_result = pd.DataFrame(
        {
            "category_english": [
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

    unsafe_revenue_insights = """
The top-performing categories suggest that personal care and
luxury/gift items are in high demand among customers.

This indicates potential areas for increased marketing efforts
and cross-selling opportunities to leverage customer interest
in these product types.

Explore cross-selling opportunities between health & beauty
and watches & gifts to capitalize on complementary product
interests.
"""

    revenue_output = generate_grounded_report(
        question=revenue_question,
        analytical_result=revenue_result,
        business_insights=unsafe_revenue_insights,
    )

    print(
        "\nFINAL REVENUE REPORT"
    )

    print(
        "-" * 70
    )

    print(
        revenue_output[
            "executive_report"
        ]
    )

    print(
        "\nFINAL REVENUE VALIDATION"
    )

    print(
        "-" * 70
    )

    print(
        revenue_output[
            "validation"
        ]
    )

    revenue_report_lower = revenue_output[
        "executive_report"
    ].lower()

    revenue_forbidden_phrases = [
        "high demand",
        "strong demand",
        "customer interest",
        "customer preference",
        "market leadership",
        "market leader",
        "complementary product interests",
        "marketing effectiveness",
    ]

    revenue_forbidden_found = [
        phrase
        for phrase in revenue_forbidden_phrases
        if phrase in revenue_report_lower
    ]

    revenue_passed = (
        revenue_output[
            "validation"
        ] is not None
        and revenue_output[
            "validation"
        ].get(
            "is_valid",
            False,
        )
        and revenue_output[
            "executive_report"
        ] != FALLBACK_REPORT
        and not revenue_forbidden_found
    )

    print(
        "\nRevenue validation passed:",
        revenue_output[
            "validation"
        ].get(
            "is_valid",
            False,
        ),
    )

    print(
        "Revenue fallback used:",
        revenue_output[
            "executive_report"
        ] == FALLBACK_REPORT,
    )

    print(
        "Revenue forbidden phrases found:",
        revenue_forbidden_found,
    )

    print(
        "Revenue attempts:",
        revenue_output[
            "attempts"
        ],
    )

    print(
        "Revenue RESULT:",
        "PASS" if revenue_passed else "FAIL",
    )

    if revenue_passed:
        passed_tests += 1
    else:
        failed_tests += 1

    # ========================================================
    # TEST 2
    # REPEAT-CUSTOMER COUNT REGRESSION
    # ========================================================

    print(
        "\n\nTEST 2 - Repeat Customer Count Regression"
    )

    print(
        "=" * 70
    )

    count_question = (
        "How many repeat customers do we have?"
    )

    count_result = pd.DataFrame(
        {
            "repeat_customer_count": [
                2997
            ]
        }
    )

    unsafe_count_insights = """
The presence of 2,997 repeat customers demonstrates strong
customer loyalty and high retention.

The company should expand loyalty programs because retention
is strong.
"""

    count_output = generate_grounded_report(
        question=count_question,
        analytical_result=count_result,
        business_insights=unsafe_count_insights,
    )

    print(
        "\nFINAL COUNT REPORT"
    )

    print(
        "-" * 70
    )

    print(
        count_output[
            "executive_report"
        ]
    )

    print(
        "\nFINAL COUNT VALIDATION"
    )

    print(
        "-" * 70
    )

    print(
        count_output[
            "validation"
        ]
    )

    count_report_lower = count_output[
        "executive_report"
    ].lower()

    count_forbidden_phrases = [
        "strong customer loyalty",
        "strong loyalty",
        "high retention",
        "strong retention",
    ]

    count_forbidden_found = [
        phrase
        for phrase in count_forbidden_phrases
        if phrase in count_report_lower
    ]

    count_passed = (
        count_output[
            "validation"
        ] is not None
        and count_output[
            "validation"
        ].get(
            "is_valid",
            False,
        )
        and count_output[
            "executive_report"
        ] != FALLBACK_REPORT
        and not count_forbidden_found
    )

    print(
        "\nCount validation passed:",
        count_output[
            "validation"
        ].get(
            "is_valid",
            False,
        ),
    )

    print(
        "Count fallback used:",
        count_output[
            "executive_report"
        ] == FALLBACK_REPORT,
    )

    print(
        "Count forbidden phrases found:",
        count_forbidden_found,
    )

    print(
        "Count attempts:",
        count_output[
            "attempts"
        ],
    )

    print(
        "Count RESULT:",
        "PASS" if count_passed else "FAIL",
    )

    if count_passed:
        passed_tests += 1
    else:
        failed_tests += 1

    # ========================================================
    # FINAL TEST SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "REPORTING AGENT VERSION 7 TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Passed: {passed_tests}"
    )

    print(
        f"Failed: {failed_tests}"
    )

    print(
        "Total:  2"
    )

    print(
        "\nReporting Agent Version 7 "
        "testing completed."
    )