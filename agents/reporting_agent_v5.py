from tools.llm_tool import ask_llm
from tools.report_validator_v8 import validate_report


# ============================================================
# REPORTING AGENT - VERSION 5
#
# Responsibilities:
# 1. Convert validated analytical output into an executive report
# 2. Validate every generated report against the source result
# 3. Rewrite the report when unsupported claims are detected
# 4. Prevent unsupported business interpretations
# 5. Use Report Validator V8 for:
#    - deterministic count checks
#    - deterministic revenue/ranking checks
#    - deterministic numeric validation
#    - recommendation premise validation
#    - claim-level semantic validation
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
    Convert an analytical result into readable text
    for the reporting LLM.
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
    Build the first-pass executive reporting prompt.
    """

    result_text = result_to_text(
        analytical_result
    )

    business_insights = (
        business_insights
        if business_insights
        else "No additional business insights were provided."
    )

    return f"""
You are an enterprise analytics reporting agent.

Your task is to create a concise executive report using ONLY
the validated analytical result as factual evidence.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{result_text}

OPTIONAL BUSINESS INSIGHTS:
{business_insights}

IMPORTANT GROUNDING RULES:

1. The validated analytical result is the authoritative source
   of factual information.

2. Business insights are suggestions only.
   They are NOT automatically factual.

3. Do not include a business insight unless it is directly
   supported by the analytical result.

4. Do not invent:
   - percentages
   - rates
   - trends
   - causes
   - customer behavior
   - customer preferences
   - customer demand
   - customer interest
   - customer loyalty
   - customer retention strength
   - customer satisfaction
   - profitability
   - market share
   - market leadership
   - product complementarity
   - marketing effectiveness

5. Revenue rankings support only facts such as:
   - category revenue values
   - ordering
   - rank position

6. Revenue rankings alone do NOT prove:
   - high demand
   - strong demand
   - customer preference
   - customer interest
   - popularity
   - profitability
   - market share
   - market leadership
   - complementary product relationships
   - marketing effectiveness

7. An absolute customer count does NOT prove:
   - strong loyalty
   - strong retention
   - high retention
   - percentage of customers
   - majority of customers
   - customer satisfaction

8. Recommendations may propose further analysis.

Examples of valid recommendations:
   - Analyze purchase frequency.
   - Calculate repeat-customer rate.
   - Compare revenue across periods.
   - Perform basket analysis to evaluate cross-selling.
   - Analyze unit volume.

9. Recommendations must NOT assume facts that have not already
   been established by the analytical result.

10. If the evidence is insufficient for a business conclusion,
    explicitly say that additional analysis is required.

Use exactly these section headings:

EXECUTIVE SUMMARY

KEY RESULTS

BUSINESS IMPLICATION

RECOMMENDED NEXT STEPS

Keep the report concise, factual, professional, and grounded.
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
    Build a correction prompt after validation failure.
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

    failed_claims = []

    for claim_result in claim_results:

        if not claim_result.get(
            "supported",
            True,
        ):

            failed_claims.append(
                claim_result.get(
                    "claim",
                    "",
                )
            )

    failed_recommendations = []

    for recommendation_result in recommendation_results:

        if not recommendation_result.get(
            "is_valid",
            True,
        ):

            failed_recommendations.append(
                recommendation_result.get(
                    "recommendation",
                    "",
                )
            )

    failed_claim_text = (
        "\n".join(
            f"- {claim}"
            for claim in failed_claims
            if claim
        )
        if failed_claims
        else "None explicitly identified."
    )

    failed_recommendation_text = (
        "\n".join(
            f"- {recommendation}"
            for recommendation in failed_recommendations
            if recommendation
        )
        if failed_recommendations
        else "None explicitly identified."
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
{failed_claim_text}

UNSUPPORTED RECOMMENDATIONS:
{failed_recommendation_text}

Rewrite the entire report.

STRICT RULES:

1. Use ONLY the validated analytical result as factual evidence.

2. Remove every unsupported factual claim.

3. Do not replace unsupported claims with different unsupported
   interpretations.

4. Revenue rankings support only:
   - exact revenue values
   - ordering
   - ranking

5. Revenue rankings do NOT establish:
   - demand
   - customer interest
   - customer preference
   - popularity
   - profitability
   - market share
   - market leadership
   - complementary product relationships
   - marketing effectiveness

6. A count of repeat customers does NOT establish:
   - strong loyalty
   - strong retention
   - high retention
   - customer satisfaction
   - percentage or share of customers

7. Recommendations may propose additional analysis.

For example:
   - Analyze basket-level purchases to evaluate cross-selling.
   - Compare category revenue across periods.
   - Calculate repeat-customer rate.
   - Analyze purchase frequency.

8. Recommendations must not rely on unsupported premises.

Bad:
"Expand cross-selling because these categories have
complementary customer interests."

Good:
"Use basket analysis to evaluate whether cross-selling
opportunities exist between the top categories."

9. If the analytical result cannot answer a broader business
   question, say additional analysis is required.

Use exactly these section headings:

EXECUTIVE SUMMARY

KEY RESULTS

BUSINESS IMPLICATION

RECOMMENDED NEXT STEPS

Return only the rewritten report.
"""


# ============================================================
# REPORT GENERATION
# ============================================================


def generate_initial_report(
    question: str,
    analytical_result,
    business_insights: str = "",
) -> str:
    """
    Generate the first report draft.
    """

    prompt = build_reporting_prompt(
        question=question,
        analytical_result=analytical_result,
        business_insights=business_insights,
    )

    return ask_llm(
        prompt
    ).strip()


def rewrite_report(
    question: str,
    analytical_result,
    previous_report: str,
    validation_result: dict,
) -> str:
    """
    Rewrite an invalid report using validator feedback.
    """

    prompt = build_rewrite_prompt(
        question=question,
        analytical_result=analytical_result,
        previous_report=previous_report,
        validation_result=validation_result,
    )

    return ask_llm(
        prompt
    ).strip()


# ============================================================
# GROUNDED REPORTING LOOP
# ============================================================


def generate_grounded_report(
    question: str,
    analytical_result,
    business_insights: str = "",
) -> dict:
    """
    Generate and validate an executive report.

    Version 5 flow:

        Analytical Result
              |
        Business Insights
              |
        Reporting Agent
              |
        Draft Report
              |
        Report Validator V8
              |
        Valid?
          /       \
        Yes        No
         |          |
       Return    Rewrite
                    |
               Revalidate
                    |
            Up to MAX_ATTEMPTS
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
                "source": "reporting_agent_v5",
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
            f"\nReporting Agent V5 - "
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
# LOCAL TESTS - REPORTING AGENT VERSION 5
# ============================================================


if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Reporting Agent Version 5"
    )

    print("=" * 70)

    # ========================================================
    # TEST 1
    #
    # Revenue-ranking regression test.
    #
    # This reproduces the semantic failure exposed during
    # Orchestrator V12 integration:
    #
    # Revenue rankings were incorrectly interpreted as:
    # - high demand
    # - customer interest
    # - complementary product interests
    #
    # Reporting Agent V5 + Validator V8 must reject those
    # conclusions and produce a grounded rewrite.
    # ========================================================

    question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    analytical_result = pd.DataFrame(
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

    intentionally_unsafe_business_insights = """
The top-performing categories suggest that personal care and
luxury/gift items are in high demand among customers.

This indicates potential areas for increased marketing efforts
and cross-selling opportunities to leverage customer interest.

Explore cross-selling opportunities between health & beauty
and watches & gifts to capitalize on complementary product
interests.
"""

    result = generate_grounded_report(
        question=question,
        analytical_result=analytical_result,
        business_insights=(
            intentionally_unsafe_business_insights
        ),
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL EXECUTIVE REPORT"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "executive_report"
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "validation"
        ]
    )

    print(
        "\nAttempts:",
        result[
            "attempts"
        ],
    )

    # ========================================================
    # BASIC PASS / FAIL CHECK
    # ========================================================

    final_report = result[
        "executive_report"
    ].lower()

    final_validation = result[
        "validation"
    ]

    forbidden_phrases = [
        "high demand",
        "strong demand",
        "customer interest",
        "customer preference",
        "market leadership",
        "market leader",
        "complementary product interests",
        "marketing effectiveness",
    ]

    forbidden_found = [
        phrase
        for phrase in forbidden_phrases
        if phrase in final_report
    ]

    passed = (
        final_validation is not None
        and final_validation.get(
            "is_valid",
            False,
        )
        and result[
            "executive_report"
        ] != FALLBACK_REPORT
        and not forbidden_found
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REPORTING AGENT VERSION 5 TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Validation passed:",
        final_validation.get(
            "is_valid",
            False,
        )
        if final_validation
        else False,
    )

    print(
        "Fallback used:",
        result[
            "executive_report"
        ] == FALLBACK_REPORT,
    )

    print(
        "Forbidden phrases found:",
        forbidden_found,
    )

    print(
        "Attempts:",
        result[
            "attempts"
        ],
    )

    print(
        "RESULT:",
        "PASS" if passed else "FAIL",
    )