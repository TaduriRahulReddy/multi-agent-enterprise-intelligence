import json
import re

from tools.llm_tool import ask_llm


# ============================================================
# REPORT VALIDATOR - VERSION 3
# CLAIM-LEVEL ANALYTICAL GROUNDING VALIDATION
# ============================================================


UNSUPPORTED_QUALITATIVE_PHRASES = [
    # Loyalty / retention strength
    "strong customer loyalty",
    "strong loyalty",
    "high loyalty",
    "loyal customer base",
    "strong customer base",
    "strong base of repeat customers",
    "strong base",
    "level of customer loyalty",
    "certain level of customer loyalty",
    "customer loyalty and retention",
    "retention within the customer base",
    "strong retention",
    "high retention",
    "positive retention",
    "good retention",
    "healthy retention",
    "good foundation for customer retention",
    "positive foundation for customer retention",

    # Customer-share claims
    "significant portion",
    "large portion",
    "substantial portion",
    "majority of customers",
    "most customers",
    "large share of customers",
    "significant share of customers",

    # Demand / market claims
    "strong demand",
    "high demand",
    "strong market performance",
    "high market share",
    "market leadership",
    "strong market position",

    # Satisfaction claims
    "high customer satisfaction",
    "strong customer satisfaction",

    # Program effectiveness
    "effective loyalty program",
    "effective loyalty programs",
    "potential effectiveness of loyalty programs",
    "loyalty programs are effective",
]


REPORT_SECTION_HEADERS = {
    "EXECUTIVE SUMMARY",
    "KEY RESULTS",
    "BUSINESS IMPLICATION",
    "BUSINESS IMPLICATIONS",
    "RECOMMENDED NEXT STEPS",
    "RECOMMENDATIONS",
}


META_CONTROL_SENTENCES = {
    (
        "the analysis confirms the ranking and values "
        "of the top 5 product categories by total revenue."
    ),
    (
        "the analysis confirms the number of customers "
        "who placed more than one order."
    ),
    (
        "additional context is required to determine "
        "the repeat-customer rate."
    ),
    (
        "unable to produce a fully grounded executive report."
    ),
}


# ============================================================
# TEXT HELPERS
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparisons.
    """

    return " ".join(
        str(text).lower().split()
    )


def result_to_text(
    analytical_result,
) -> str:
    """
    Convert the validated analytical result into readable text.
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


def result_has_only_absolute_count(
    analytical_result,
) -> bool:
    """
    Detect whether the analytical result is a simple
    one-row, one-column count.

    Example:

    repeat_customer_count
    2997
    """

    if not hasattr(
        analytical_result,
        "shape",
    ):
        return False

    rows, columns = analytical_result.shape

    if rows != 1:
        return False

    if columns != 1:
        return False

    column_name = str(
        analytical_result.columns[0]
    ).lower()

    count_indicators = [
        "count",
        "number",
        "total_customers",
        "customer_count",
    ]

    return any(
        indicator in column_name
        for indicator in count_indicators
    )


# ============================================================
# REPORT PARSING
# ============================================================


def parse_report_sections(
    executive_report: str,
) -> dict:
    """
    Parse the executive report into named sections.

    This allows factual claims and recommendations to be
    treated differently.
    """

    sections = {}

    current_section = None

    for raw_line in executive_report.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.upper() in REPORT_SECTION_HEADERS:

            current_section = line.upper()

            sections.setdefault(
                current_section,
                [],
            )

            continue

        if current_section is None:

            current_section = "UNLABELED"

            sections.setdefault(
                current_section,
                [],
            )

        sections[
            current_section
        ].append(
            line
        )

    return sections


def split_into_sentences(
    text: str,
) -> list[str]:
    """
    Split text into compact claim-sized sentences.

    Bullet prefixes are removed.
    """

    cleaned = text.strip()

    cleaned = re.sub(
        r"^[\-\*\u2022]\s*",
        "",
        cleaned,
    )

    if not cleaned:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def is_meta_control_sentence(
    sentence: str,
) -> bool:
    """
    Identify known report-control or analytical-status
    language that does not introduce a new business fact.
    """

    normalized = normalize_text(
        sentence
    )

    if normalized in META_CONTROL_SENTENCES:
        return True

    meta_prefixes = [
        "the analysis confirms",
        "the result provides",
        "additional context is required",
        "additional analysis is required",
        "further analysis is required",
    ]

    return any(
        normalized.startswith(
            prefix
        )
        for prefix in meta_prefixes
    )


def extract_factual_claims(
    executive_report: str,
) -> list[str]:
    """
    Extract factual claims that should be grounded against
    the validated analytical result.

    Recommendations are excluded here and handled separately.
    """

    sections = parse_report_sections(
        executive_report
    )

    claims = []

    recommendation_sections = {
        "RECOMMENDED NEXT STEPS",
        "RECOMMENDATIONS",
    }

    for section_name, lines in sections.items():

        if section_name in recommendation_sections:
            continue

        for line in lines:

            for sentence in split_into_sentences(
                line
            ):

                if is_meta_control_sentence(
                    sentence
                ):
                    continue

                if sentence not in claims:

                    claims.append(
                        sentence
                    )

    return claims


def extract_recommendations(
    executive_report: str,
) -> list[str]:
    """
    Extract recommendation statements separately.

    Recommendations do not need to be facts already contained
    in the analytical result, but they must not rely on an
    unsupported factual premise.
    """

    sections = parse_report_sections(
        executive_report
    )

    recommendations = []

    recommendation_sections = {
        "RECOMMENDED NEXT STEPS",
        "RECOMMENDATIONS",
    }

    for section_name in recommendation_sections:

        for line in sections.get(
            section_name,
            [],
        ):

            cleaned = re.sub(
                r"^[\-\*\u2022]\s*",
                "",
                line,
            ).strip()

            if cleaned:

                recommendations.append(
                    cleaned
                )

    return recommendations


# ============================================================
# DETERMINISTIC GUARDRAILS
# ============================================================


def deterministic_grounding_check(
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Apply deterministic grounding rules before claim-level
    semantic validation.
    """

    normalized_report = normalize_text(
        executive_report
    )

    # --------------------------------------------------------
    # RULE 1:
    # Absolute counts cannot establish qualitative strength.
    # --------------------------------------------------------

    if result_has_only_absolute_count(
        analytical_result
    ):

        for phrase in UNSUPPORTED_QUALITATIVE_PHRASES:

            if phrase in normalized_report:

                return {
                    "is_valid": False,
                    "reason": (
                        "Unsupported qualitative claim detected "
                        "for an absolute-count result: "
                        f"'{phrase}'."
                    ),
                    "source": "deterministic_rule",
                }

    # --------------------------------------------------------
    # RULE 2:
    # Catch additional absolute-count interpretations.
    # --------------------------------------------------------

    if result_has_only_absolute_count(
        analytical_result
    ):

        unsupported_patterns = [
            (
                "loyalty",
                "Unsupported loyalty interpretation detected "
                "for an absolute-count result.",
            ),
            (
                "retention is strong",
                "Unsupported retention-strength claim detected.",
            ),
            (
                "retention is high",
                "Unsupported retention-strength claim detected.",
            ),
            (
                "high retention",
                "Unsupported retention-strength claim detected.",
            ),
            (
                "strong retention",
                "Unsupported retention-strength claim detected.",
            ),
            (
                "strong base",
                "Unsupported strength claim detected for an "
                "absolute-count result.",
            ),
        ]

        for pattern, reason in unsupported_patterns:

            if pattern in normalized_report:

                return {
                    "is_valid": False,
                    "reason": reason,
                    "source": "deterministic_rule",
                }

    return {
        "is_valid": True,
        "reason": (
            "No deterministic grounding violations detected."
        ),
        "source": "deterministic_rule",
    }


# ============================================================
# CLAIM-LEVEL LLM VALIDATION
# ============================================================


def build_claim_validation_prompt(
    question: str,
    analytical_result: str,
    claim: str,
) -> str:
    """
    Ask the LLM to judge one exact factual claim.

    Python owns the claim identity. The LLM does not return
    or rewrite the claim itself.
    """

    return f"""
You are a strict factual grounding validator for an enterprise
analytics system.

Determine whether ONE factual claim is supported by the
validated analytical result.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

FACTUAL CLAIM TO VALIDATE:
{claim}

GROUNDING RULES:

1. The analytical result is authoritative.

2. Validate ONLY the factual claim supplied above.

3. Do not invent additional claims while validating.

4. A faithful paraphrase of the analytical result is supported.

Example:

Analytical result:
repeat_customer_count = 2997

Supported:
"There are 2,997 repeat customers."

Supported:
"The analysis identified 2,997 customers who placed more
than one order."

5. Rankings support ranking statements.

Example:

If the analytical result ranks health_beauty first by revenue,
then:

"Health & beauty has the highest revenue."

is supported.

6. Revenue values do NOT by themselves support claims about:

- demand strength
- customer preference
- profitability
- market share
- market leadership
- marketing effectiveness

7. A count alone does NOT establish:

- retention strength
- loyalty strength
- customer-share percentages
- majority/minority status
- customer satisfaction

8. If the claim contains multiple factual assertions, all material
parts must be supported.

9. When uncertain, return false.

Return ONLY valid JSON using exactly this schema:

{{
  "supported": true,
  "supporting_evidence": "short evidence from the analytical result",
  "reason": "short explanation"
}}

or:

{{
  "supported": false,
  "supporting_evidence": "",
  "reason": "short explanation"
}}
"""


def parse_json_validation_response(
    response: str,
) -> dict:
    """
    Parse one claim-level JSON validation result.
    """

    cleaned = response.strip()

    if cleaned.startswith(
        "```"
    ):

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:

        parsed = json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        return {
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "Unable to parse claim-level grounding "
                "validation response."
            ),
        }

    supported = parsed.get(
        "supported"
    )

    if not isinstance(
        supported,
        bool,
    ):

        return {
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "Claim-level validator did not return a "
                "boolean supported field."
            ),
        }

    return {
        "supported": supported,
        "supporting_evidence": str(
            parsed.get(
                "supporting_evidence",
                "",
            )
        ).strip(),
        "reason": str(
            parsed.get(
                "reason",
                "",
            )
        ).strip(),
    }


def validate_single_claim(
    question: str,
    analytical_result,
    claim: str,
) -> dict:
    """
    Validate exactly one factual claim.

    Python attaches the original claim to the result so the
    LLM cannot alter claim identity.
    """

    result_text = result_to_text(
        analytical_result
    )

    prompt = build_claim_validation_prompt(
        question=question,
        analytical_result=result_text,
        claim=claim,
    )

    response = ask_llm(
        prompt
    )

    llm_result = parse_json_validation_response(
        response
    )

    return {
        "claim": claim,
        "supported": llm_result[
            "supported"
        ],
        "supporting_evidence": llm_result[
            "supporting_evidence"
        ],
        "reason": llm_result[
            "reason"
        ],
    }


# ============================================================
# RECOMMENDATION VALIDATION
# ============================================================


def recommendation_has_unsupported_premise(
    recommendation: str,
    analytical_result,
) -> dict:
    """
    Check recommendations for embedded unsupported premises.

    Recommendations such as:

    "Calculate the repeat-customer rate."

    are allowed because they propose future analysis.

    But:

    "Expand loyalty campaigns because retention is strong."

    is rejected because the recommendation relies on the
    unsupported factual premise that retention is strong.
    """

    normalized = normalize_text(
        recommendation
    )

    if result_has_only_absolute_count(
        analytical_result
    ):

        for phrase in UNSUPPORTED_QUALITATIVE_PHRASES:

            if phrase in normalized:

                return {
                    "is_valid": False,
                    "reason": (
                        "Recommendation contains an unsupported "
                        f"factual premise: '{phrase}'."
                    ),
                    "recommendation": recommendation,
                }

        premise_patterns = [
            "because retention is strong",
            "because retention is high",
            "because loyalty is strong",
            "because customer loyalty is strong",
            "based on strong retention",
            "based on high retention",
            "based on strong customer loyalty",
            "due to strong retention",
            "due to high retention",
        ]

        for pattern in premise_patterns:

            if pattern in normalized:

                return {
                    "is_valid": False,
                    "reason": (
                        "Recommendation relies on an unsupported "
                        f"premise: '{pattern}'."
                    ),
                    "recommendation": recommendation,
                }

    return {
        "is_valid": True,
        "reason": (
            "Recommendation does not contain an obvious "
            "unsupported factual premise."
        ),
        "recommendation": recommendation,
    }


def validate_recommendations(
    analytical_result,
    recommendations: list[str],
) -> dict:
    """
    Validate recommendation wording separately from factual
    analytical claims.
    """

    results = []

    for recommendation in recommendations:

        result = recommendation_has_unsupported_premise(
            recommendation=recommendation,
            analytical_result=analytical_result,
        )

        results.append(
            result
        )

        if not result[
            "is_valid"
        ]:

            return {
                "is_valid": False,
                "reason": result[
                    "reason"
                ],
                "source": "recommendation_guardrail",
                "recommendation_results": results,
            }

    return {
        "is_valid": True,
        "reason": (
            "Recommendations do not rely on obvious "
            "unsupported factual premises."
        ),
        "source": "recommendation_guardrail",
        "recommendation_results": results,
    }


# ============================================================
# FULL REPORT VALIDATION
# ============================================================


def validate_report(
    question: str,
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Report Validator Version 3.

    Workflow:

    Executive Report
        ↓
    Input Validation
        ↓
    Deterministic Guardrails
        ↓
    Extract Factual Claims
        ↓
    Claim-Level LLM Validation
        ↓
    Validate Recommendations Separately
        ↓
    VALID / INVALID

    This prevents a whole-report validator from hallucinating
    claims that do not actually exist in the report.
    """

    # --------------------------------------------------------
    # STEP 0: INPUT VALIDATION
    # --------------------------------------------------------

    if analytical_result is None:

        return {
            "is_valid": False,
            "reason": (
                "No validated analytical result was provided."
            ),
            "source": "input_validation",
            "claim_results": [],
            "recommendation_results": [],
        }

    if not executive_report:

        return {
            "is_valid": False,
            "reason": (
                "No executive report was provided."
            ),
            "source": "input_validation",
            "claim_results": [],
            "recommendation_results": [],
        }

    # --------------------------------------------------------
    # STEP 1: DETERMINISTIC GUARDRAILS
    # --------------------------------------------------------

    deterministic_result = (
        deterministic_grounding_check(
            analytical_result=analytical_result,
            executive_report=executive_report,
        )
    )

    if not deterministic_result[
        "is_valid"
    ]:

        return {
            **deterministic_result,
            "claim_results": [],
            "recommendation_results": [],
        }

    # --------------------------------------------------------
    # STEP 2: EXTRACT FACTUAL CLAIMS
    # --------------------------------------------------------

    factual_claims = extract_factual_claims(
        executive_report
    )

    claim_results = []

    # --------------------------------------------------------
    # STEP 3: VALIDATE EACH FACTUAL CLAIM
    # --------------------------------------------------------

    for claim in factual_claims:

        claim_result = validate_single_claim(
            question=question,
            analytical_result=analytical_result,
            claim=claim,
        )

        claim_results.append(
            claim_result
        )

        if not claim_result[
            "supported"
        ]:

            return {
                "is_valid": False,
                "reason": (
                    "Unsupported factual claim detected: "
                    f"'{claim}'. "
                    f"{claim_result['reason']}"
                ),
                "source": "claim_level_llm_validator",
                "claim_results": claim_results,
                "recommendation_results": [],
            }

    # --------------------------------------------------------
    # STEP 4: VALIDATE RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = extract_recommendations(
        executive_report
    )

    recommendation_validation = (
        validate_recommendations(
            analytical_result=analytical_result,
            recommendations=recommendations,
        )
    )

    if not recommendation_validation[
        "is_valid"
    ]:

        return {
            "is_valid": False,
            "reason": recommendation_validation[
                "reason"
            ],
            "source": recommendation_validation[
                "source"
            ],
            "claim_results": claim_results,
            "recommendation_results": (
                recommendation_validation[
                    "recommendation_results"
                ]
            ),
        }

    # --------------------------------------------------------
    # STEP 5: FINAL SUCCESS
    # --------------------------------------------------------

    return {
        "is_valid": True,
        "reason": (
            f"All {len(claim_results)} factual claims are "
            "supported by the validated analytical result, "
            "and recommendations contain no unsupported "
            "factual premises."
        ),
        "source": "claim_level_llm_validator",
        "claim_results": claim_results,
        "recommendation_results": (
            recommendation_validation[
                "recommendation_results"
            ]
        ),
    }


# ============================================================
# LOCAL TESTS - REPORT VALIDATOR VERSION 3
# ============================================================


if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Report Validator Version 3"
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

    passed_tests = 0
    failed_tests = 0

    # ========================================================
    # TEST CASES
    # ========================================================

    test_cases = [
        (
            "Grounded Report",
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
The analysis confirms that 2,997 customers placed more than one order.

RECOMMENDED NEXT STEPS
- Compare repeat customers with the total unique customer base.
- Analyze purchase frequency among repeat customers.
""",
            True,
        ),
        (
            "Strong Loyalty Claim",
            """
EXECUTIVE SUMMARY
The company has a strong customer loyalty base.

KEY RESULTS
- We have 2,997 repeat customers.

BUSINESS IMPLICATION
A significant portion of customers regularly return.

RECOMMENDED NEXT STEPS
- Expand loyalty campaigns because retention is strong.
""",
            False,
        ),
        (
            "Subtle Loyalty Claim",
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
This count indicates a certain level of customer loyalty
and retention within the customer base.

RECOMMENDED NEXT STEPS
- Consider expanding loyalty campaigns based on the
  strong base of repeat customers.
- Analyze repeat-customer purchase frequency.
""",
            False,
        ),
        (
            "Valid Repeat-Customer Wording",
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
The result provides the number of customers who placed
more than one order. Additional context is required to
determine the repeat-customer rate.

RECOMMENDED NEXT STEPS
- Calculate the repeat-customer rate using the total
  unique customer base.
- Compare revenue from repeat and one-time customers.
""",
            True,
        ),
        (
            "Unsupported Retention Strength",
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
The company has strong retention.

RECOMMENDED NEXT STEPS
- Analyze repeat-customer purchase frequency.
""",
            False,
        ),
        (
            "Valid Further Analysis Recommendation",
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- The analysis identified 2,997 customers who placed more than one order.

BUSINESS IMPLICATION
Additional context is required to determine the repeat-customer rate.

RECOMMENDED NEXT STEPS
- Compare repeat customers with total unique customers.
- Calculate the repeat-customer rate.
""",
            True,
        ),
        (
            "Unsupported Recommendation Premise",
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
The report contains only the repeat-customer count.

RECOMMENDED NEXT STEPS
- Expand loyalty campaigns because retention is strong.
""",
            False,
        ),
    ]

    # ========================================================
    # RUN TESTS
    # ========================================================

    for (
        test_name,
        report,
        expected_result,
    ) in test_cases:

        print(
            "\n" + "-" * 70
        )

        print(
            test_name
        )

        print(
            "-" * 70
        )

        validation = validate_report(
            question=test_question,
            analytical_result=test_result,
            executive_report=report,
        )

        actual_result = validation.get(
            "is_valid",
            False,
        )

        print(
            validation
        )

        print(
            f"Expected: {expected_result}"
        )

        print(
            f"Actual:   {actual_result}"
        )

        if actual_result == expected_result:

            print(
                "RESULT: PASS"
            )

            passed_tests += 1

        else:

            print(
                "RESULT: FAIL"
            )

            failed_tests += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST SUMMARY"
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
        f"Total:  {len(test_cases)}"
    )

    print(
        "\nReport Validator Version 3 "
        "testing completed."
    )