from tools.llm_tool import ask_llm


# ============================================================
# REPORT VALIDATOR - VERSION 2
# HYBRID RULE-BASED + LLM GROUNDING VALIDATION
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


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic phrase checks.
    """

    return " ".join(
        text.lower().split()
    )


def result_to_text(
    analytical_result,
) -> str:
    """
    Convert an analytical result into readable text.
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


def deterministic_grounding_check(
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Apply deterministic grounding rules before
    semantic LLM validation.

    This catches obvious unsupported claims.
    """

    normalized_report = normalize_text(
        executive_report
    )

    # --------------------------------------------------------
    # RULE 1:
    # If the result is only an absolute count, reject
    # unsupported strength / share / loyalty claims.
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
                        f"for an absolute-count result: '{phrase}'."
                    ),
                    "source": "deterministic_rule",
                }

    # --------------------------------------------------------
    # RULE 2:
    # Reject several broader patterns that may not match
    # an exact phrase in the list.
    # --------------------------------------------------------

    if result_has_only_absolute_count(
        analytical_result
    ):

        unsupported_patterns = [
            (
                "loyalty",
                "Unsupported loyalty interpretation detected "
                "for an absolute-count result."
            ),
            (
                "retention is strong",
                "Unsupported retention-strength claim detected."
            ),
            (
                "retention is high",
                "Unsupported retention-strength claim detected."
            ),
            (
                "high retention",
                "Unsupported retention-strength claim detected."
            ),
            (
                "strong retention",
                "Unsupported retention-strength claim detected."
            ),
            (
                "strong base",
                "Unsupported strength claim detected for an "
                "absolute-count result."
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


def build_llm_validation_prompt(
    question: str,
    analytical_result: str,
    executive_report: str,
) -> str:
    """
    Build the semantic grounding-validation prompt.
    """

    return f"""
You are a STRICT Grounding Validator for an enterprise
intelligence system.

Your job is to verify whether the executive report is
fully supported by the validated analytical result.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

EXECUTIVE REPORT:
{executive_report}

STRICT INTERPRETATION RULES:

1. The analytical result is authoritative.

2. Column names are meaningful evidence.

Example:

repeat_customer_count = 2997

VALID statements:

"There are 2,997 repeat customers."

"Repeat customer count is 2,997."

"The analysis identified 2,997 customers who placed
more than one order."

3. The phrase "repeat customers" is allowed when the
analytical result explicitly contains a field such as:

repeat_customer_count

4. An absolute count alone DOES NOT support claims about:

- customer loyalty strength
- retention strength
- strong customer base
- high customer loyalty
- high retention
- positive retention
- significant portion of customers
- large portion of customers
- majority of customers
- strong demand
- high demand
- market leadership
- program effectiveness

5. A statement such as:

"This count indicates a certain level of customer loyalty."

is INVALID when the analytical result only contains an
absolute count.

6. A statement such as:

"This result suggests a positive foundation for retention."

is INVALID unless the analytical result includes a rate,
benchmark, comparison, historical trend, or target.

7. A statement such as:

"Expand loyalty programs because retention is strong."

is INVALID unless retention strength is actually measured.

8. Rankings support ranking statements.

For example:

"Health Beauty ranks first by revenue."

Rankings alone do NOT prove:

- customer preference
- strong demand
- profitability
- market leadership
- marketing effectiveness

9. Revenue values alone do NOT prove demand strength,
customer preference, profitability, or market share.

10. Recommendations may suggest further analysis.

Grounded examples:

- Compare repeat customers with total unique customers.
- Calculate the repeat-customer rate.
- Compare repeat and one-time customer revenue.
- Analyze purchase frequency among repeat customers.

11. Recommendations are INVALID if they rely on an
unsupported premise.

12. Evaluate the entire report:
- executive summary
- key results
- business implication
- recommended next steps

13. If ANY material claim is unsupported,
mark the report INVALID.

14. When uncertain, prefer INVALID.

Return EXACTLY:

is_valid: <true_or_false>
reason: <short explanation>

Do not include markdown.
Do not include other text.
"""


def parse_validation_response(
    response: str,
) -> dict:
    """
    Parse structured validation output from the LLM.
    """

    parsed = {
        "is_valid": False,
        "reason": (
            "Unable to parse grounding validation response."
        ),
        "source": "llm_validator",
    }

    for line in response.strip().splitlines():

        line = line.strip()

        if line.startswith(
            "is_valid:"
        ):

            value = (
                line.split(
                    ":",
                    1,
                )[1]
                .strip()
                .lower()
            )

            parsed["is_valid"] = (
                value == "true"
            )

        elif line.startswith(
            "reason:"
        ):

            parsed["reason"] = (
                line.split(
                    ":",
                    1,
                )[1]
                .strip()
            )

    return parsed


def llm_grounding_check(
    question: str,
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Run semantic grounding validation using Qwen.
    """

    result_text = result_to_text(
        analytical_result
    )

    prompt = build_llm_validation_prompt(
        question=question,
        analytical_result=result_text,
        executive_report=executive_report,
    )

    response = ask_llm(
        prompt
    )

    return parse_validation_response(
        response
    )


def validate_report(
    question: str,
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Hybrid report-validation workflow.

    Executive Report
        ↓
    Deterministic Rules
        ↓
    Pass?
       ├── No → INVALID
       └── Yes
              ↓
         LLM Grounding Check
              ↓
         VALID / INVALID
    """

    # --------------------------------------------------------
    # Basic input validation
    # --------------------------------------------------------

    if analytical_result is None:

        return {
            "is_valid": False,
            "reason": (
                "No validated analytical result "
                "was provided."
            ),
            "source": "input_validation",
        }

    if not executive_report:

        return {
            "is_valid": False,
            "reason": (
                "No executive report was provided."
            ),
            "source": "input_validation",
        }

    # --------------------------------------------------------
    # STEP 1: deterministic validation
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

        return deterministic_result

    # --------------------------------------------------------
    # STEP 2: semantic LLM validation
    # --------------------------------------------------------

    semantic_result = (
        llm_grounding_check(
            question=question,
            analytical_result=analytical_result,
            executive_report=executive_report,
        )
    )

    return semantic_result


# ============================================================
# TEST REPORT VALIDATOR VERSION 2
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Report Validator Version 2"
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

    # ========================================================
    # TEST 1: GROUNDED REPORT
    # ========================================================

    grounded_report = """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
The analysis confirms that 2,997 customers placed
more than one order.

RECOMMENDED NEXT STEPS
- Compare repeat customers with the total unique
  customer base.
- Analyze purchase frequency among repeat customers.
"""

    grounded_validation = validate_report(
        question=test_question,
        analytical_result=test_result,
        executive_report=grounded_report,
    )

    print(
        "\nGrounded Report Validation:"
    )

    print(
        grounded_validation
    )

    # ========================================================
    # TEST 2: STRONG LOYALTY CLAIM
    # ========================================================

    unsupported_report = """
EXECUTIVE SUMMARY
The company has a strong customer loyalty base.

KEY RESULTS
- We have 2,997 repeat customers.

BUSINESS IMPLICATION
A significant portion of customers regularly return.

RECOMMENDED NEXT STEPS
- Expand loyalty campaigns because retention is strong.
"""

    unsupported_validation = validate_report(
        question=test_question,
        analytical_result=test_result,
        executive_report=unsupported_report,
    )

    print(
        "\nUnsupported Report Validation:"
    )

    print(
        unsupported_validation
    )

    # ========================================================
    # TEST 3: SUBTLE LOYALTY CLAIM
    # ========================================================

    subtle_unsupported_report = """
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
"""

    subtle_validation = validate_report(
        question=test_question,
        analytical_result=test_result,
        executive_report=subtle_unsupported_report,
    )

    print(
        "\nSubtle Unsupported Report Validation:"
    )

    print(
        subtle_validation
    )

    # ========================================================
    # TEST 4: VALID REPEAT-CUSTOMER WORDING
    # ========================================================

    repeat_customer_report = """
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
"""

    repeat_customer_validation = validate_report(
        question=test_question,
        analytical_result=test_result,
        executive_report=repeat_customer_report,
    )

    print(
        "\nRepeat-Customer Wording Validation:"
    )

    print(
        repeat_customer_validation
    )

    # ========================================================
    # EXPECTED BEHAVIOR
    # ========================================================

    print(
        "\nExpected behavior:"
    )

    print(
        "Grounded report -> True"
    )

    print(
        "Strong loyalty report -> False"
    )

    print(
        "Subtle loyalty report -> False"
    )

    print(
        "Valid repeat-customer wording -> True"
    )

    print(
        "\nReport Validator Version 2 "
        "test completed."
    )