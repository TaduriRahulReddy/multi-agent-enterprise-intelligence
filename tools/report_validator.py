from tools.llm_tool import ask_llm


# ============================================================
# REPORT VALIDATOR - VERSION 1
# STRICT GROUNDED EXECUTIVE RESPONSE VALIDATION
# ============================================================


def build_report_validation_prompt(
    question: str,
    analytical_result: str,
    executive_report: str,
) -> str:
    """
    Build a strict grounding-validation prompt.
    """

    return f"""
You are a STRICT Grounding Validator for an enterprise
intelligence system.

Your task is to determine whether every claim in the
executive report is directly supported by the validated
analytical result.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

EXECUTIVE REPORT:
{executive_report}

STRICT VALIDATION RULES:

1. Every factual claim must be directly supported by the
   analytical result.

2. If the analytical result contains only an absolute count,
   the report may state that count.

3. If only an absolute count is available, the report MUST
   be marked INVALID if it uses qualitative language such as:

   - strong
   - weak
   - high
   - low
   - significant
   - substantial
   - large
   - small
   - healthy
   - poor
   - loyal
   - high loyalty
   - strong loyalty
   - strong retention
   - high retention
   - strong demand
   - majority
   - large portion
   - significant portion

   unless the analytical result also contains a denominator,
   benchmark, historical comparison, target, percentage,
   or other evidence that proves that claim.

4. Example:

   Analytical result:
   repeat_customer_count = 2997

   VALID statement:
   "There are 2,997 repeat customers."

   INVALID statement:
   "The company has a strong customer loyalty base."

   INVALID statement:
   "A significant portion of customers are returning."

   INVALID statement:
   "Retention is strong."

5. Rankings may support statements such as:

   "Health Beauty ranks first by revenue."

   Rankings alone DO NOT support:

   "Health Beauty has strong customer demand."

6. Revenue values do NOT automatically prove:

   - profitability
   - demand strength
   - market share
   - customer preference
   - marketing effectiveness

7. Recommendations may suggest further analysis.

8. Recommendations must be marked INVALID if they rely on
   an unsupported premise.

9. Do not allow invented numbers.

10. Do not allow unsupported percentages.

11. Do not allow unsupported trends.

12. Do not allow unsupported growth claims.

13. Do not allow unsupported causal claims.

14. Do not allow unsupported customer loyalty claims.

15. Do not allow unsupported retention claims.

16. Do not allow unsupported demand claims.

17. Do not allow unsupported market-performance claims.

18. Evaluate the ENTIRE report, including:
   - executive summary
   - key results
   - business implication
   - recommended next steps

19. If ANY important statement is unsupported,
   mark the entire report INVALID.

20. When uncertain, prefer INVALID rather than assuming
   that the claim is supported.

Return EXACTLY this format:

is_valid: <true_or_false>
reason: <short explanation>

Do not include markdown.
Do not include any other text.
"""


def parse_validation_response(response: str) -> dict:
    """
    Convert the LLM validator response into
    a Python dictionary.
    """

    result = {
        "is_valid": False,
        "reason": "Unable to validate report.",
    }

    for line in response.strip().splitlines():

        line = line.strip()

        if line.startswith("is_valid:"):

            value = (
                line.split(":", 1)[1]
                .strip()
                .lower()
            )

            result["is_valid"] = (
                value == "true"
            )

        elif line.startswith("reason:"):

            result["reason"] = (
                line.split(":", 1)[1]
                .strip()
            )

    return result


def validate_report(
    question: str,
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Validate whether an executive report is grounded
    in the validated analytical result.
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
        }

    if not executive_report:

        return {
            "is_valid": False,
            "reason": (
                "No executive report was provided."
            ),
        }

    # --------------------------------------------------------
    # Convert analytical result to text
    # --------------------------------------------------------

    if hasattr(
        analytical_result,
        "to_string",
    ):

        result_text = analytical_result.to_string(
            index=False
        )

    else:

        result_text = str(
            analytical_result
        )

    # --------------------------------------------------------
    # Build strict validation prompt
    # --------------------------------------------------------

    prompt = build_report_validation_prompt(
        question=question,
        analytical_result=result_text,
        executive_report=executive_report,
    )

    # --------------------------------------------------------
    # Send validation request to local LLM
    # --------------------------------------------------------

    response = ask_llm(
        prompt
    )

    # --------------------------------------------------------
    # Parse structured response
    # --------------------------------------------------------

    validation_result = (
        parse_validation_response(
            response
        )
    )

    return validation_result


# ============================================================
# TEST REPORT VALIDATOR VERSION 1
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Report Validator Version 1"
    )

    print("=" * 70)

    # ========================================================
    # TEST 1: GROUNDED REPORT
    # ========================================================

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

    grounded_report = """
EXECUTIVE SUMMARY
The business has 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
The result confirms that 2,997 customers placed
more than one order.

RECOMMENDED NEXT STEPS
- Compare the repeat-customer count with the total
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
    # TEST 2: UNSUPPORTED REPORT
    # ========================================================

    unsupported_report = """
EXECUTIVE SUMMARY
The company has a strong and highly loyal customer base.

KEY RESULTS
- We have 2,997 repeat customers.

BUSINESS IMPLICATION
A significant portion of customers are loyal and
regularly return for purchases.

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
    # EXPECTED RESULTS
    # ========================================================

    print(
        "\nExpected behavior:"
    )

    print(
        "Grounded report -> is_valid: True"
    )

    print(
        "Unsupported report -> is_valid: False"
    )

    print(
        "\nReport Validator Version 1 "
        "test completed."
    )