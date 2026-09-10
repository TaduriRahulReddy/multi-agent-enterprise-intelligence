import requests
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# ============================================================
# RESEARCH GROUNDING VALIDATOR - VERSION 4
# BALANCED CLAIM-LEVEL VALIDATION + STRONGER SENTIMENT CHECKS
# ============================================================


def deterministic_research_checks(
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Run deterministic checks before calling the LLM validator.

    Version 4 checks for:
    - unsupported operational/business inferences
    - unsupported success conditions
    - unsupported frequency/strength qualifiers
    - unsupported causal language
    - unsupported customer satisfaction/dissatisfaction claims
    """

    answer_lower = research_answer.lower()
    context_lower = retrieved_context.lower()

    # --------------------------------------------------------
    # CHECK 1: Unsupported inference phrases
    # --------------------------------------------------------

    risky_phrases = [
        "supply chain",
        "logistical issues",
        "logistics issues",
        "customer loyalty",
        "strong demand",
        "market demand",
        "customer preference",
        "customer satisfaction",
        "satisfaction levels",
        "customer dissatisfaction",
        "dissatisfaction levels",
        "negative impact on satisfaction",
        "impacted satisfaction",
        "affected satisfaction",
        "retention strength",
        "competitive position",
        "successful strategy",
        "operational inefficiency",
        "operational issues",
        "business success",
        "business growth",
    ]

    for phrase in risky_phrases:
        if phrase in answer_lower and phrase not in context_lower:
            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported interpretation detected: "
                    f"'{phrase}' appears in the research answer "
                    "but is not supported by the retrieved evidence."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # CHECK 2: Unsupported success-condition language
    # --------------------------------------------------------

    success_phrases = [
        "if successful",
        "if it is successful",
        "if this is successful",
        "if the change is successful",
        "if the strategy is successful",
        "when successful",
        "once successful",
        "if the strategy succeeds",
        "if the change succeeds",
        "depending on success",
    ]

    for phrase in success_phrases:
        if phrase in answer_lower and phrase not in context_lower:
            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported conditional interpretation detected: "
                    f"'{phrase}' is not stated in the retrieved evidence."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # CHECK 3: Unsupported qualifier language
    # --------------------------------------------------------

    qualifier_phrases = [
        "frequently",
        "very frequently",
        "significantly",
        "substantially",
        "strongly",
        "widely",
        "extensively",
        "major",
        "severe",
        "highly",
        "large-scale",
    ]

    for phrase in qualifier_phrases:
        if phrase in answer_lower and phrase not in context_lower:
            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported qualifier detected: "
                    f"'{phrase}' strengthens the evidence beyond "
                    "what is explicitly stated."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # CHECK 4: Unsupported causal wording
    # --------------------------------------------------------

    causal_patterns = [
        r"\bcaused by\b",
        r"\bcausing\b",
        r"\bdue to\b",
        r"\bresulted from\b",
        r"\bled to\b",
        r"\bbecause of\b",
        r"\bresulting in\b",
    ]

    for pattern in causal_patterns:
        match = re.search(pattern, answer_lower)

        if match:
            phrase = match.group(0)

            if phrase not in context_lower:
                return {
                    "is_valid": False,
                    "reason": (
                        f"Unsupported causal wording detected: "
                        f"'{phrase}' appears in the research answer "
                        "but is not explicitly supported by the evidence."
                    ),
                    "source": "deterministic_rule",
                }

    # --------------------------------------------------------
    # CHECK 5: Satisfaction-related semantic patterns
    # --------------------------------------------------------

    satisfaction_patterns = [
        r"\bnegatively impacted .*satisfaction\b",
        r"\bimpacted .*satisfaction\b",
        r"\baffected .*satisfaction\b",
        r"\breduced .*satisfaction\b",
        r"\blowered .*satisfaction\b",
        r"\bcaused .*dissatisfaction\b",
        r"\bresulted in .*dissatisfaction\b",
    ]

    for pattern in satisfaction_patterns:
        match = re.search(pattern, answer_lower)

        if match:
            matched_text = match.group(0)

            if matched_text not in context_lower:
                return {
                    "is_valid": False,
                    "reason": (
                        "Unsupported customer sentiment interpretation detected: "
                        f"'{matched_text}' is not stated in the retrieved evidence."
                    ),
                    "source": "deterministic_rule",
                }

    return {
        "is_valid": True,
        "reason": "No deterministic grounding violations detected.",
        "source": "deterministic_rule",
    }


def llm_research_validation(
    question: str,
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Use Qwen to validate whether all meaningful claims in the answer
    are grounded in retrieved evidence.

    The validator should be strict about unsupported claims while still
    accepting faithful paraphrases.
    """

    prompt = f"""
You are a strict but fair enterprise research-grounding validator.

Your task is to determine whether EVERY meaningful claim in the
RESEARCH ANSWER is supported by the RETRIEVED EVIDENCE.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

RESEARCH ANSWER:
{research_answer}

IMPORTANT PRINCIPLE:

Validate ONLY claims that actually appear in the research answer.

Do NOT require the answer to include every fact in the retrieved evidence.

An answer can focus only on the evidence relevant to the user's question.

STRICT VALIDATION RULES:

1. Every meaningful factual claim in the research answer must be either:
   - directly stated in the evidence, or
   - a faithful paraphrase that preserves exactly the same meaning.

2. Exact wording is not required.

3. Do not reject a correct paraphrase merely because it uses different wording.

Example:

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change to additional categories."

Valid:
"Management planned to monitor conversion rates and customer response
before deciding whether to expand the pricing change."

4. Do not require the answer to include every fact from the evidence.

Example:

If the retrieved evidence contains both:
- pricing information
- delivery-delay information

and the user asks about pricing, the answer may discuss only pricing.

That is VALID.

5. Do not invent business causes.

Evidence:
"Delivery delays were a recurring source of negative feedback."

Invalid:
"Supply chain failures caused the delivery delays."

6. Do not invent operational explanations.

Delivery delays do not automatically prove:
- warehouse failures
- logistics failures
- staffing shortages
- inventory problems
- supply chain problems

7. Do not invent customer satisfaction or dissatisfaction.

Evidence:
"Delivery delays were a recurring source of negative feedback."

Valid:
"Delivery delays were a recurring source of negative feedback."

Valid:
"The report identifies delivery delays as a recurring customer issue."

Invalid:
"Delivery delays negatively impacted customer satisfaction levels."

Invalid:
"Customers were dissatisfied with the company's delivery service."

The evidence says negative feedback, but does not explicitly state
customer satisfaction levels.

8. Do not strengthen sentiment.

Evidence:
"Recurring source of negative feedback."

Invalid:
"Customers were highly dissatisfied."

Invalid:
"Customers were extremely unhappy."

9. Do not invent frequency.

Evidence:
"Recurring source of negative feedback."

Invalid:
"Customers frequently complained."

10. Do not invent success conditions.

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change."

Invalid:
"Management will expand the pricing change if it is successful."

11. Do not invent outcomes.

A plan to monitor something does not prove:
- the monitoring occurred
- the strategy worked
- the strategy failed
- expansion happened

12. Do not invent business impact.

Do not infer:
- revenue impact
- profit impact
- customer retention impact
- market demand
- customer loyalty
unless explicitly stated.

13. Ignore formatting labels such as:

DOCUMENT FINDING
SUPPORTING EVIDENCE
SOURCES
INTERPRETATION
EVIDENCE SUFFICIENCY

14. Supporting evidence that directly repeats or faithfully summarizes
the retrieved evidence is valid.

15. Judge only the actual wording in the answer.

Do not reject an answer because you think a supported statement might
imply something stronger.

16. If even ONE meaningful claim is unsupported, return INVALID.

17. If ALL meaningful claims are supported, return VALID.

Return exactly one of these formats:

VALID: <brief reason>

or

INVALID: <brief reason>
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        return {
            "is_valid": False,
            "reason": (
                f"Research validator could not reach Ollama: {exc}"
            ),
            "source": "validator_error",
        }

    validation_text = (
        response
        .json()
        .get("response", "")
        .strip()
    )

    if validation_text.upper().startswith("VALID:"):
        return {
            "is_valid": True,
            "reason": validation_text.split(":", 1)[1].strip(),
            "source": "llm_validator",
        }

    if validation_text.upper().startswith("INVALID:"):
        return {
            "is_valid": False,
            "reason": validation_text.split(":", 1)[1].strip(),
            "source": "llm_validator",
        }

    return {
        "is_valid": False,
        "reason": (
            "Validator returned an unexpected response format: "
            f"{validation_text}"
        ),
        "source": "llm_validator",
    }


def validate_research_answer(
    question: str,
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Main Research Validator Version 4.

    Stage 1:
        Deterministic grounding checks.

    Stage 2:
        LLM claim-level grounding validation.

    Returns:
        {
            "is_valid": bool,
            "reason": str,
            "source": str
        }
    """

    deterministic_result = deterministic_research_checks(
        research_answer=research_answer,
        retrieved_context=retrieved_context,
    )

    if not deterministic_result["is_valid"]:
        return deterministic_result

    return llm_research_validation(
        question=question,
        research_answer=research_answer,
        retrieved_context=retrieved_context,
    )


# ============================================================
# TEST RESEARCH VALIDATOR VERSION 4
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RESEARCH GROUNDING VALIDATOR - VERSION 4")
    print("=" * 70)

    # ========================================================
    # CUSTOMER PROBLEM TEST DATA
    # ========================================================

    customer_question = (
        "What customer problems were reported?"
    )

    customer_context = """
Quarterly Business Report - Q2

Customer service teams reported that delivery delays were
a recurring source of negative feedback in some regions.
"""

    grounded_customer_answer = """
DOCUMENT FINDING
Customer service teams reported that delivery delays were
a recurring source of negative feedback in some regions.

INTERPRETATION
The report identifies delivery delays as a recurring customer issue.
"""

    supply_chain_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring customer problem.

INTERPRETATION
The company appears to have supply chain problems.
"""

    satisfaction_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring customer problem.

INTERPRETATION
Delivery delays affected customer satisfaction.
"""

    satisfaction_levels_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring source of negative feedback.

INTERPRETATION
Delivery delays negatively impacted customer satisfaction levels.
"""

    dissatisfaction_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring customer problem.

INTERPRETATION
The delays caused customer dissatisfaction.
"""

    frequency_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring customer problem.

INTERPRETATION
Customers frequently complained about delivery delays.
"""

    # ========================================================
    # PRICING TEST DATA
    # ========================================================

    pricing_question = (
        "What pricing changes were introduced in Q2?"
    )

    pricing_context = """
During Q2, the company increased average selling prices
for selected premium product categories by approximately
7 percent.

Management planned to monitor conversion rates and
customer response before expanding the pricing change
to additional categories.
"""

    grounded_pricing_answer = """
DOCUMENT FINDING
The company increased average selling prices for selected premium
product categories by approximately 7 percent during Q2.

INTERPRETATION
Management planned to monitor conversion rates and customer response
before deciding whether to expand the pricing change.
"""

    success_condition_answer = """
DOCUMENT FINDING
The company increased average selling prices for selected premium
product categories by approximately 7 percent.

INTERPRETATION
Management may expand the pricing change if it is successful.
"""

    # ========================================================
    # TEST CASES
    # ========================================================

    tests = [
        (
            "Grounded Customer Answer",
            customer_question,
            grounded_customer_answer,
            customer_context,
        ),
        (
            "Unsupported Supply Chain Answer",
            customer_question,
            supply_chain_answer,
            customer_context,
        ),
        (
            "Unsupported Satisfaction Answer",
            customer_question,
            satisfaction_answer,
            customer_context,
        ),
        (
            "Unsupported Satisfaction Levels Answer",
            customer_question,
            satisfaction_levels_answer,
            customer_context,
        ),
        (
            "Unsupported Dissatisfaction Answer",
            customer_question,
            dissatisfaction_answer,
            customer_context,
        ),
        (
            "Unsupported Frequency Answer",
            customer_question,
            frequency_answer,
            customer_context,
        ),
        (
            "Grounded Pricing Answer",
            pricing_question,
            grounded_pricing_answer,
            pricing_context,
        ),
        (
            "Unsupported Success Condition",
            pricing_question,
            success_condition_answer,
            pricing_context,
        ),
    ]

    # ========================================================
    # RUN TESTS
    # ========================================================

    for test_name, question, answer, context in tests:

        print("\n" + test_name)
        print("-" * 70)

        result = validate_research_answer(
            question=question,
            research_answer=answer,
            retrieved_context=context,
        )

        print(result)