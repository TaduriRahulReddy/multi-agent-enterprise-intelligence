import requests
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# ============================================================
# RESEARCH GROUNDING VALIDATOR - VERSION 3
# BALANCED CLAIM-LEVEL GROUNDING VALIDATION
# ============================================================


def deterministic_research_checks(
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Run deterministic checks before calling the LLM validator.

    These checks catch common unsupported claims such as:
    - invented operational explanations
    - unsupported success conditions
    - unsupported strength/frequency qualifiers
    - unsupported causal language
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
    Use Qwen to validate whether the answer is grounded in the
    retrieved evidence.

    Version 3 validates claims independently and does not require
    the answer to summarize every fact in the retrieved context.
    """

    prompt = f"""
You are a strict but fair enterprise research-grounding validator.

Your job is to decide whether the RESEARCH ANSWER is supported by
the RETRIEVED EVIDENCE.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

RESEARCH ANSWER:
{research_answer}

IMPORTANT VALIDATION PRINCIPLE:

Validate ONLY claims that actually appear in the research answer.

Do NOT require the answer to include every fact in the retrieved evidence.

An answer may focus only on the evidence relevant to the user's question.

STRICT RULES:

1. For each meaningful factual claim in the research answer, determine
   whether it is directly stated or faithfully paraphrased by the
   retrieved evidence.

2. A claim is VALID if it preserves the same meaning as the evidence.

3. Exact wording is NOT required.

Example:

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change to additional categories."

Valid paraphrase:
"Management planned to monitor conversion rates and customer response
before deciding whether to expand the pricing change."

This is VALID.

4. Do not reject a statement simply because the evidence describes
   a plan rather than a completed action.

Example:

Evidence:
"Management planned to monitor conversion rates."

Answer:
"Management planned to monitor conversion rates."

This is VALID.

5. Do not require proof of implementation or outcome unless the answer
   explicitly claims implementation or outcome.

6. Do NOT require the answer to mention every retrieved fact.

If the evidence contains information about pricing and delivery delays,
and the question asks only about pricing, the answer may discuss only
pricing.

That is VALID.

7. Do not invent causes.

Evidence:
"Delivery delays were a recurring source of negative feedback."

Invalid:
"Supply chain problems caused the delivery delays."

8. Do not invent stronger customer sentiment.

Evidence:
"Delivery delays were a recurring source of negative feedback."

Valid:
"Delivery delays were a recurring customer issue."

Invalid:
"Customers were highly dissatisfied with delivery performance."

9. Do not invent frequency.

Evidence:
"Recurring source of negative feedback."

Valid:
"Recurring source of negative feedback."

Invalid:
"Customers frequently complained about delivery delays."

10. Do not invent success conditions.

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change."

Invalid:
"Management will expand the pricing change if it is successful."

11. Do not infer business strategy, operational failures, customer
    loyalty, demand, satisfaction, or financial impact unless explicitly
    supported.

12. Evaluate each claim independently.

If every meaningful claim is supported, return VALID.

If even one meaningful claim is unsupported, return INVALID.

13. Ignore formatting labels such as:

DOCUMENT FINDING
SUPPORTING EVIDENCE
SOURCES
INTERPRETATION
EVIDENCE SUFFICIENCY

14. Evidence quoted or paraphrased directly from the retrieved context
    is valid.

15. Do not reject a claim because you think it could imply something
    stronger. Judge only the actual wording of the claim.

Return exactly:

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
    Main Version 3 validation function.

    Stage 1:
        Deterministic grounding checks.

    Stage 2:
        Balanced claim-level LLM validation.
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
# TEST RESEARCH VALIDATOR VERSION 3
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RESEARCH GROUNDING VALIDATOR - VERSION 3")
    print("=" * 70)

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

    frequency_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring customer problem.

INTERPRETATION
Customers frequently complained about delivery delays.
"""

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

    for test_name, question, answer, context in tests:

        print("\n" + test_name)
        print("-" * 70)

        result = validate_research_answer(
            question=question,
            research_answer=answer,
            retrieved_context=context,
        )

        print(result)