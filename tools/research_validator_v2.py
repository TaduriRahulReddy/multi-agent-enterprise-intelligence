import requests
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# ============================================================
# RESEARCH GROUNDING VALIDATOR - VERSION 2
# STRICT PHRASE + QUALIFIER + LLM GROUNDING VALIDATION
# ============================================================


def deterministic_research_checks(
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Run deterministic checks before calling the LLM validator.

    Version 2 adds:
    - unsupported operational/business inference checks
    - unsupported qualifier checks
    - unsupported success/frequency/intensity language checks
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

        if (
            phrase in answer_lower
            and phrase not in context_lower
        ):

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

        if (
            phrase in answer_lower
            and phrase not in context_lower
        ):

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported conditional interpretation detected: "
                    f"'{phrase}' is not stated in the retrieved evidence."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # CHECK 3: Unsupported strength/frequency qualifiers
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

        if (
            phrase in answer_lower
            and phrase not in context_lower
        ):

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

        match = re.search(
            pattern,
            answer_lower,
        )

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
        "reason": (
            "No deterministic grounding violations detected."
        ),
        "source": "deterministic_rule",
    }


def llm_research_validation(
    question: str,
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Ask Qwen to perform strict claim-level grounding validation.
    """

    prompt = f"""
You are a strict research-grounding validator.

Your task is to determine whether EVERY meaningful claim in a
research answer is fully supported by retrieved enterprise evidence.

Be conservative.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

RESEARCH ANSWER:
{research_answer}

STRICT VALIDATION RULES:

1. Every factual claim must be directly supported by the evidence.

2. Reasonable paraphrasing is allowed only when the meaning does not change.

3. Do not allow invented:
   - causes
   - business effects
   - operational explanations
   - customer behavior
   - strategy outcomes
   - recommendations
   - success conditions

4. Do not strengthen the evidence.

Example:

Evidence:
"Delivery delays were a recurring source of negative feedback."

Allowed:
"Delivery delays were reported as a recurring source of negative feedback."

Allowed:
"The report identifies delivery delays as a recurring customer issue."

Not allowed:
"Customers frequently complained about delivery delays."

The word "frequently" strengthens the evidence.

5. Do not convert monitoring plans into success conditions.

Example:

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change."

Allowed:
"Management planned to monitor conversion rates and customer response
before deciding whether to expand the pricing change."

Not allowed:
"Management planned to expand the change if it was successful."

Success was not explicitly stated.

6. Do not infer causal explanations.

Example:

Evidence:
"Delivery delays were a recurring source of negative feedback."

Not allowed:
"Supply chain problems caused customer dissatisfaction."

7. Do not infer operational problems from symptoms.

Delivery delays do NOT automatically prove:
- supply chain failures
- logistics failures
- warehouse problems
- staffing shortages

8. Do not infer customer satisfaction, loyalty, demand, or preference
unless explicitly stated.

9. Do not reject an answer because of words or claims that do not
actually appear in the research answer.

10. If one meaningful statement is unsupported, mark the entire
answer INVALID.

11. Ignore section headings such as:
DOCUMENT FINDING
SUPPORTING EVIDENCE
SOURCES
INTERPRETATION
EVIDENCE SUFFICIENCY

Focus only on the claims inside those sections.

Return exactly one format:

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
            "reason": (
                validation_text
                .split(":", 1)[1]
                .strip()
            ),
            "source": "llm_validator",
        }

    if validation_text.upper().startswith("INVALID:"):

        return {
            "is_valid": False,
            "reason": (
                validation_text
                .split(":", 1)[1]
                .strip()
            ),
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
    Main Version 2 research validation function.

    Stage 1:
        Deterministic grounding checks.

    Stage 2:
        LLM claim-level grounding validation.
    """

    deterministic_result = (
        deterministic_research_checks(
            research_answer=research_answer,
            retrieved_context=retrieved_context,
        )
    )

    if not deterministic_result["is_valid"]:

        return deterministic_result

    return llm_research_validation(
        question=question,
        research_answer=research_answer,
        retrieved_context=retrieved_context,
    )


# ============================================================
# TEST RESEARCH VALIDATOR VERSION 2
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RESEARCH GROUNDING VALIDATOR - VERSION 2")
    print("=" * 70)

    customer_question = (
        "What customer problems were reported "
        "in the quarterly report?"
    )

    customer_context = """
Quarterly Business Report - Q2

Customer service teams reported that delivery delays were
a recurring source of negative feedback in some regions.
"""

    grounded_customer_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring source of
negative feedback in some regions.

INTERPRETATION
The report identifies delivery delays as a recurring
customer issue.
"""

    supply_chain_answer = """
DOCUMENT FINDING
Delivery delays were reported as a recurring customer problem.

INTERPRETATION
The company appears to have supply chain problems.
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
The company increased average selling prices for selected
premium product categories by approximately 7 percent.

INTERPRETATION
Management planned to monitor conversion rates and customer
response before deciding whether to expand the pricing change.
"""

    success_condition_answer = """
DOCUMENT FINDING
The company increased average selling prices for selected
premium product categories by approximately 7 percent.

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

    for (
        test_name,
        question,
        answer,
        context,
    ) in tests:

        print("\n" + test_name)
        print("-" * 70)

        result = validate_research_answer(
            question=question,
            research_answer=answer,
            retrieved_context=context,
        )

        print(result)