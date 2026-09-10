import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


def deterministic_research_checks(research_answer: str, retrieved_context: str) -> dict:
    """
    Run simple deterministic checks before calling the LLM validator.

    This is mainly designed to catch common unsupported inference patterns
    in RAG answers.
    """

    answer_lower = research_answer.lower()
    context_lower = retrieved_context.lower()

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
    ]

    for phrase in risky_phrases:
        if phrase in answer_lower and phrase not in context_lower:
            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported interpretation detected: '{phrase}' appears "
                    "in the research answer but is not supported by the retrieved evidence."
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
    Ask Qwen to determine whether the research answer is fully grounded
    in the retrieved enterprise evidence.
    """

    prompt = f"""
You are a strict research-grounding validator.

Your job is to determine whether a research answer is fully supported
by retrieved enterprise documents.

You must be conservative.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

RESEARCH ANSWER:
{research_answer}

VALIDATION RULES:

1. Every factual claim must be directly supported by the retrieved evidence.

2. Do not allow the model to invent causes, explanations, business effects,
   customer behavior, operational problems, or recommendations that are
   not explicitly supported.

3. A reasonable paraphrase of the evidence is allowed.

4. A summary of explicitly stated evidence is allowed.

5. Do NOT infer causal explanations.

Example:
Evidence:
"Delivery delays were a recurring source of negative feedback."

Allowed:
"Customers reported delivery delays."

Not allowed:
"The company has supply chain problems."

6. Do NOT infer sentiment strength beyond what the source states.

7. Do NOT infer business strategy beyond what the source explicitly states.

8. Do NOT reject an answer because of text that is not actually present
   in the answer.

9. Supporting evidence quotations or summaries should correspond to the
   retrieved evidence.

10. If the answer contains both grounded and unsupported statements,
    the entire answer should be considered invalid.

Return exactly one of these formats:

VALID: <brief reason>

or

INVALID: <brief reason>
"""

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

    validation_text = response.json()["response"].strip()

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
    Main validation function.

    Step 1:
        Run deterministic grounding checks.

    Step 2:
        If deterministic checks pass, run the LLM grounding validator.

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


if __name__ == "__main__":

    print("=" * 70)
    print("RESEARCH GROUNDING VALIDATOR - VERSION 1")
    print("=" * 70)

    question = "What customer problems were reported in the quarterly report?"

    context = """
Quarterly Business Report - Q2

Customer service teams reported that delivery delays were a recurring
source of negative feedback in some regions.
"""

    grounded_answer = """
DOCUMENT FINDING
Customer service teams reported that delivery delays were a recurring
source of negative feedback in some regions.

INTERPRETATION
The report identifies delivery delays as a recurring customer complaint.
"""

    unsupported_answer = """
DOCUMENT FINDING
Customer service teams reported that delivery delays were a recurring
source of negative feedback in some regions.

INTERPRETATION
The company appears to have supply chain and logistical issues that are
causing customer dissatisfaction.
"""

    print("\nGrounded Answer Test")
    print("-" * 70)

    grounded_result = validate_research_answer(
        question=question,
        research_answer=grounded_answer,
        retrieved_context=context,
    )

    print(grounded_result)

    print("\nUnsupported Answer Test")
    print("-" * 70)

    unsupported_result = validate_research_answer(
        question=question,
        research_answer=unsupported_answer,
        retrieved_context=context,
    )

    print(unsupported_result)