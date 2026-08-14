from tools.llm_tool import ask_llm


# ============================================================
# RESEARCH AGENT - VERSION 1
# EVIDENCE-BASED DOCUMENT REASONING
# ============================================================


def build_research_prompt(
    question: str,
    retrieved_context: str,
) -> str:
    """
    Build a prompt that asks the Research Agent
    to answer using only retrieved document evidence.
    """

    return f"""
You are a Research Agent in a multi-agent enterprise
intelligence system.

Your job is to answer business questions using retrieved
document evidence.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{retrieved_context}

STRICT RESEARCH RULES:

1. Use only information present in the retrieved context.

2. Do not invent facts, metrics, policies, dates,
   explanations, or causes.

3. If the context does not contain enough evidence,
   clearly say that the available evidence is insufficient.

4. Do not claim causation unless the retrieved evidence
   explicitly supports it.

5. Distinguish documented facts from interpretation.

6. If multiple pieces of evidence are available,
   synthesize them carefully.

7. Do not mention embeddings, vector databases,
   chunking, prompts, or technical implementation.

8. Keep the answer concise and business-focused.

Return exactly this structure:

DOCUMENT FINDING
<main evidence-based finding>

SUPPORTING EVIDENCE
- <evidence point 1>
- <evidence point 2 if available>

INTERPRETATION
<careful grounded interpretation>

EVIDENCE SUFFICIENCY
<sufficient or insufficient>
"""


def analyze_documents(
    question: str,
    retrieved_context: str,
) -> str:
    """
    Generate an evidence-grounded research response.
    """

    if not retrieved_context.strip():

        return """
DOCUMENT FINDING
No relevant document evidence was retrieved.

SUPPORTING EVIDENCE
- No supporting evidence is available.

INTERPRETATION
The question cannot be answered reliably from the
available document context.

EVIDENCE SUFFICIENCY
insufficient
""".strip()

    prompt = build_research_prompt(
        question=question,
        retrieved_context=retrieved_context,
    )

    response = ask_llm(
        prompt
    )

    return response.strip()


# ============================================================
# TEST RESEARCH AGENT VERSION 1
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Research Agent Version 1"
    )

    print("=" * 70)

    test_question = (
        "What pricing changes were introduced?"
    )

    test_context = """
Quarterly Business Report:

During Q2, the company increased average selling prices
for selected premium product categories by approximately
7 percent.

The report also stated that management planned to monitor
conversion rates and customer response before expanding
the pricing change to additional categories.
"""

    result = analyze_documents(
        question=test_question,
        retrieved_context=test_context,
    )

    print("\nResearch Result:")
    print("-" * 70)
    print(result)
    print("-" * 70)

    print(
        "\nResearch Agent Version 1 "
        "test completed successfully."
    )