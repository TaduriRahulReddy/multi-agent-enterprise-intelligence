from tools.llm_tool import ask_llm
from tools.research_validator_v4 import validate_research_answer
from rag.retriever import retrieve_chunks, build_retrieved_context


DEFAULT_TOP_K = 3
MAX_VALIDATION_ATTEMPTS = 3


# ============================================================
# RESEARCH AGENT - VERSION 7
# STRICT RAG + GROUNDING VALIDATION + SELF-CORRECTION
# ============================================================


def build_research_prompt(
    question: str,
    retrieved_context: str,
) -> str:
    """
    Build a conservative RAG prompt.

    Version 7 reduces hallucination by forcing the INTERPRETATION
    section to stay very close to retrieved evidence.
    """

    return f"""
You are an enterprise research agent.

Your job is to answer the user's question using ONLY the retrieved
enterprise evidence.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

STRICT GROUNDING RULES:

1. Use only information explicitly present in the retrieved evidence.

2. Do not invent:
   - causes
   - business impact
   - customer satisfaction
   - customer dissatisfaction
   - customer loyalty
   - customer demand
   - customer preference
   - operational failures
   - supply chain issues
   - logistics problems
   - recommendations
   - outcomes
   - future success conditions

3. Do not strengthen the evidence.

Example:

Evidence:
"Delivery delays were a recurring source of negative feedback."

ALLOWED:
"Delivery delays were reported as a recurring source of negative feedback."

ALLOWED:
"The report identifies delivery delays as a recurring customer issue."

NOT ALLOWED:
"Customers were highly dissatisfied."

NOT ALLOWED:
"Delivery delays affected customer satisfaction."

NOT ALLOWED:
"The company has a systemic logistics problem."

4. Do not infer recommendations.

Do not write statements such as:
- "needs further investigation"
- "should be addressed"
- "requires resolution"
- "management should investigate"
- "the company should improve"

unless the retrieved evidence explicitly states them.

5. Do not add strategy labels.

Do not describe something as:
- strategic
- successful
- significant
- major
- systemic
- widespread
- highly effective
- important

unless that meaning is explicitly supported by the source.

6. Do not invent causal relationships.

Example:

Evidence:
"Delivery delays were a recurring source of negative feedback."

NOT ALLOWED:
"Supply chain problems caused customer dissatisfaction."

7. Do not turn monitoring plans into success conditions.

Example:

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change."

ALLOWED:
"Management planned to monitor conversion rates and customer response
before deciding whether to expand the pricing change."

NOT ALLOWED:
"Management planned to expand the pricing change if it was successful."

8. The INTERPRETATION section must remain extremely conservative.

It should only:
- restate the evidence in simpler language, or
- connect directly stated facts without adding new meaning.

9. If there is no useful interpretation beyond the evidence, write exactly:

The retrieved evidence directly answers the question, and no additional
interpretation is necessary.

10. Do not introduce information from general knowledge.

11. Do not use outside assumptions.

12. Keep the answer concise and factual.

13. If the retrieved evidence does not answer the question adequately,
mark the evidence as insufficient.

Return exactly this format:

DOCUMENT FINDING
<direct evidence-based answer>

SUPPORTING EVIDENCE
- <first relevant evidence point>
- <optional second relevant evidence point>

SOURCES
- <source filename>

INTERPRETATION
<conservative evidence-based restatement>

EVIDENCE SUFFICIENCY
sufficient

If the evidence is not sufficient, return:

DOCUMENT FINDING
The retrieved evidence does not provide enough information to answer
the question confidently.

SUPPORTING EVIDENCE
- <most relevant available evidence>

SOURCES
- <source filename>

INTERPRETATION
No additional interpretation is provided because the available evidence
is insufficient.

EVIDENCE SUFFICIENCY
insufficient
"""


def build_rewrite_prompt(
    question: str,
    retrieved_context: str,
    previous_answer: str,
    validation_reason: str,
) -> str:
    """
    Build a strict rewrite prompt when grounding validation fails.
    """

    return f"""
You are rewriting an enterprise research answer because the previous
answer failed grounding validation.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

PREVIOUS ANSWER:
{previous_answer}

GROUNDING VALIDATION FAILURE:
{validation_reason}

Rewrite the answer so that EVERY meaningful claim is directly supported
by the retrieved evidence.

STRICT REWRITE RULES:

1. Remove the unsupported claim identified by the validator.

2. Remove any other unsupported interpretation.

3. Use only facts explicitly present in the retrieved evidence.

4. Do not invent:
   - causes
   - recommendations
   - customer satisfaction
   - customer dissatisfaction
   - customer loyalty
   - supply chain problems
   - logistics failures
   - business impact
   - demand
   - strategy
   - outcomes
   - success conditions

5. Do not strengthen wording.

6. Do not add new facts while trying to correct the previous answer.

7. The INTERPRETATION section should be a conservative restatement
of the evidence.

8. If no additional interpretation is necessary, use exactly:

The retrieved evidence directly answers the question, and no additional
interpretation is necessary.

9. Keep the answer concise.

Return exactly:

DOCUMENT FINDING
<direct grounded answer>

SUPPORTING EVIDENCE
- <grounded evidence>

SOURCES
- <source filename>

INTERPRETATION
<strictly grounded interpretation>

EVIDENCE SUFFICIENCY
sufficient

If there is not enough evidence, use:

EVIDENCE SUFFICIENCY
insufficient
"""


def research(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Run the Research Agent Version 7.

    Flow:
    1. Retrieve document chunks.
    2. Build RAG context.
    3. Generate grounded research answer.
    4. Validate the answer.
    5. Rewrite when validation fails.
    6. Return the validated answer and metadata.
    """

    print("=" * 70)
    print("RESEARCH AGENT - VERSION 7")
    print("STRICT RAG + GROUNDING VALIDATION + SELF-CORRECTION")
    print("=" * 70)

    print("\nResearch Question:")
    print(question)

    # ========================================================
    # STEP 1: RETRIEVE DOCUMENT EVIDENCE
    # ========================================================

    print("\nSearching enterprise documents...")

    retrieved_chunks = retrieve_chunks(
        question=question,
        top_k=top_k,
    )

    print(f"Retrieved chunks: {len(retrieved_chunks)}")

    # ========================================================
    # HANDLE NO RETRIEVAL RESULTS
    # ========================================================

    if not retrieved_chunks:

        final_answer = """
DOCUMENT FINDING
No relevant enterprise evidence was retrieved.

SUPPORTING EVIDENCE
- No relevant document chunks were found.

SOURCES
- No source available.

INTERPRETATION
No additional interpretation is provided because relevant evidence
was not retrieved.

EVIDENCE SUFFICIENCY
insufficient
""".strip()

        return {
            "question": question,
            "retrieved_chunks": [],
            "context": "",
            "answer": final_answer,
            "validation": {
                "is_valid": False,
                "reason": "No relevant enterprise evidence was retrieved.",
                "source": "retrieval",
            },
            "validation_attempts": 0,
        }

    # ========================================================
    # DISPLAY RETRIEVED EVIDENCE
    # ========================================================

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"Retrieved Evidence {index}")

        print(
            "Source:",
            chunk.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Chunk ID:",
            chunk.get(
                "chunk_id",
                "Unknown",
            ),
        )

        print(
            "Distance:",
            chunk.get(
                "distance",
                "Unknown",
            ),
        )

    # ========================================================
    # STEP 2: BUILD CONTEXT
    # ========================================================

    print("\nBuilding grounded research context...")

    retrieved_context = build_retrieved_context(
        retrieved_chunks
    )

    # ========================================================
    # STEP 3: GENERATE INITIAL ANSWER
    # ========================================================

    print(
        "Generating evidence-based answer using Qwen 2.5 7B..."
    )

    prompt = build_research_prompt(
        question=question,
        retrieved_context=retrieved_context,
    )

    research_answer = ask_llm(
        prompt
    ).strip()

    validation_result = None

    # ========================================================
    # STEP 4: VALIDATE + SELF-CORRECT
    # ========================================================

    for attempt in range(
        1,
        MAX_VALIDATION_ATTEMPTS + 1,
    ):

        print(
            f"\nResearch Validation Attempt {attempt}:"
        )

        print("-" * 70)
        print(research_answer)
        print("-" * 70)

        validation_result = validate_research_answer(
            question=question,
            research_answer=research_answer,
            retrieved_context=retrieved_context,
        )

        print("\nGrounding Validation:")
        print(validation_result)

        # ====================================================
        # VALID ANSWER
        # ====================================================

        if validation_result["is_valid"]:

            print(
                f"\nResearch answer validated successfully "
                f"on attempt {attempt}."
            )

            return {
                "question": question,
                "retrieved_chunks": retrieved_chunks,
                "context": retrieved_context,
                "answer": research_answer,
                "validation": validation_result,
                "validation_attempts": attempt,
            }

        # ====================================================
        # INVALID ANSWER
        # ====================================================

        print(
            "\nResearch answer is not fully grounded."
        )

        print("\nValidation Source:")
        print(
            validation_result.get(
                "source",
                "unknown",
            )
        )

        print("\nValidation Reason:")
        print(
            validation_result.get(
                "reason",
                "No reason provided.",
            )
        )

        # ====================================================
        # LAST ATTEMPT
        # ====================================================

        if attempt == MAX_VALIDATION_ATTEMPTS:
            break

        print(
            "\nSending grounding feedback back to Qwen "
            "for strict rewrite..."
        )

        rewrite_prompt = build_rewrite_prompt(
            question=question,
            retrieved_context=retrieved_context,
            previous_answer=research_answer,
            validation_reason=validation_result.get(
                "reason",
                "Grounding validation failed.",
            ),
        )

        research_answer = ask_llm(
            rewrite_prompt
        ).strip()

    # ========================================================
    # STEP 5: SAFE FALLBACK
    # ========================================================

    print(
        "\nMaximum research rewrite attempts reached."
    )

    fallback_answer = """
DOCUMENT FINDING
Unable to produce a fully grounded research answer.

SUPPORTING EVIDENCE
- Retrieved evidence was available, but the generated answer could not
  pass grounding validation.

SOURCES
- See retrieved enterprise evidence.

INTERPRETATION
No additional interpretation is provided because grounding validation
failed.

EVIDENCE SUFFICIENCY
insufficient
""".strip()

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "context": retrieved_context,
        "answer": fallback_answer,
        "validation": validation_result,
        "validation_attempts": MAX_VALIDATION_ATTEMPTS,
    }


# ============================================================
# LOCAL TESTS
# ============================================================

if __name__ == "__main__":

    print("Testing Research Agent Version 7")

    test_questions = [
        "What pricing changes were introduced in Q2?",
        "What customer problems were reported?",
    ]

    for question in test_questions:

        print("\n\n" + "#" * 70)
        print("NEW RESEARCH QUESTION")
        print("#" * 70)

        result = research(
            question=question
        )

        print("\n" + "=" * 70)
        print("FINAL RESEARCH ANSWER")
        print("=" * 70)

        print(
            result["answer"]
        )

        print("\nValidation Result:")
        print(
            result["validation"]
        )

        print("\nValidation Attempts:")
        print(
            result["validation_attempts"]
        )

    print(
        "\nResearch Agent Version 7 testing completed."
    )