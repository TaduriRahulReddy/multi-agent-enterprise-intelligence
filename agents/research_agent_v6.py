from tools.llm_tool import ask_llm
from tools.research_validator_v4 import validate_research_answer
from rag.retriever import retrieve_chunks, build_retrieved_context


# ============================================================
# RESEARCH AGENT - VERSION 3
# RAG RETRIEVAL + GROUNDING VALIDATION + SELF-CORRECTION
# ============================================================


DEFAULT_TOP_K = 3
MAX_VALIDATION_ATTEMPTS = 3


def build_research_prompt(
    question: str,
    retrieved_context: str,
) -> str:
    """
    Build the initial evidence-grounded research prompt.
    """

    return f"""
You are a Research Agent in a multi-agent enterprise
intelligence system.

Your job is to answer the user's question using only
retrieved document evidence.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{retrieved_context}

STRICT RESEARCH RULES:

1. Use only information contained in the retrieved context.

2. Do not invent:
   - facts
   - metrics
   - percentages
   - dates
   - causes
   - policies
   - business outcomes

3. If the retrieved evidence is insufficient,
   clearly say so.

4. Do not claim causation unless explicitly documented.

5. Do not infer business impact beyond what the
   evidence directly supports.

6. Do not infer operational causes.

Example:

If the evidence says:
"Delivery delays were a recurring source of negative feedback."

You may say:
"Delivery delays were reported as a recurring customer problem."

You must NOT say:
"The company has supply chain problems."

7. If a document says a price increased by 7 percent,
   do not describe that as:
   - modest impact
   - major impact
   - successful pricing strategy
   unless the document explicitly says so.

8. Preserve important documented numbers exactly.

9. Include the source document names in the answer.

10. Keep the response concise and business-focused.

11. Do not mention:
    - embeddings
    - ChromaDB
    - vector stores
    - chunking
    - prompts
    - technical implementation

Return EXACTLY this structure:

DOCUMENT FINDING
<main evidence-based answer>

SUPPORTING EVIDENCE
- <evidence point 1>
- <evidence point 2 if available>

SOURCES
- <source document>

INTERPRETATION
<careful grounded interpretation>

EVIDENCE SUFFICIENCY
<sufficient or insufficient>
"""


def build_rewrite_prompt(
    question: str,
    retrieved_context: str,
    previous_answer: str,
    validation_reason: str,
) -> str:
    """
    Build a correction prompt when the previous research answer
    fails grounding validation.
    """

    return f"""
You are correcting a research answer that failed
grounding validation.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{retrieved_context}

PREVIOUS RESEARCH ANSWER:
{previous_answer}

GROUNDING VALIDATION FAILURE:
{validation_reason}

Rewrite the research answer so that every statement is
strictly supported by the retrieved evidence.

IMPORTANT CORRECTION RULES:

1. Remove every unsupported inference.

2. Do not invent causes or explanations.

3. Do not infer:
   - supply chain problems
   - logistical problems
   - customer loyalty
   - customer satisfaction
   - strong demand
   - business success
   - strategic effectiveness

unless explicitly stated in the evidence.

4. Preserve documented numbers exactly.

5. Supporting evidence must come directly from the
   retrieved context.

6. The INTERPRETATION section must remain conservative.

7. If evidence is insufficient, say so.

8. Do not add information that was not present in the
   retrieved document evidence.

Return EXACTLY this structure:

DOCUMENT FINDING
<main evidence-based answer>

SUPPORTING EVIDENCE
- <evidence point 1>
- <evidence point 2 if available>

SOURCES
- <source document>

INTERPRETATION
<careful grounded interpretation>

EVIDENCE SUFFICIENCY
<sufficient or insufficient>
"""


def build_insufficient_answer() -> str:
    """
    Return a safe response when no relevant evidence is retrieved.
    """

    return """
DOCUMENT FINDING
No relevant document evidence was retrieved.

SUPPORTING EVIDENCE
- No supporting evidence is available.

SOURCES
- None

INTERPRETATION
The question cannot be answered reliably from the available enterprise documents.

EVIDENCE SUFFICIENCY
insufficient
""".strip()


def research(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Complete Research Agent Version 3 workflow.

    Question
        ↓
    Semantic Retrieval
        ↓
    Relevant Chunks
        ↓
    Build Context
        ↓
    Qwen Research Answer
        ↓
    Research Grounding Validator
        ↓
    Valid?
        ├── Yes → Return answer
        └── No  → Rewrite using validator feedback
                         ↓
                     Revalidate
    """

    print("\n" + "=" * 70)
    print("RESEARCH AGENT - VERSION 3")
    print("RAG + GROUNDING VALIDATION + SELF-CORRECTION")
    print("=" * 70)

    print("\nResearch Question:")
    print(question)

    # --------------------------------------------------------
    # STEP 1: Retrieve relevant chunks
    # --------------------------------------------------------

    print("\nSearching enterprise documents...")

    retrieved_chunks = retrieve_chunks(
        question=question,
        top_k=top_k,
    )

    print(
        f"Retrieved chunks: "
        f"{len(retrieved_chunks)}"
    )

    # --------------------------------------------------------
    # STEP 2: Handle no retrieval
    # --------------------------------------------------------

    if not retrieved_chunks:

        answer = build_insufficient_answer()

        return {
            "question": question,
            "retrieved_chunks": [],
            "context": "",
            "answer": answer,
            "validation": {
                "is_valid": True,
                "reason": (
                    "No evidence was retrieved, so the agent "
                    "returned an explicit insufficient-evidence response."
                ),
                "source": "system_rule",
            },
            "validation_attempts": 0,
        }

    # --------------------------------------------------------
    # STEP 3: Display retrieved evidence
    # --------------------------------------------------------

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"Retrieved Evidence {rank}"
        )

        print(
            f"Source: "
            f"{chunk['metadata'].get('source')}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Distance: "
            f"{chunk['distance']}"
        )

    # --------------------------------------------------------
    # STEP 4: Build retrieved context
    # --------------------------------------------------------

    context = build_retrieved_context(
        retrieved_chunks
    )

    print(
        "\nBuilding grounded research context..."
    )

    # --------------------------------------------------------
    # STEP 5: Generate initial research answer
    # --------------------------------------------------------

    prompt = build_research_prompt(
        question=question,
        retrieved_context=context,
    )

    print(
        "Generating evidence-based answer "
        "using Qwen 2.5 7B..."
    )

    response = ask_llm(
        prompt
    )

    answer = response.strip()

    # --------------------------------------------------------
    # STEP 6: Validate and self-correct
    # --------------------------------------------------------

    final_validation = None
    validation_attempts = 0

    for attempt in range(
        1,
        MAX_VALIDATION_ATTEMPTS + 1,
    ):

        validation_attempts = attempt

        print(
            f"\nResearch Validation Attempt {attempt}:"
        )

        print("-" * 70)
        print(answer)
        print("-" * 70)

        validation_result = validate_research_answer(
            question=question,
            research_answer=answer,
            retrieved_context=context,
        )

        final_validation = validation_result

        print("\nGrounding Validation:")
        print(validation_result)

        # ----------------------------------------------------
        # VALID ANSWER
        # ----------------------------------------------------

        if validation_result["is_valid"]:

            print(
                "\nResearch answer validated successfully "
                f"on attempt {attempt}."
            )

            break

        # ----------------------------------------------------
        # INVALID ANSWER
        # ----------------------------------------------------

        print("\nResearch answer is not fully grounded.")

        print("\nValidation Source:")
        print(
            validation_result["source"]
        )

        print("\nValidation Reason:")
        print(
            validation_result["reason"]
        )

        # ----------------------------------------------------
        # STOP IF MAX ATTEMPTS REACHED
        # ----------------------------------------------------

        if attempt == MAX_VALIDATION_ATTEMPTS:

            print(
                "\nMaximum research rewrite attempts reached."
            )

            answer = """
DOCUMENT FINDING
Unable to produce a fully grounded research answer.

SUPPORTING EVIDENCE
- Retrieved evidence was available, but the generated answer could not pass grounding validation.

SOURCES
- See retrieved enterprise evidence.

INTERPRETATION
No additional interpretation is provided because grounding validation failed.

EVIDENCE SUFFICIENCY
insufficient
""".strip()

            break

        # ----------------------------------------------------
        # REWRITE USING VALIDATOR FEEDBACK
        # ----------------------------------------------------

        print(
            "\nSending grounding feedback back to Qwen "
            "for strict rewrite..."
        )

        rewrite_prompt = build_rewrite_prompt(
            question=question,
            retrieved_context=context,
            previous_answer=answer,
            validation_reason=validation_result["reason"],
        )

        rewritten_response = ask_llm(
            rewrite_prompt
        )

        answer = rewritten_response.strip()

    # --------------------------------------------------------
    # STEP 7: Return structured result
    # --------------------------------------------------------

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "answer": answer,
        "validation": final_validation,
        "validation_attempts": validation_attempts,
    }


# ============================================================
# TEST RESEARCH AGENT VERSION 3
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Research Agent Version 3"
    )

    test_questions = [
        (
            "What pricing changes were "
            "introduced in Q2?"
        ),
        (
            "What customer problems were "
            "reported?"
        ),
    ]

    for question in test_questions:

        print("\n\n" + "#" * 70)
        print("NEW RESEARCH QUESTION")
        print("#" * 70)

        result = research(
            question=question,
            top_k=3,
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
        "\nResearch Agent Version 3 "
        "testing completed."
    )