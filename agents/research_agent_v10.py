from tools.llm_tool import ask_llm
from tools.research_validator_v7 import validate_research_answer
from rag.retriever import retrieve_chunks, build_retrieved_context


DEFAULT_TOP_K = 3
MAX_VALIDATION_ATTEMPTS = 3


# ============================================================
# RESEARCH AGENT - VERSION 10
# STRICT RAG + CLAIM-LEVEL GROUNDING VALIDATION
# ============================================================


def build_research_prompt(
    question: str,
    retrieved_context: str,
) -> str:
    """
    Build a conservative evidence-grounded research prompt.

    Version 10 preserves the stable Version 9 generation behavior
    while using Research Validator Version 7 for claim-level grounding.
    """

    return f"""
You are an enterprise research agent.

Answer the user's question using ONLY the retrieved enterprise evidence.

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
   - demand
   - customer preference
   - operational failures
   - supply chain problems
   - logistics problems
   - recommendations
   - outcomes
   - success conditions

3. Do not strengthen the evidence.

Example:

Evidence:
"Delivery delays were a recurring source of negative feedback."

ALLOWED:
"Delivery delays were reported as a recurring source of negative feedback."

NOT ALLOWED:
"Delivery delays significantly hurt customer satisfaction."

NOT ALLOWED:
"The company has a systemic logistics issue."

4. Do not add unsupported qualifiers.

Avoid words such as:
- widely
- significantly
- severely
- strongly
- extensively
- major
- systemic

unless the retrieved evidence explicitly supports them.

5. Do not infer recommendations.

Do not write:
- needs further investigation
- should be addressed
- requires resolution
- management should investigate
- the company should improve

unless the evidence explicitly states that.

6. Do not add strategy labels.

Do not call something:
- strategic
- successful
- effective
- important

unless that description is directly supported.

7. Do not invent causal relationships.

8. Do not turn monitoring plans into success conditions.

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change."

ALLOWED:
"Management planned to monitor conversion rates and customer response
before considering expansion to additional categories."

NOT ALLOWED:
"Management planned to expand the change if it was successful."

9. EVIDENCE SUFFICIENCY RULE:

If the retrieved evidence contains a direct factual answer to the
question, classify the evidence as:

EVIDENCE SUFFICIENCY
sufficient

Do NOT mark evidence insufficient merely because the retrieved document
may not list every possible answer.

Example:

Question:
"What customer problems were reported?"

Evidence:
"Delivery delays were a recurring source of negative feedback."

This is sufficient evidence to answer that delivery delays were reported.

10. Mark evidence insufficient only when the retrieved evidence does not
contain a direct or reasonably complete answer to the question.

11. SUPPORTING EVIDENCE must contain actual factual evidence.

Do NOT use:
- document titles
- section headings
- filenames

as supporting evidence.

12. SOURCES must contain the source filename shown after "Document:".

13. The INTERPRETATION section must remain extremely conservative.

It may:
- faithfully restate the evidence, or
- say that no additional interpretation is necessary.

14. If no additional interpretation adds value, write exactly:

The retrieved evidence directly answers the question, and no additional
interpretation is necessary.

15. Keep the answer concise.

Return exactly this structure:

DOCUMENT FINDING
<direct evidence-based answer>

SUPPORTING EVIDENCE
- <actual factual evidence sentence>
- <optional second factual evidence sentence>

SOURCES
- <source filename>

INTERPRETATION
<strictly grounded restatement or no-additional-interpretation sentence>

EVIDENCE SUFFICIENCY
sufficient

If the evidence genuinely does not answer the question:

DOCUMENT FINDING
The retrieved evidence does not provide enough information to answer
the question confidently.

SUPPORTING EVIDENCE
- <most relevant available factual evidence>

SOURCES
- <source filename>

INTERPRETATION
No additional interpretation is provided because the available evidence
is insufficient.

EVIDENCE SUFFICIENCY
insufficient
"""


def extract_rejected_phrase(
    validation_reason: str,
) -> str | None:
    """
    Attempt to extract an exact phrase identified in a validation
    failure message.
    """

    if not validation_reason:
        return None

    first_quote = validation_reason.find("'")

    if first_quote == -1:
        return None

    second_quote = validation_reason.find(
        "'",
        first_quote + 1,
    )

    if second_quote == -1:
        return None

    phrase = validation_reason[
        first_quote + 1:second_quote
    ].strip()

    if not phrase:
        return None

    return phrase


def build_rewrite_prompt(
    question: str,
    retrieved_context: str,
    previous_answer: str,
    validation_reason: str,
    rejected_phrase: str | None = None,
) -> str:
    """
    Build a strict correction prompt after grounding failure.
    """

    rejected_phrase_section = ""

    if rejected_phrase:

        rejected_phrase_section = f"""
IMPORTANT REJECTED PHRASE:

The grounding validator identified this text in the failed answer:

"{rejected_phrase}"

You MUST remove or correct the unsupported claim containing this text.

Do NOT:
- reuse unsupported wording
- preserve the same unsupported meaning using different words
- replace it with another unsupported interpretation

Replace the affected claim with a statement directly supported by the
retrieved evidence.
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

{rejected_phrase_section}

STRICT REWRITE RULES:

1. The rewritten answer MUST be materially different from the failed
answer wherever the validator identified a problem.

2. Remove the unsupported claim identified by the validator.

3. Do not return the same failed sentence again.

4. Remove any other unsupported interpretation.

5. Use only facts explicitly present in the retrieved evidence.

6. Do not invent:
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

7. Do not use unsupported qualifiers such as:
   - widely
   - significantly
   - severely
   - strongly
   - systemic
   - major

unless they occur in or are directly supported by the evidence.

8. If the evidence directly answers the user's question, use:

EVIDENCE SUFFICIENCY
sufficient

Do not mark the answer insufficient simply because the evidence does not
enumerate every possible answer.

9. SUPPORTING EVIDENCE must contain actual factual sentences from the
retrieved evidence.

10. Do not use document titles or filenames as supporting evidence.

11. SOURCES must contain the source filename from the retrieved evidence.

12. INTERPRETATION must remain conservative.

13. If interpretation is unnecessary, write exactly:

The retrieved evidence directly answers the question, and no additional
interpretation is necessary.

14. Before returning the answer, compare it with the validation failure
and make sure the unsupported claim has been removed.

Return exactly:

DOCUMENT FINDING
<direct grounded answer>

SUPPORTING EVIDENCE
- <actual grounded factual evidence>

SOURCES
- <source filename>

INTERPRETATION
<strictly grounded interpretation>

EVIDENCE SUFFICIENCY
sufficient

If there is genuinely not enough evidence, use:

EVIDENCE SUFFICIENCY
insufficient
"""


def research(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Run Research Agent Version 10.

    Flow:
    1. Retrieve evidence.
    2. Build grounded context.
    3. Generate initial answer.
    4. Validate using Research Validator V7.
    5. Rewrite unsupported claims when necessary.
    6. Return grounded answer or safe fallback.
    """

    print("=" * 70)
    print("RESEARCH AGENT - VERSION 10")
    print("STRICT RAG + CLAIM-LEVEL GROUNDING VALIDATION")
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

    if not retrieved_chunks:

        final_answer = """
DOCUMENT FINDING
No relevant enterprise evidence was retrieved.

SUPPORTING EVIDENCE
- No relevant factual evidence was found.

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

        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        print("\n" + "-" * 70)
        print(f"Retrieved Evidence {index}")
        print("Source:", source)

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

        print("\nRetrieved Text:")

        print(
            chunk.get(
                "text",
                "",
            )
        )

    # ========================================================
    # STEP 2: BUILD RAG CONTEXT
    # ========================================================

    print("\nBuilding grounded research context...")

    retrieved_context = build_retrieved_context(
        retrieved_chunks
    )

    # ========================================================
    # STEP 3: INITIAL GENERATION
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

    previous_failed_answers = set()

    # ========================================================
    # STEP 4: VALIDATION + SELF-CORRECTION LOOP
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
        # VALID
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
        # INVALID
        # ====================================================

        print(
            "\nResearch answer is not fully grounded."
        )

        validation_source = validation_result.get(
            "source",
            "unknown",
        )

        validation_reason = validation_result.get(
            "reason",
            "No reason provided.",
        )

        print("\nValidation Source:")
        print(validation_source)

        print("\nValidation Reason:")
        print(validation_reason)

        previous_failed_answers.add(
            research_answer.strip()
        )

        # ====================================================
        # STOP AFTER LAST ATTEMPT
        # ====================================================

        if attempt == MAX_VALIDATION_ATTEMPTS:
            break

        # ====================================================
        # EXTRACT REJECTED PHRASE / CLAIM
        # ====================================================

        rejected_phrase = extract_rejected_phrase(
            validation_reason
        )

        if rejected_phrase:

            print("\nRejected Phrase or Claim:")
            print(rejected_phrase)

        print(
            "\nSending grounding feedback back to Qwen "
            "for strict rewrite..."
        )

        # ====================================================
        # GENERATE REWRITE
        # ====================================================

        rewrite_prompt = build_rewrite_prompt(
            question=question,
            retrieved_context=retrieved_context,
            previous_answer=research_answer,
            validation_reason=validation_reason,
            rejected_phrase=rejected_phrase,
        )

        rewritten_answer = ask_llm(
            rewrite_prompt
        ).strip()

        # ====================================================
        # DETECT EXACT REPEAT
        # ====================================================

        if rewritten_answer in previous_failed_answers:

            print(
                "\nRewrite repeated a previously failed answer."
            )

            stronger_prompt = f"""
The previous rewrite was rejected because it repeated the same failed
answer.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

FAILED ANSWER:
{rewritten_answer}

VALIDATION FAILURE:
{validation_reason}

Write a NEW answer.

You MUST NOT repeat the failed interpretation or unsupported claim.

Use the safest possible approach:

- answer directly from the evidence
- use actual evidence under SUPPORTING EVIDENCE
- avoid unsupported qualifiers
- avoid recommendations
- avoid business interpretation
- avoid customer satisfaction claims
- avoid causal claims

For INTERPRETATION, use exactly:

The retrieved evidence directly answers the question, and no additional
interpretation is necessary.

Return exactly:

DOCUMENT FINDING
<direct grounded answer>

SUPPORTING EVIDENCE
- <actual evidence sentence>

SOURCES
- <source filename>

INTERPRETATION
The retrieved evidence directly answers the question, and no additional
interpretation is necessary.

EVIDENCE SUFFICIENCY
sufficient
"""

            rewritten_answer = ask_llm(
                stronger_prompt
            ).strip()

        research_answer = rewritten_answer

    # ========================================================
    # STEP 5: SAFE FALLBACK
    # ========================================================

    print(
        "\nMaximum research rewrite attempts reached."
    )

    source_names = []

    for chunk in retrieved_chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
        )

        if (
            source
            and source not in source_names
        ):
            source_names.append(
                source
            )

    if source_names:

        fallback_sources = "\n".join(
            f"- {source}"
            for source in source_names
        )

    else:

        fallback_sources = "- Unknown source"

    fallback_answer = f"""
DOCUMENT FINDING
Unable to produce a fully grounded research answer.

SUPPORTING EVIDENCE
- Retrieved evidence was available, but the generated answer could not
  pass grounding validation.

SOURCES
{fallback_sources}

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

    print(
        "Testing Research Agent Version 10"
    )

    test_questions = [
        "What pricing changes were introduced in Q2?",
        "What customer problems were reported?",
    ]

    for question in test_questions:

        print(
            "\n\n" + "#" * 70
        )

        print(
            "NEW RESEARCH QUESTION"
        )

        print(
            "#" * 70
        )

        result = research(
            question=question
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL RESEARCH ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            result["answer"]
        )

        print(
            "\nValidation Result:"
        )

        print(
            result["validation"]
        )

        print(
            "\nValidation Attempts:"
        )

        print(
            result["validation_attempts"]
        )

    print(
        "\nResearch Agent Version 10 testing completed."
    )