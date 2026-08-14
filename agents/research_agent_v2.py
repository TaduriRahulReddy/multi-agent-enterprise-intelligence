from tools.llm_tool import ask_llm
from rag.retriever import retrieve_chunks, build_retrieved_context


# ============================================================
# RESEARCH AGENT - VERSION 2
# AUTOMATIC RAG RETRIEVAL + GROUNDED DOCUMENT REASONING
# ============================================================


DEFAULT_TOP_K = 3


def build_research_prompt(
    question: str,
    retrieved_context: str,
) -> str:
    """
    Build the evidence-grounded research prompt.
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

6. If a document says a price increased by 7 percent,
   do not describe that as:
   - modest impact
   - major impact
   - successful pricing strategy
   unless the document explicitly says so.

7. Preserve important documented numbers exactly.

8. Include the source document names in the answer.

9. Keep the response concise and business-focused.

10. Do not mention:
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


def research(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Complete RAG Research Agent workflow.

    Question
        ↓
    Semantic Retrieval
        ↓
    Relevant Chunks
        ↓
    Build Context
        ↓
    Qwen
        ↓
    Grounded Research Answer
    """

    print("\n" + "=" * 70)
    print("RESEARCH AGENT - VERSION 2")
    print("AUTOMATIC RAG RETRIEVAL")
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

        answer = """
DOCUMENT FINDING
No relevant document evidence was retrieved.

SUPPORTING EVIDENCE
- No supporting evidence is available.

SOURCES
- None

INTERPRETATION
The question cannot be answered reliably from the
available enterprise documents.

EVIDENCE SUFFICIENCY
insufficient
""".strip()

        return {
            "question": question,
            "retrieved_chunks": [],
            "context": "",
            "answer": answer,
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
    # STEP 5: Ask Qwen using retrieved evidence
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
    # STEP 6: Return structured result
    # --------------------------------------------------------

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "answer": answer,
    }


# ============================================================
# TEST RESEARCH AGENT VERSION 2
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Research Agent Version 2"
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
        print("RESEARCH ANSWER")
        print("=" * 70)

        print(
            result["answer"]
        )

    print(
        "\nResearch Agent Version 2 "
        "testing completed."
    )