from rag.embeddings import embed_text
from rag.vector_store import get_collection


# ============================================================
# RAG RETRIEVER - VERSION 1
# SEMANTIC DOCUMENT RETRIEVAL FROM CHROMADB
# ============================================================


DEFAULT_TOP_K = 3


def retrieve_chunks(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks
    for a user question.

    Workflow:

    Question
        ↓
    Question Embedding
        ↓
    ChromaDB Similarity Search
        ↓
    Top-K Relevant Chunks
    """

    if not question.strip():
        return []

    collection = get_collection()

    collection_count = collection.count()

    if collection_count == 0:

        print(
            "The ChromaDB collection is empty."
        )

        return []

    # --------------------------------------------------------
    # Generate question embedding
    # --------------------------------------------------------

    question_embedding = embed_text(
        question
    )

    # Do not request more results than exist.
    actual_top_k = min(
        top_k,
        collection_count,
    )

    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=actual_top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    ids = results.get(
        "ids",
        [[]],
    )[0]

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    # --------------------------------------------------------
    # Convert Chroma response into simple records
    # --------------------------------------------------------

    for index in range(
        len(ids)
    ):

        record = {
            "chunk_id": ids[index],
            "text": documents[index],
            "metadata": metadatas[index],
            "distance": distances[index],
        }

        retrieved_chunks.append(
            record
        )

    return retrieved_chunks


def build_retrieved_context(
    retrieved_chunks: list[dict],
) -> str:
    """
    Convert retrieved chunks into context that can
    be passed to the Research Agent.

    Source metadata is preserved so the Research Agent
    knows where each piece of evidence came from.
    """

    if not retrieved_chunks:

        return ""

    context_sections = []

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        chunk_index = metadata.get(
            "chunk_index",
            0,
        )

        text = chunk.get(
            "text",
            "",
        )

        section = f"""
SOURCE {rank}
Document: {source}
Chunk: {chunk_index + 1}

{text}
""".strip()

        context_sections.append(
            section
        )

    return "\n\n".join(
        context_sections
    )


def retrieve_context(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """
    Convenience function:

    Question
        ↓
    Retrieve chunks
        ↓
    Build Research Agent context
    """

    chunks = retrieve_chunks(
        question=question,
        top_k=top_k,
    )

    return build_retrieved_context(
        chunks
    )


# ============================================================
# TEST RETRIEVER VERSION 1
# ============================================================

if __name__ == "__main__":

    print(
        "Testing RAG Retriever Version 1"
    )

    print("=" * 70)

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

        print("\n" + "#" * 70)
        print("QUESTION")
        print("#" * 70)

        print(
            question
        )

        retrieved_chunks = retrieve_chunks(
            question=question,
            top_k=3,
        )

        print(
            f"\nChunks retrieved: "
            f"{len(retrieved_chunks)}"
        )

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):

            print("\n" + "-" * 70)

            print(
                f"Rank: {rank}"
            )

            print(
                f"Chunk ID: "
                f"{chunk['chunk_id']}"
            )

            print(
                f"Source: "
                f"{chunk['metadata'].get('source')}"
            )

            print(
                f"Distance: "
                f"{chunk['distance']}"
            )

            print(
                "\nRetrieved Text:"
            )

            print(
                chunk["text"]
            )

        context = build_retrieved_context(
            retrieved_chunks
        )

        print(
            "\nResearch Agent Context:"
        )

        print("-" * 70)

        print(
            context
        )

        print("-" * 70)

    print("\n" + "=" * 70)

    print(
        "RAG Retriever Version 1 "
        "test completed."
    )