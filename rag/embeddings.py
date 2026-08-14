from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_all_documents


# ============================================================
# EMBEDDING MODEL - VERSION 1
# LOCAL SENTENCE TRANSFORMER EMBEDDINGS
# ============================================================


MODEL_NAME = "all-MiniLM-L6-v2"


# Load the embedding model once
embedding_model = SentenceTransformer(
    MODEL_NAME
)


def embed_text(text: str) -> list[float]:
    """
    Convert one text string into an embedding vector.
    """

    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True,
    )

    return embedding.tolist()


def embed_chunks(
    chunks: list[dict] | None = None,
) -> list[dict]:
    """
    Generate embeddings for document chunks.

    Each returned record contains:
    - chunk metadata
    - chunk text
    - embedding vector
    """

    if chunks is None:

        chunks = chunk_all_documents()

    embedded_chunks = []

    for chunk in chunks:

        embedding = embed_text(
            chunk["text"]
        )

        embedded_record = {
            **chunk,
            "embedding": embedding,
        }

        embedded_chunks.append(
            embedded_record
        )

    return embedded_chunks


# ============================================================
# TEST EMBEDDINGS VERSION 1
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Embeddings Version 1"
    )

    print("=" * 70)

    chunks = chunk_all_documents()

    print(
        f"\nChunks loaded: {len(chunks)}"
    )

    embedded_chunks = embed_chunks(
        chunks
    )

    print(
        f"Embedded chunks: "
        f"{len(embedded_chunks)}"
    )

    for record in embedded_chunks:

        print("\n" + "-" * 70)

        print(
            f"Chunk ID: "
            f"{record['chunk_id']}"
        )

        print(
            f"Source: "
            f"{record['source']}"
        )

        print(
            f"Embedding dimensions: "
            f"{len(record['embedding'])}"
        )

        print(
            "\nFirst 10 embedding values:"
        )

        print(
            record["embedding"][:10]
        )

    print("\n" + "=" * 70)

    print(
        "Embeddings Version 1 "
        "test completed."
    )