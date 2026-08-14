from pathlib import Path

import chromadb

from rag.chunking import chunk_all_documents
from rag.embeddings import embed_chunks


# ============================================================
# VECTOR STORE - VERSION 1
# PERSIST DOCUMENT CHUNKS IN CHROMADB
# ============================================================


CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "enterprise_documents"


def get_chroma_client():
    """
    Create a persistent ChromaDB client.

    The database will be stored locally under:
    chroma_db/
    """

    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    return client


def get_collection():
    """
    Get or create the document collection.
    """

    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def clear_collection():
    """
    Remove the existing collection and recreate it.

    Useful while developing the ingestion pipeline.
    """

    client = get_chroma_client()

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def index_documents(
    reset_collection: bool = True,
) -> int:
    """
    Load documents, create chunks, generate embeddings,
    and store everything in ChromaDB.

    Returns the number of indexed chunks.
    """

    print(
        "Loading and chunking documents..."
    )

    chunks = chunk_all_documents()

    print(
        f"Chunks found: {len(chunks)}"
    )

    if not chunks:
        print(
            "No document chunks available to index."
        )
        return 0

    print(
        "Generating embeddings..."
    )

    embedded_chunks = embed_chunks(
        chunks
    )

    if reset_collection:

        collection = clear_collection()

    else:

        collection = get_collection()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for record in embedded_chunks:

        ids.append(
            record["chunk_id"]
        )

        documents.append(
            record["text"]
        )

        embeddings.append(
            record["embedding"]
        )

        metadata = {
            "source": record["source"],
            "path": record["path"],
            "chunk_index": record["chunk_index"],
            "total_chunks": record["total_chunks"],
        }

        metadatas.append(
            metadata
        )

    print(
        "Writing chunks to ChromaDB..."
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    count = collection.count()

    print(
        f"Collection now contains "
        f"{count} chunks."
    )

    return count


def get_collection_count() -> int:
    """
    Return the number of chunks currently stored.
    """

    collection = get_collection()

    return collection.count()


# ============================================================
# TEST VECTOR STORE VERSION 1
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Vector Store Version 1"
    )

    print("=" * 70)

    indexed_count = index_documents(
        reset_collection=True
    )

    print(
        f"\nIndexed chunks: "
        f"{indexed_count}"
    )

    stored_count = get_collection_count()

    print(
        f"Stored ChromaDB chunks: "
        f"{stored_count}"
    )

    collection = get_collection()

    sample = collection.get(
        limit=3
    )

    print("\nStored IDs:")

    for item_id in sample.get(
        "ids",
        [],
    ):
        print(
            f"  {item_id}"
        )

    print("\nStored Documents:")

    for document in sample.get(
        "documents",
        [],
    ):
        preview = (
            document
            .replace("\n", " ")
            [:300]
        )

        print(
            f"  {preview}"
        )

    print("\n" + "=" * 70)

    print(
        "Vector Store Version 1 "
        "test completed."
    )