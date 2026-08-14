from rag.document_loader import load_all_documents


# ============================================================
# DOCUMENT CHUNKING - VERSION 1
# OVERLAPPING CHARACTER-BASED CHUNKING
# ============================================================


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """
    Normalize unnecessary whitespace while preserving
    readable document text.
    """

    lines = []

    for line in text.splitlines():

        cleaned_line = " ".join(
            line.split()
        )

        if cleaned_line:
            lines.append(cleaned_line)

    return "\n".join(lines)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Example:

    Chunk 1:
    characters 0 -> 800

    Chunk 2:
    characters 650 -> 1450

    This preserves some context between neighboring chunks.
    """

    if chunk_size <= 0:

        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:

        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:

        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size."
        )

    text = clean_text(
        text
    )

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(
        text
    )

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        # We reached the end of the document.
        if end >= text_length:
            break

        start = (
            end - chunk_overlap
        )

    return chunks


def chunk_document(
    document: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split one loaded document into chunks and attach
    metadata to every chunk.
    """

    source = document.get(
        "source",
        "unknown",
    )

    path = document.get(
        "path",
        "",
    )

    text = document.get(
        "text",
        "",
    )

    text_chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunk_records = []

    total_chunks = len(
        text_chunks
    )

    for index, chunk in enumerate(
        text_chunks
    ):

        chunk_id = (
            f"{source}::chunk_{index + 1}"
        )

        chunk_record = {
            "chunk_id": chunk_id,
            "source": source,
            "path": path,
            "chunk_index": index,
            "total_chunks": total_chunks,
            "text": chunk,
        }

        chunk_records.append(
            chunk_record
        )

    return chunk_records


def chunk_all_documents(
    documents: list[dict] | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunk all loaded documents.

    If documents are not supplied, load them
    automatically from data/documents.
    """

    if documents is None:

        documents = (
            load_all_documents()
        )

    all_chunks = []

    for document in documents:

        document_chunks = (
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

        all_chunks.extend(
            document_chunks
        )

    return all_chunks


# ============================================================
# TEST DOCUMENT CHUNKING VERSION 1
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Document Chunking Version 1"
    )

    print("=" * 70)

    documents = (
        load_all_documents()
    )

    print(
        f"\nDocuments loaded: "
        f"{len(documents)}"
    )

    chunks = (
        chunk_all_documents(
            documents=documents
        )
    )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    for chunk in chunks:

        print("\n" + "-" * 70)

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Source: "
            f"{chunk['source']}"
        )

        print(
            f"Chunk Number: "
            f"{chunk['chunk_index'] + 1}"
            f"/{chunk['total_chunks']}"
        )

        print(
            f"Characters: "
            f"{len(chunk['text'])}"
        )

        print("\nChunk Text:")
        print(
            chunk["text"]
        )

    print("\n" + "=" * 70)

    print(
        "Document Chunking Version 1 "
        "test completed."
    )