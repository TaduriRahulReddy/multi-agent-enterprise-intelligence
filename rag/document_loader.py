from pathlib import Path

from pypdf import PdfReader


# ============================================================
# DOCUMENT LOADER - VERSION 1
# LOAD PDF AND TEXT DOCUMENTS
# ============================================================


DOCUMENTS_DIR = Path("data/documents")


def load_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file.
    """

    reader = PdfReader(str(file_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

    return "\n".join(pages)


def load_text_file(file_path: Path) -> str:
    """
    Load a plain-text or markdown file.
    """

    return file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def load_document(file_path: Path) -> dict:
    """
    Load one supported document and return
    its metadata and extracted text.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        text = load_pdf(file_path)

    elif suffix in [
        ".txt",
        ".md",
    ]:
        text = load_text_file(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    return {
        "source": file_path.name,
        "path": str(file_path),
        "text": text,
    }


def load_all_documents(
    directory: Path = DOCUMENTS_DIR,
) -> list[dict]:
    """
    Load every supported document from the
    documents directory.
    """

    supported_extensions = {
        ".pdf",
        ".txt",
        ".md",
    }

    documents = []

    if not directory.exists():

        raise FileNotFoundError(
            f"Documents directory does not exist: {directory}"
        )

    for file_path in sorted(
        directory.iterdir()
    ):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in supported_extensions
        ):
            continue

        document = load_document(
            file_path
        )

        documents.append(
            document
        )

    return documents


# ============================================================
# TEST DOCUMENT LOADER
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Document Loader Version 1"
    )

    print("=" * 70)

    documents = load_all_documents()

    print(
        f"\nDocuments loaded: {len(documents)}"
    )

    for document in documents:

        print("\n" + "-" * 70)

        print(
            f"Source: {document['source']}"
        )

        print(
            f"Characters extracted: "
            f"{len(document['text']):,}"
        )

        preview = (
            document["text"]
            .replace("\n", " ")
            [:500]
        )

        print("\nPreview:")
        print(preview)

    print("\n" + "=" * 70)

    print(
        "Document Loader Version 1 "
        "test completed."
    )