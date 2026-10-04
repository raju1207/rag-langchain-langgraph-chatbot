from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    validate_config,
)


def load_documents():
    """
    Load all PDF documents from the documents directory.
    """

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {DOCUMENTS_DIR}"
        )

    all_documents = []

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(
            str(pdf_file)
        )

        documents = loader.load()

        all_documents.extend(documents)

    print(
        f"Loaded {len(all_documents)} document pages."
    )

    return all_documents


def split_documents(documents):
    """
    Split documents into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


def create_vectorstore(chunks):
    """
    Create embeddings and store them in FAISS.
    """

    print("Creating embeddings...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    print(
        f"FAISS index saved to: {VECTORSTORE_DIR}"
    )

    return vectorstore


def main():

    print("\n==============================")
    print("RAG DOCUMENT INGESTION")
    print("==============================\n")

    validate_config()

    documents = load_documents()

    chunks = split_documents(
        documents
    )

    create_vectorstore(
        chunks
    )

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()