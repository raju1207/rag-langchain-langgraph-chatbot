
from time import sleep

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents():
    """Load PDF documents from the configured documents directory."""

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {DOCUMENTS_DIR}"
        )

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} document pages.")

    return all_documents


def split_documents(documents):
    """Split documents into smaller overlapping text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        raise ValueError(
            "No text chunks were created from the PDF documents."
        )

    return chunks


def create_vectorstore(chunks):
    """Embed chunks using Ollama and save the FAISS index."""

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Total chunks: {len(chunks)}")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        num_ctx=2048,
        keep_alive="600",
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore = None
    batch_size = 1

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        print(
            f"Embedding chunk {start + 1}/{len(chunks)}"
        )

        for attempt in range(1, 4):
            try:
                batch_store = FAISS.from_documents(
                    batch,
                    embeddings,
                )

                if vectorstore is None:
                    vectorstore = batch_store
                else:
                    vectorstore.merge_from(batch_store)

                break

            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(
                        f"Failed to embed chunk {start + 1} "
                        f"after {attempt} attempts. "
                        "Check whether the Ollama server and "
                        "nomic-embed-text model are running."
                    ) from error

                print(
                    f"Attempt {attempt} failed: {error}"
                )
                print("Retrying in 3 seconds...")
                sleep(3)

    if vectorstore is None:
        raise ValueError(
            "Unable to create the FAISS vector database."
        )

    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"FAISS index saved to: {VECTORSTORE_DIR}")
    print("Created index.faiss and index.pkl.")

    return vectorstore


def main():
    """Run the complete document ingestion pipeline."""

    print("\n================================")
    print("RAG DOCUMENT INGESTION")
    print("================================\n")

    documents = load_documents()
    chunks = split_documents(documents)
    create_vectorstore(chunks)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()
