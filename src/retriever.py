
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    TOP_K,
)


def load_vectorstore():
    """Load the FAISS index using the same embedding model as ingestion."""

    index_file = VECTORSTORE_DIR / "index.faiss"
    metadata_file = VECTORSTORE_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS index is missing at {VECTORSTORE_DIR}. "
            "Run 'python -m src.ingest' first."
        )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever():
    """Create a similarity-based retriever."""

    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def retrieve_documents(question: str):
    """Retrieve relevant document chunks for a question."""

    return get_retriever().invoke(question)


if __name__ == "__main__":
    question = input("Enter your question: ")
    documents = retrieve_documents(question)

    print("\nRetrieved Documents:\n")

    for index, document in enumerate(documents, start=1):
        print(f"--- Document {index} ---")
        print(document.page_content[:500])
        print("Metadata:", document.metadata)
        print()
