from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    validate_config,
)


def load_vectorstore():
    """
    Load the previously created FAISS vector store.
    """

    validate_config()

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "FAISS vector store not found. "
            "Run ingest.py first."
        )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def get_retriever():
    """
    Create a similarity-based retriever.
    """

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K
        },
    )

    return retriever


def retrieve_documents(question: str):
    """
    Retrieve relevant document chunks for a question.
    """

    retriever = get_retriever()

    documents = retriever.invoke(
        question
    )

    return documents


if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )

    documents = retrieve_documents(
        question
    )

    print("\nRetrieved Documents:\n")

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"--- Document {index} ---"
        )

        print(
            document.page_content[:500]
        )

        print(
            "Metadata:",
            document.metadata
        )

        print()