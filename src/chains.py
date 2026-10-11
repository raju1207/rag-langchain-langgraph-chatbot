
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import get_retriever


# Local LLM: no OpenAI API credits required
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# RAG prompt
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, clearly say:
"I could not find the answer in the provided documents."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# Load the existing FAISS retriever
retriever = get_retriever()


def format_documents(documents):
    """Format retrieved document chunks as context."""

    formatted_documents = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page_label",
                                     document.metadata.get("page", "Unknown"))

        formatted_documents.append(
            f"Source: {source}\n"
            f"Page: {page}\n\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n".join(formatted_documents)


def retrieve_context(question: str):
    """Retrieve relevant document chunks."""

    documents = retriever.invoke(question)
    context = format_documents(documents)

    return documents, context


def generate_answer(question: str, context: str):
    """Generate an answer using the local Ollama model."""

    chain = RAG_PROMPT | llm | StrOutputParser()

    return chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )