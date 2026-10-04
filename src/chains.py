from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_MODEL
from src.retriever import get_retriever


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the
information provided in the context below.

If the answer cannot be found in the context,
clearly say:

"I could not find the answer in the provided documents."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# Retriever
# --------------------------------------------------

retriever = get_retriever()


def format_documents(documents):
    """
    Convert retrieved documents into a single
    context string.
    """

    formatted_documents = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        content = document.page_content

        formatted_documents.append(
            f"""
Source: {source}
Page: {page}

Content:
{content}
"""
        )

    return "\n\n".join(
        formatted_documents
    )


def retrieve_context(question: str):
    """
    Retrieve relevant documents and format them.
    """

    documents = retriever.invoke(
        question
    )

    context = format_documents(
        documents
    )

    return documents, context


def generate_answer(
    question: str,
    context: str,
):
    """
    Generate an answer using the LLM.
    """

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return answer