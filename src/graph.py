from typing import TypedDict, List

from langchain_core.documents import Document
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.chains import (
    retrieve_context,
    generate_answer,
)


# --------------------------------------------------
# Graph State
# --------------------------------------------------

class RAGState(TypedDict):

    question: str

    documents: List[Document]

    context: str

    answer: str


# --------------------------------------------------
# Retrieve Node
# --------------------------------------------------

def retrieve_node(
    state: RAGState
):

    question = state["question"]

    documents, context = retrieve_context(
        question
    )

    return {
        "documents": documents,
        "context": context,
    }


# --------------------------------------------------
# Generate Node
# --------------------------------------------------

def generate_node(
    state: RAGState
):

    question = state["question"]

    context = state["context"]

    answer = generate_answer(
        question,
        context,
    )

    return {
        "answer": answer
    }


# --------------------------------------------------
# Build Graph
# --------------------------------------------------

def build_graph():

    graph = StateGraph(
        RAGState
    )

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()


# --------------------------------------------------
# Run Graph
# --------------------------------------------------

rag_graph = build_graph()


def ask_question(
    question: str
):

    initial_state = {
        "question": question,
        "documents": [],
        "context": "",
        "answer": "",
    }

    result = rag_graph.invoke(
        initial_state
    )

    return result


if __name__ == "__main__":

    question = input(
        "Ask your question: "
    )

    result = ask_question(
        question
    )

    print("\nAnswer:")
    print(
        result["answer"]
    )