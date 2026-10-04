from src.retriever import retrieve_documents


def test_retriever_returns_documents():

    question = "What is this document about?"

    documents = retrieve_documents(
        question
    )

    assert documents is not None

    assert len(documents) > 0

    for document in documents:

        assert document.page_content