import streamlit as st

from src.graph import ask_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="centered",
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title(
    "🤖 RAG Document Q&A Chatbot"
)

st.write(
    "Ask questions about the documents "
    "available in the knowledge base."
)


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# User input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # User message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # Assistant response

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                result = ask_question(
                    question
                )

                answer = result[
                    "answer"
                ]

                st.markdown(
                    answer
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:

                error_message = (
                    f"Error: {error}"
                )

                st.error(
                    error_message
                )