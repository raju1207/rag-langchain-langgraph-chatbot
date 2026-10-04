# RAG LangChain LangGraph Chatbot

A document question-answering chatbot built using:

- Python
- LangChain
- LangGraph
- OpenAI
- FAISS
- Streamlit

## Architecture

PDF Documents
    ↓
Document Loader
    ↓
Text Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
LangGraph
    ↓
LLM
    ↓
Answer


## Project Structure

```text
rag-langchain-langgraph-chatbot/

├── data/
│   └── documents/

├── vectorstore/
│   └── faiss_index/

├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── chains.py
│   └── graph.py

├── app/
│   └── streamlit_app.py

├── tests/
│   └── test_retriever.py

├── requirements.txt
├── .env.example
├── .gitignore
└── README.md