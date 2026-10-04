import os
from pathlib import Path

from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


# --------------------------------------------------
# API configuration
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)


# --------------------------------------------------
# RAG configuration
# --------------------------------------------------

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)

TOP_K = int(
    os.getenv("TOP_K", "4")
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_config():
    """
    Check whether the required configuration exists.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Please add it to your .env file."
        )

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VECTORSTORE_DIR.parent.mkdir(
        parents=True,
        exist_ok=True)