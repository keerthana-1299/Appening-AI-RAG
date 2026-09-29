import os
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# API KEYS
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# =========================================================
# PINECONE SETTINGS
# =========================================================

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)


# =========================================================
# EMBEDDING SETTINGS
# =========================================================

# Gemini is used for creating embeddings
EMBEDDING_MODEL = "gemini-embedding-001"

# Pinecone vector dimension
EMBEDDING_DIMENSION = 1536


# =========================================================
# LLM SETTINGS
# =========================================================

# Groq is used for generating the final answer
LLM_MODEL = "openai/gpt-oss-20b"


# =========================================================
# RAG SETTINGS
# =========================================================

# Size of each PDF chunk
CHUNK_SIZE = 800

# Overlap between chunks
CHUNK_OVERLAP = 100

# Number of chunks retrieved from Pinecone
TOP_K = 4