from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)


class GeminiEmbeddings:
    """Generate document and query embeddings using Gemini."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def embed_documents(self, texts):
        """Create embeddings for document chunks."""

        if not texts:
            return []

        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

        return [embedding.values for embedding in result.embeddings]

    def embed_query(self, text):
        """Create an embedding for a user query."""

        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

        return result.embeddings[0].values