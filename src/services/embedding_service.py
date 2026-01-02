"""
Embedding service using Cohere API for text vectorization.
"""
import cohere
from typing import List
from src.config import config


class EmbeddingService:
    """Wrapper for Cohere embedding API."""

    def __init__(self):
        """Initialize Cohere client."""
        self.client = cohere.ClientV2(api_key=config.COHERE_API_KEY)
        self.model = config.COHERE_MODEL
        self.embedding_dim = config.COHERE_EMBEDDING_DIM

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text to embed

        Returns:
            1024-dimensional embedding vector

        Raises:
            Exception: If Cohere API call fails
        """
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document",
            embedding_types=["float"]
        )
        return response.embeddings.float[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts (batch processing).

        Args:
            texts: List of input texts to embed

        Returns:
            List of 1024-dimensional embedding vectors

        Raises:
            Exception: If Cohere API call fails
        """
        if not texts:
            return []

        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document",
            embedding_types=["float"]
        )
        return response.embeddings.float

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding vector for a search query.

        Args:
            query: Search query text

        Returns:
            1024-dimensional embedding vector optimized for search

        Raises:
            Exception: If Cohere API call fails
        """
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query",
            embedding_types=["float"]
        )
        return response.embeddings.float[0]


# Global singleton instance
embedding_service = EmbeddingService()
