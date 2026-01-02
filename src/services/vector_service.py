"""
Vector database service using Qdrant for semantic search.
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import config


class VectorService:
    """Wrapper for Qdrant vector database operations."""

    COLLECTION_NAME = "physical_ai_book"

    def __init__(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
        self.embedding_dim = config.COHERE_EMBEDDING_DIM

    def create_collection(self) -> None:
        """
        Create Qdrant collection for book content if it doesn't exist.

        Collection config:
        - Vector size: 1024 (Cohere embed-english-v3.0)
        - Distance: Cosine similarity
        - Index: HNSW
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.COLLECTION_NAME}")
        else:
            print(f"Qdrant collection already exists: {self.COLLECTION_NAME}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Upsert content chunks to Qdrant collection.

        Args:
            chunks: List of chunk dictionaries with fields:
                - chunk_id: Unique identifier (UUID string)
                - text: Chunk text content
                - embedding: 1024-dim vector
                - chapter_title: Chapter name
                - section_heading: Section heading (optional)
                - chunk_index: Index within document
                - source_file_path: Original file path

        Raises:
            Exception: If Qdrant upsert fails
        """
        points = []
        for chunk in chunks:
            point = PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "chapter_title": chunk["chapter_title"],
                    "section_heading": chunk.get("section_heading"),
                    "chunk_index": chunk["chunk_index"],
                    "source_file_path": chunk["source_file_path"]
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )

    def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content chunks using vector similarity.

        Args:
            query_vector: 1024-dimensional query embedding
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of dictionaries with fields:
                - text: Chunk text content
                - chapter_title: Chapter name
                - section_heading: Section heading (if available)
                - score: Similarity score (0-1, higher is better)

        Raises:
            Exception: If Qdrant search fails
        """
        search_results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points

        results = []
        for result in search_results:
            results.append({
                "text": result.payload["text"],
                "chapter_title": result.payload["chapter_title"],
                "section_heading": result.payload.get("section_heading"),
                "score": result.score,
                "source_file_path": result.payload["source_file_path"]
            })

        return results


# Global singleton instance
vector_service = VectorService()
