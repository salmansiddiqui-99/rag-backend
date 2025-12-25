"""
Embedding Service - Creates and manages vector embeddings for content chunks.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages Cohere embeddings and Qdrant vector storage."""

    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        if settings.COHERE_API_KEY:
            self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        else:
            self.cohere_client = None
        self.embedding_model = "embed-english-v3.0"  # Cohere's embedding model
        self.embedding_dims = 1024  # Cohere embeddings are 1024-dimensional
        self.collection_name = settings.QDRANT_COLLECTION  # Use the configured collection name

    def create_or_update_collection(self) -> bool:
        """
        Create Qdrant collection if it doesn't exist.

        Returns:
            True if created or already exists, False on error
        """
        try:
            # Check if collection exists
            try:
                self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} already exists")
                return True
            except Exception:
                pass

            # Create collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dims, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Cohere.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1024 dims for Cohere embed-english-v3.0)

        Raises:
            Exception: If embedding fails
        """
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not initialized. COHERE_API_KEY not set.")

            response = self.cohere_client.embed(
                model=self.embedding_model,
                input_type="search_document",
                texts=[text]
            )
            # Extract embedding from response object (handle ClientV2 response format)
            embeddings_list = response.embeddings.float if hasattr(response.embeddings, 'float') else response.embeddings
            return embeddings_list[0]
        except Exception as e:
            logger.error(f"Embedding failed for text: {str(e)}")
            raise

    async def embed_text_async(self, text: str) -> List[float]:
        """
        Async version of embed_text using Cohere.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not initialized. COHERE_API_KEY not set.")

            # Cohere SDK doesn't have async API, so we call sync version
            response = self.cohere_client.embed(
                model=self.embedding_model,
                input_type="search_document",
                texts=[text]
            )
            # Extract embedding from response object (handle ClientV2 response format)
            embeddings_list = response.embeddings.float if hasattr(response.embeddings, 'float') else response.embeddings
            return embeddings_list[0]
        except Exception as e:
            logger.error(f"Async embedding failed: {str(e)}")
            raise

    def embed_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Embed multiple chunks with batching for efficiency.

        Args:
            chunks: List of chunk dicts from chunking service
            batch_size: Number of chunks per API call

        Returns:
            List of chunks with added 'embedding' key
        """
        embedded_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logger.info(f"Embedding batch {i//batch_size + 1} ({len(batch)} chunks)")

            # Prepare texts for embedding
            texts = [chunk["text"] for chunk in batch]

            try:
                if not self.cohere_client:
                    raise ValueError("Cohere client not initialized. COHERE_API_KEY not set.")

                # Batch embed using Cohere API
                texts = [chunk["text"] for chunk in batch]
                response = self.cohere_client.embed(
                    model=self.embedding_model,
                    input_type="search_document",
                    texts=texts
                )

                # Extract embeddings from response object (handle ClientV2 response format)
                embeddings_list = response.embeddings.float if hasattr(response.embeddings, 'float') else response.embeddings

                # Add embeddings to chunks
                for chunk, embedding in zip(batch, embeddings_list):
                    chunk["embedding"] = embedding
                    embedded_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Batch embedding failed: {str(e)}")
                # Continue with remaining batches
                continue

        logger.info(f"Successfully embedded {len(embedded_chunks)}/{len(chunks)} chunks")
        return embedded_chunks

    def upsert_to_qdrant(
        self,
        embedded_chunks: List[Dict[str, Any]],
    ) -> bool:
        """
        Insert or update chunks in Qdrant vector store.

        Args:
            embedded_chunks: List of chunks with embeddings

        Returns:
            True if successful, False otherwise
        """
        if not embedded_chunks:
            logger.warning("No chunks to upsert")
            return False

        try:
            # Ensure collection exists
            self.create_or_update_collection()

            # Prepare points for Qdrant
            points = []
            for i, chunk in enumerate(embedded_chunks):
                # Create unique ID from chapter_id + position
                point_id = hash(f"{chunk['chapter_id']}_{chunk['position']}") % (10 ** 10)

                point = PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload={
                        "chapter_id": chunk["chapter_id"],
                        "section_title": chunk["section_title"],
                        "section_level": chunk["section_level"],
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "position": chunk["position"],
                    },
                )
                points.append(point)

            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Upserted {len(points)} points to Qdrant")
            return True

        except Exception as e:
            logger.error(f"Upsert to Qdrant failed: {str(e)}")
            return False

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Search Qdrant for similar chunks.

        Args:
            query_text: Query string
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of matching chunks with similarity scores
        """
        try:
            # Embed query
            query_embedding = self.embed_text(query_text)

            # Search Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=similarity_threshold,
            )

            # Format results
            matches = []
            for result in results:
                matches.append(
                    {
                        "chunk_id": result.id,
                        "similarity": result.score,
                        "text": result.payload["text"],
                        "chapter_id": result.payload["chapter_id"],
                        "section_title": result.payload["section_title"],
                        "token_count": result.payload["token_count"],
                    }
                )

            logger.info(f"Found {len(matches)} similar chunks for query (threshold: {similarity_threshold})")
            return matches

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def delete_chapter(self, chapter_id: str) -> bool:
        """
        Delete all chunks for a chapter from Qdrant.

        Args:
            chapter_id: Chapter identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "chapter_id",
                                "match": {
                                    "value": chapter_id,
                                },
                            }
                        ]
                    }
                },
            )
            logger.info(f"Deleted all chunks for chapter {chapter_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chapter chunks: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Qdrant collection.

        Returns:
            Dict with collection stats
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "point_count": collection_info.points_count,
                "vector_size": self.embedding_dims,
                "distance_metric": "cosine",
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
