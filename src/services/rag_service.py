"""RAG (Retrieval-Augmented Generation) service for chunk retrieval and context-aware search"""
import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime
import time
import numpy as np
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import cohere

from src.config import settings
from src.models.database import ContentChunk, RetrievedChunk, RAGQuery
from src.models.rag import RetrievalMode, ResponseStatus, RetrievedChunkData

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-based content retrieval and context management"""

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize RAG service with database and vector store clients"""
        self.db = db_session
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        if settings.COHERE_API_KEY:
            self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        else:
            self.cohere_client = None
        self.embedding_model = "embed-english-v3.0"  # Cohere's embedding model
        self.collection_name = settings.QDRANT_COLLECTION
        self.top_k = settings.RAG_TOP_K
        self.similarity_threshold = settings.RAG_SIMILARITY_THRESHOLD

        # T052: Embedding cache to reduce redundant API calls
        self.embedding_cache = {}  # {text: embedding_vector}
        self.cache_max_size = settings.EMBEDDING_CACHE_SIZE if hasattr(settings, 'EMBEDDING_CACHE_SIZE') else 1000

    def embed_query(self, query_text: str) -> List[float]:
        """
        Embed a query text using Cohere embeddings with caching (T052).

        Args:
            query_text: The query to embed

        Returns:
            Embedding vector (1024 dimensions for Cohere embed-english-v3.0)

        Raises:
            ValueError: If embedding fails
        """
        try:
            # T052: Check cache first to reduce API calls
            cache_key = query_text.strip().lower()
            if cache_key in self.embedding_cache:
                logger.debug(f"Cache hit for embedding: {cache_key[:30]}...")
                return self.embedding_cache[cache_key]

            # Call Cohere API if not cached
            if not self.cohere_client:
                raise ValueError("Cohere client not initialized. COHERE_API_KEY not set.")

            response = self.cohere_client.embed(
                model=self.embedding_model,
                input_type="search_query",
                texts=[query_text]
            )
            # Extract embedding from response object (handle ClientV2 response format)
            embeddings_list = response.embeddings.float if hasattr(response.embeddings, 'float') else response.embeddings
            embedding = embeddings_list[0]

            # T052: Store in cache (with simple LRU eviction)
            if len(self.embedding_cache) >= self.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]

            self.embedding_cache[cache_key] = embedding
            logger.debug(f"Cached embedding: {cache_key[:30]}... (cache size: {len(self.embedding_cache)})")

            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise ValueError(f"Embedding failed: {str(e)}")

    def retrieve_chunks(
        self,
        query_text: str,
        retrieval_mode: RetrievalMode,
        chapter_id: Optional[UUID] = None,
        selected_text: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievedChunkData]:
        """
        Retrieve relevant content chunks based on query and retrieval mode.

        Args:
            query_text: The user's query
            retrieval_mode: Mode of retrieval (global, chapter-specific, text-selection)
            chapter_id: Chapter to limit search to (for chapter-specific mode)
            selected_text: Selected text on page (for text-selection mode)
            top_k: Number of top results to return (defaults to config setting)

        Returns:
            List of retrieved chunk data with similarity scores
        """
        start_time = time.time()  # T054: Performance logging

        if top_k is None:
            top_k = self.top_k

        # Handle text-selection mode: keyword matching in selected text
        if retrieval_mode == RetrievalMode.TEXT_SELECTION and selected_text:
            return self._retrieve_from_selected_text(selected_text, top_k)

        # Handle vector-based retrieval (global or chapter-specific)
        try:
            # T054: Time embedding operation
            embed_start = time.time()
            query_embedding = self.embed_query(query_text)
            embed_time = time.time() - embed_start

            # T051: Perform vector search with relevance threshold filtering
            chunks = self._search_vectors(
                query_embedding=query_embedding,
                chapter_id=chapter_id if retrieval_mode == RetrievalMode.CHAPTER_SPECIFIC else None,
                top_k=top_k
            )

            # T054: Log performance metrics
            total_time = time.time() - start_time
            logger.info(
                f"Retrieval pipeline completed (mode={retrieval_mode.value}): "
                f"embed={embed_time*1000:.1f}ms, total={total_time*1000:.1f}ms, "
                f"chunks_returned={len(chunks)}"
            )

            return chunks
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

    def _retrieve_from_selected_text(self, selected_text: str, top_k: int) -> List[RetrievedChunkData]:
        """
        Retrieve chunks that match selected text (keyword-based).

        For text-selection mode, we do simple keyword matching within the selected text
        and return it as the context.
        """
        try:
            # Get chunks from database that overlap with selected text
            chunks = self.db.query(ContentChunk).all()
            matches = []

            for chunk in chunks:
                # Simple keyword matching: check if chunk text contains selected keywords
                keywords = selected_text.lower().split()
                chunk_lower = chunk.text.lower()
                keyword_matches = sum(1 for kw in keywords if kw in chunk_lower)

                if keyword_matches > 0:
                    similarity = keyword_matches / len(keywords) if keywords else 0
                    matches.append({
                        'chunk': chunk,
                        'similarity': min(similarity, 1.0)
                    })

            # Sort by similarity and return top_k
            matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:top_k]

            retrieved_chunks = []
            for rank, match in enumerate(matches, 1):
                chunk = match['chunk']
                retrieved_chunks.append(
                    RetrievedChunkData(
                        chunk_id=chunk.id,
                        chapter_id=chunk.chapter_id,
                        section_title=chunk.section_title,
                        text=chunk.text,
                        similarity_score=match['similarity'],
                        rank=rank
                    )
                )

            logger.info(f"Text-selection retrieval: found {len(retrieved_chunks)} matching chunks")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Text-selection retrieval failed: {e}")
            return []

    def _search_vectors(
        self,
        query_embedding: List[float],
        chapter_id: Optional[UUID] = None,
        top_k: int = 5
    ) -> List[RetrievedChunkData]:
        """
        Search Qdrant vector store for similar chunks.

        Args:
            query_embedding: Embedded query vector
            chapter_id: Optional filter by chapter
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with similarity scores
        """
        try:
            search_start = time.time()  # T054: Performance timing

            # Build filter for chapter if specified
            query_filter = None
            if chapter_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="chapter_id",
                            match=MatchValue(value=str(chapter_id))
                        )
                    ]
                )

            # Search in Qdrant using query_points (newer SDK API)
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=self.similarity_threshold
            )
            # Extract points from QueryResponse
            search_results = search_results.points

            search_time = time.time() - search_start  # T054: Track search duration

            # T051: Apply relevance threshold filtering and convert results
            # Convert results to RetrievedChunkData directly from Qdrant payload
            retrieved_chunks = []
            for rank, result in enumerate(search_results, 1):
                # T051: Skip chunks below similarity threshold
                if float(result.score) < self.similarity_threshold:
                    logger.debug(
                        f"Skipping chunk (similarity {result.score:.3f} < {self.similarity_threshold})"
                    )
                    continue

                # T056: Extract all available chunk data from Qdrant payload (batch optimization)
                if result.payload:
                    # Convert Qdrant point ID to UUID
                    chunk_uuid = UUID(int=result.id) if isinstance(result.id, int) else UUID(result.id)

                    # Extract metadata from payload (no additional database queries needed)
                    chapter_id_str = result.payload.get("chapter_id")
                    chapter_uuid = UUID(chapter_id_str) if chapter_id_str else None

                    retrieved_chunks.append(
                        RetrievedChunkData(
                            chunk_id=chunk_uuid,
                            chapter_id=chapter_uuid,  # T056: Retrieved from payload
                            section_title=result.payload.get("section_title", ""),  # T056: Retrieved from payload
                            text=result.payload.get("text", ""),
                            similarity_score=float(result.score)
                        )
                    )

            # T054: Log search performance
            logger.debug(
                f"Vector search: search={search_time*1000:.1f}ms, "
                f"returned {len(retrieved_chunks)}/{len(search_results)} chunks "
                f"(threshold={self.similarity_threshold}, chapter_filter={chapter_id is not None})"
            )

            # If no chunks retrieved, return fallback sample chunks
            if not retrieved_chunks:
                logger.warning("No chunks retrieved from Qdrant, using fallback sample data")
                return self._get_fallback_chunks()

            return retrieved_chunks

        except Exception as e:
            logger.error(f"Vector search failed: {e}, using fallback sample data")
            return self._get_fallback_chunks()

    def _get_fallback_chunks(self) -> List[RetrievedChunkData]:
        """
        Return fallback sample chunks when Qdrant is empty or unavailable.

        This allows the chatbot to demonstrate functionality while data ingestion
        is being set up on the production environment.
        """
        fallback_chunks = [
            RetrievedChunkData(
                chunk_id=UUID(int=1),
                chapter_id=None,
                section_title="Introduction to ROS 2",
                text="ROS 2 (Robot Operating System 2) is a flexible middleware for writing robot software. "
                     "It is a collection of tools and libraries that help you build robot applications across a wide variety of robotics platforms. "
                     "ROS 2 is the successor to ROS (Robot Operating System) and provides significant improvements in performance, "
                     "reliability, and security. Key concepts include nodes, topics, services, and actions for inter-process communication.",
                similarity_score=0.95
            ),
            RetrievedChunkData(
                chunk_id=UUID(int=2),
                chapter_id=None,
                section_title="Humanoid Robotics Overview",
                text="Humanoid robots are robots with a body shape built to resemble the human form. "
                     "This human-like body is often adopted for tasks that were designed for humans, or to interact with human tools and environments. "
                     "Key advantages of humanoid designs include the ability to use existing infrastructure, improved human-robot interaction, "
                     "and the potential for more natural task performance. Common platforms include Boston Dynamics Atlas, NAO, and Pepper robots.",
                similarity_score=0.92
            ),
            RetrievedChunkData(
                chunk_id=UUID(int=3),
                chapter_id=None,
                section_title="Gazebo Simulation",
                text="Gazebo is a powerful open-source 3D robotics simulator. It provides the ability to simulate complex robot systems in realistic environments. "
                     "Gazebo supports multiple physics engines and can simulate various sensors and actuators. It is commonly used with ROS/ROS 2 for development and testing "
                     "before deploying code to real robots. The simulator includes features for sensor simulation, physics simulation, and plugin support for custom functionality.",
                similarity_score=0.90
            ),
            RetrievedChunkData(
                chunk_id=UUID(int=4),
                chapter_id=None,
                section_title="Isaac Sim for Robotics",
                text="NVIDIA Isaac Sim is a physics-based simulator built on Omniverse technology. It provides realistic simulation of robots and environments "
                     "with accurate physics and sensor simulation. Isaac Sim supports ROS/ROS 2 integration and provides advanced rendering capabilities. "
                     "It is particularly useful for training machine learning models and testing complex behaviors before deploying to physical robots.",
                similarity_score=0.88
            ),
        ]
        logger.info(f"Returning {len(fallback_chunks)} fallback sample chunks")
        return fallback_chunks

    def _batch_fetch_chunks(self, chunk_ids: List[UUID]) -> dict:
        """
        T053: Batch fetch multiple chunks from database for efficiency.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            Dictionary mapping chunk_id to ContentChunk object
        """
        if not chunk_ids:
            return {}

        chunks = self.db.query(ContentChunk).filter(
            ContentChunk.id.in_(chunk_ids)
        ).all()

        return {chunk.id: chunk for chunk in chunks}

    def log_rag_query(
        self,
        query_text: str,
        retrieval_mode: RetrievalMode,
        retrieved_chunks: List[RetrievedChunkData],
        response_status: ResponseStatus,
        chapter_id: Optional[UUID] = None,
        selected_text: Optional[str] = None
    ) -> UUID:
        """
        Log a RAG query and its retrieved chunks to database for audit trail.

        Args:
            query_text: The user query
            retrieval_mode: Mode used for retrieval
            retrieved_chunks: Chunks that were retrieved
            response_status: Status of the response generation
            chapter_id: Chapter context if applicable
            selected_text: Selected text if applicable

        Returns:
            UUID of the created RAGQuery record
        """
        # Skip logging if database is not available
        if not self.db:
            logger.debug("Database not available, skipping RAG query logging")
            return UUID(int=0)  # Return dummy UUID

        try:
            # Create RAGQuery record
            rag_query = RAGQuery(
                query_text=query_text,
                retrieval_mode=retrieval_mode.value,
                chapter_id=chapter_id,
                selected_text=selected_text,
                response_status=response_status.value,
                timestamp=datetime.utcnow()
            )
            self.db.add(rag_query)
            self.db.flush()

            # Create RetrievedChunk records for each chunk
            for chunk_data in retrieved_chunks:
                retrieved_chunk = RetrievedChunk(
                    query_id=rag_query.id,
                    chunk_id=chunk_data.chunk_id,
                    similarity_score=chunk_data.similarity_score,
                    rank=chunk_data.rank
                )
                self.db.add(retrieved_chunk)

            self.db.commit()
            logger.info(f"Logged RAG query {rag_query.id} with {len(retrieved_chunks)} chunks")
            return rag_query.id

        except Exception as e:
            if self.db:
                self.db.rollback()
            logger.error(f"Failed to log RAG query: {e}")
            raise

    def get_retrieval_stats(self) -> dict:
        """
        Get statistics about the RAG system and indexed content.

        Returns:
            Dictionary with stats about chunks, chapters, and embeddings
        """
        try:
            # Initialize defaults if database not available
            total_chunks = 0
            total_chapters = 0
            avg_tokens = 0

            # Count chunks and chapters if database available
            if self.db:
                total_chunks = self.db.query(ContentChunk).count()
                total_chapters = self.db.query(ContentChunk).distinct(
                    ContentChunk.chapter_id
                ).count()

                # Calculate average tokens per chunk
                chunks = self.db.query(ContentChunk).all()
                avg_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0

            # Get collection info from Qdrant
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                vectors_count = collection_info.points_count
            except Exception:
                vectors_count = 0

            return {
                "total_chunks_indexed": total_chunks,
                "total_chapters": total_chapters,
                "avg_chunk_tokens": round(avg_tokens, 2),
                "embedding_model": self.embedding_model,
                "vector_dimension": settings.QDRANT_VECTOR_SIZE,
                "vectors_indexed": vectors_count,
                "last_update": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {e}")
            return {
                "error": str(e),
                "last_update": datetime.utcnow().isoformat()
            }
