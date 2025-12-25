"""API endpoints for RAG (Retrieval-Augmented Generation) - Internal/Admin endpoints"""
import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.models.rag import RetrievedChunkData, RetrievalMode
from src.services.rag_service import RAGService
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


def get_db():
    """Database session dependency"""
    # TODO: Implement proper database session management
    pass


@router.get("/retrieve", response_model=dict)
async def retrieve_chunks(
    query: str = Query(..., min_length=10, max_length=500),
    retrieval_mode: str = Query(default="global", description="global|chapter-specific|text-selection"),
    chapter_id: Optional[UUID] = Query(None),
    top_k: int = Query(5, ge=1, le=10),
    similarity_threshold: float = Query(0.75, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
) -> dict:
    """
    Internal endpoint for retrieving chunks (T058).

    Used by the chatbot service to get relevant context.

    Query Parameters:
    - query: Search query text
    - retrieval_mode: "global", "chapter-specific", or "text-selection"
    - chapter_id: Chapter to limit search to (for chapter-specific mode)
    - top_k: Number of results to return (1-10)
    - similarity_threshold: Minimum similarity score (0-1)

    Returns:
        Dictionary with retrieved chunks and metadata
    """
    try:
        # Validate retrieval mode
        try:
            mode = RetrievalMode(retrieval_mode)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid retrieval_mode. Must be one of: {', '.join(m.value for m in RetrievalMode)}"
            )

        logger.info(f"Retrieving chunks: query={query[:30]}..., mode={retrieval_mode}")

        # Initialize RAG service
        rag_service = RAGService(db)

        # Retrieve chunks
        chunks = rag_service.retrieve_chunks(
            query_text=query,
            retrieval_mode=mode,
            chapter_id=chapter_id,
            top_k=top_k
        )

        return {
            "success": True,
            "chunks": [
                {
                    "chunk_id": str(c.chunk_id),
                    "chapter_id": str(c.chapter_id),
                    "section_title": c.section_title,
                    "text": c.text,
                    "similarity_score": round(c.similarity_score, 3),
                    "rank": c.rank
                }
                for c in chunks
            ],
            "total_found": len(chunks),
            "retrieval_mode": retrieval_mode,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )


@router.get("/stats", response_model=dict)
async def get_rag_stats(db: Session = Depends(get_db)) -> dict:
    """
    Get RAG system statistics (T059).

    Returns information about indexed content, embeddings, and storage.

    Returns:
        Dictionary with system statistics
    """
    try:
        logger.info("Fetching RAG statistics")

        rag_service = RAGService(db)
        stats = rag_service.get_retrieval_stats()

        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/health", response_model=dict)
async def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Check if RAG services are operational.

    Verifies connectivity to Qdrant and OpenAI.

    Response:
    - qdrant: "operational" | "down"
    - openai: "operational" | "down"
    - database: "operational" | "down"
    """
    try:
        rag_service = RAGService(db)

        # Check Qdrant
        qdrant_status = "down"
        try:
            rag_service.qdrant_client.get_collections()
            qdrant_status = "operational"
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")

        # Check OpenAI
        openai_status = "down"
        try:
            rag_service.embed_query("test")
            openai_status = "operational"
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")

        # Check Database
        db_status = "down"
        try:
            # Try a simple query
            from src.models.database import Chapter
            db.query(Chapter).first()
            db_status = "operational"
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")

        return {
            "qdrant": qdrant_status,
            "openai": openai_status,
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "qdrant": "unknown",
            "openai": "unknown",
            "database": "unknown",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
