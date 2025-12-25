"""API endpoints for chatbot interactions with RAG"""
import logging
from typing import Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import json

from src.models.rag import (
    RAGRequest, RAGResponse, RAGResponseData, RAGQueryCreate,
    RetrievalMode, ResponseStatus, RetrievedChunkData, ChatRequest
)
from src.services.rag_service import RAGService
from src.services.chatbot_service import ChatbotService
from src.services.response_verifier import ResponseVerifier
from src.config import settings
from sqlalchemy.orm import Session

from src.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])


@router.post("/query", response_model=RAGResponse)
async def query_chatbot(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> RAGResponse:
    """
    Submit a question to the RAG chatbot.

    The chatbot retrieves relevant context and generates a response using only that context.

    Request:
    - query: User's question (1-8191 chars)
    - mode: 'global' (search entire book) or 'selected_text' (use provided text only)
    - selected_text: (Optional) Highlighted text for text-selection mode

    Response:
    - success: bool indicating if response was generated
    - data: RAGResponseData with response and metadata
    - error: Error message if failed
    - timestamp: Query timestamp

    Returns:
        RAGResponse with streaming or full response
    """
    try:
        # Validate request
        if not request.query or len(request.query) < 10:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 10 characters"
            )

        # Initialize services
        rag_service = RAGService(db)
        chatbot_service = ChatbotService()
        verifier = ResponseVerifier()

        # Determine retrieval mode based on request.mode
        if request.mode == "selected_text" and request.selected_text:
            retrieval_mode = RetrievalMode.TEXT_SELECTION
        else:
            retrieval_mode = RetrievalMode.GLOBAL

        logger.info(
            f"Processing query: {request.query[:50]}... "
            f"(mode={retrieval_mode.value})"
        )

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = rag_service.retrieve_chunks(
            query_text=request.query,
            retrieval_mode=retrieval_mode,
            chapter_id=None,
            selected_text=request.selected_text if request.mode == "selected_text" else None,
            top_k=settings.RAG_TOP_K
        )

        # Step 2: Check if context is sufficient (use config threshold)
        is_sufficient, reason = chatbot_service.check_context_sufficiency(
            retrieved_chunks,
            min_similarity=settings.RAG_SIMILARITY_THRESHOLD
        )

        if not is_sufficient:
            logger.warning(f"Insufficient context: {reason}")
            response_status = ResponseStatus.NO_CONTEXT

            # Log query
            query_id = rag_service.log_rag_query(
                query_text=request.query,
                retrieval_mode=retrieval_mode,
                retrieved_chunks=retrieved_chunks,
                response_status=response_status,
                chapter_id=None,
                selected_text=request.selected_text if request.mode == "selected_text" else None
            )

            return RAGResponse(
                success=False,
                data=RAGResponseData(
                    query_id=query_id,
                    query_text=request.query,
                    retrieval_mode=retrieval_mode,
                    response_status=response_status,
                    retrieved_chunks=retrieved_chunks,
                    timestamp=datetime.utcnow()
                ),
                error=f"Cannot answer: {reason}",
                timestamp=datetime.utcnow()
            )

        # Step 3: Generate response (streaming)
        def response_generator():
            """Generate response tokens and stream them"""
            full_response = ""
            try:
                for token in chatbot_service.generate_response(
                    query_text=request.query,
                    chunks=retrieved_chunks,
                    stream=True
                ):
                    full_response += token
                    # Stream as JSON chunks
                    yield json.dumps({
                        "type": "token",
                        "data": token
                    }) + "\n"

                # T057: Verify response grounding in context
                grounding_verification = chatbot_service.verify_grounding_in_context(
                    response_text=full_response,
                    chunks=retrieved_chunks
                )

                # T056: Filter low-confidence responses with fallback
                final_response, is_confident = chatbot_service.filter_low_confidence_responses(
                    response_text=full_response,
                    chunks=retrieved_chunks,
                    confidence_threshold=0.5
                )

                # T058: If confidence too low, use fallback response
                if not is_confident:
                    logger.warning("Response filtered due to low confidence - using fallback")
                    final_response = "I cannot answer this based on the available content."
                    response_status = ResponseStatus.NO_CONTEXT
                else:
                    response_status = ResponseStatus.SUCCESS

                # Log query with final response
                query_id = rag_service.log_rag_query(
                    query_text=request.query,
                    retrieval_mode=retrieval_mode,
                    retrieved_chunks=retrieved_chunks,
                    response_status=response_status,
                    chapter_id=None,
                    selected_text=request.selected_text if request.mode == "selected_text" else None
                )

                # Send final metadata with all verification results
                yield json.dumps({
                    "type": "metadata",
                    "data": {
                        "query_id": str(query_id),
                        "chunks_used": len(retrieved_chunks),
                        "total_tokens": chatbot_service.count_tokens(full_response),
                        "grounding_verification": grounding_verification,
                        "is_confident": is_confident,
                        "final_response_filtered": not is_confident
                    }
                }) + "\n"

            except Exception as e:
                logger.error(f"Error during response generation: {e}")
                yield json.dumps({
                    "type": "error",
                    "data": str(e)
                }) + "\n"

        return StreamingResponse(
            response_generator(),
            media_type="application/x-ndjson"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chatbot query failed: {e}", exc_info=True)
        return RAGResponse(
            success=False,
            error=f"Query processing failed: {str(e)}",
            timestamp=datetime.utcnow()
        )


@router.get("/modes", response_model=list[str])
async def get_retrieval_modes() -> list[str]:
    """
    Get available retrieval modes.

    Returns:
        List of available retrieval modes
    """
    return [mode.value for mode in RetrievalMode]


@router.get("/stats", response_model=dict)
async def get_rag_stats(db: Session = Depends(get_db)) -> dict:
    """
    Get statistics about RAG system and indexed content.

    Returns:
        Dictionary with system statistics
    """
    try:
        rag_service = RAGService(db)
        return rag_service.get_retrieval_stats()
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
