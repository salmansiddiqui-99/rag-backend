"""API endpoints for selected text mode (text-only queries without Qdrant retrieval)"""
import logging
from typing import Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.models.rag import (
    SelectedTextRequest, SelectedTextResponse, RetrievedChunkData,
    RetrievalMode, ResponseStatus, RAGResponse, RAGResponseData
)
from src.services.rag_service import RAGService
from src.services.chatbot_service import ChatbotService
from src.services.response_verifier import ResponseVerifier
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/selected-text", tags=["selected-text"])


from src.database import get_db


@router.post("/query", response_model=SelectedTextResponse)
async def query_selected_text(
    request: SelectedTextRequest,
    db: Session = Depends(get_db)
) -> SelectedTextResponse:
    """
    Query the chatbot using only selected text as context.

    This endpoint bypasses Qdrant retrieval and uses only the provided selected_text
    as context for the LLM. This constrains answers to only what's in the selection.

    Request:
    - query_text: User's question (10-500 chars)
    - selected_text: Highlighted text from the page (20+ chars)

    Response:
    - success: bool indicating if response was generated
    - response_text: Generated response
    - used_selection: Whether selection was used
    - timestamp: Query timestamp

    Returns:
        SelectedTextResponse with response and metadata
    """
    try:
        # Validate request
        if not request.query or len(request.query) < 10:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 10 characters"
            )

        if not request.selected_text or len(request.selected_text) < 20:
            raise HTTPException(
                status_code=400,
                detail="Selected text must be at least 20 characters"
            )

        logger.info(
            f"Processing selected-text query: {request.query[:50]}... "
            f"(selection: {len(request.selected_text)} chars)"
        )

        # Initialize services
        rag_service = RAGService(db)
        chatbot_service = ChatbotService()
        verifier = ResponseVerifier()

        # Step 1: Create a synthetic chunk from selected text
        selected_chunk = RetrievedChunkData(
            chunk_id=None,
            text=request.selected_text,
            section_title="Selected Text",
            similarity_score=1.0,
            chapter_id=None
        )

        # Step 2: Check if context is sufficient (at least some text provided)
        if not selected_chunk.text.strip():
            logger.warning("Selected text is empty")
            return SelectedTextResponse(
                success=False,
                response_text="Selected text is empty. Please select text from the page.",
                used_selection=False,
                timestamp=datetime.utcnow()
            )

        # Step 3: Generate response using only selected text
        def response_generator():
            """Generate response using selected text only"""
            full_response = ""
            try:
                for token in chatbot_service.generate_response(
                    query_text=request.query,
                    chunks=[selected_chunk],
                    stream=True
                ):
                    full_response += token
                    yield token

                # Verify response is grounded in selection
                verification = verifier.verify_context_only(
                    response_text=full_response,
                    chunks=[selected_chunk]
                )

                logger.info(
                    f"Selected-text query processed. "
                    f"Response: {len(full_response)} chars, "
                    f"Verified: {verification.get('verified', False)}"
                )

            except Exception as e:
                logger.error(f"Error during selected-text response generation: {e}")
                raise

        # For non-streaming response (full response at once)
        full_response = ""
        try:
            for token in chatbot_service.generate_response(
                query_text=request.query,
                chunks=[selected_chunk],
                stream=False
            ):
                full_response += token

            return SelectedTextResponse(
                success=True,
                response_text=full_response,
                used_selection=True,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Selected-text query failed: {e}", exc_info=True)
            return SelectedTextResponse(
                success=False,
                response_text=f"Error: {str(e)}",
                used_selection=False,
                timestamp=datetime.utcnow()
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Selected-text endpoint error: {e}", exc_info=True)
        return SelectedTextResponse(
            success=False,
            response_text=f"Query processing failed: {str(e)}",
            used_selection=False,
            timestamp=datetime.utcnow()
        )


@router.post("/validate")
async def validate_selection(
    request: SelectedTextRequest
) -> dict:
    """
    Validate a text selection without generating a response.

    Useful for checking if selection meets minimum requirements before querying.

    Request:
    - selected_text: Text to validate (20+ chars required)

    Returns:
        Validation result with feedback
    """
    try:
        if not request.selected_text:
            return {
                "valid": False,
                "message": "Selected text is empty"
            }

        if len(request.selected_text) < 20:
            return {
                "valid": False,
                "message": f"Selected text too short ({len(request.selected_text)} chars, need 20+)"
            }

        return {
            "valid": True,
            "message": f"Selection valid ({len(request.selected_text)} chars)",
            "char_count": len(request.selected_text),
            "word_count": len(request.selected_text.split())
        }

    except Exception as e:
        logger.error(f"Selection validation error: {e}")
        return {
            "valid": False,
            "message": f"Validation failed: {str(e)}"
        }
