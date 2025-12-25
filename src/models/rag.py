"""
RAG Chatbot Request and Response Schemas (Phase 2)

This module defines all Pydantic v2 models for RAG chatbot API contracts:
- Embedding service (embed requests/responses)
- Query/retrieval service (search requests/responses)
- Chatbot streaming (chat requests/responses)
- Selected text (constraint-based queries)

All models include validation, examples, and descriptions per API contracts:
- embed-contract.md
- query-contract.md
- chat-contract.md
- selected-text-contract.md
"""

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import List, Optional
from uuid import UUID
from enum import Enum
from datetime import datetime


# ============================================================================
# RETRIEVAL MODES AND RESPONSE MODELS
# ============================================================================

class RetrievalMode(str, Enum):
    """Retrieval mode enumeration"""
    GLOBAL = "global"
    CHAPTER_SPECIFIC = "chapter-specific"
    TEXT_SELECTION = "text-selection"


class ResponseStatus(str, Enum):
    """Response status enumeration"""
    SUCCESS = "success"
    NO_CONTEXT = "no-context"
    ERROR = "error"


class RetrievedChunkData(BaseModel):
    """Retrieved chunk data with similarity score"""
    chunk_id: Optional[UUID] = None
    text: str
    section_title: Optional[str] = None
    similarity_score: float = 0.0
    chapter_id: Optional[UUID] = None

    model_config = ConfigDict(from_attributes=True)


class RAGResponseData(BaseModel):
    """RAG response data"""
    query_id: Optional[UUID] = None
    query_text: str
    retrieval_mode: RetrievalMode = RetrievalMode.GLOBAL
    response_status: ResponseStatus = ResponseStatus.SUCCESS
    retrieved_chunks: List[RetrievedChunkData] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO format string"""
        return value.isoformat()


class RAGRequest(BaseModel):
    """Generic RAG request"""
    query_text: str = Field(..., min_length=10, max_length=500)
    chapter_id: Optional[UUID] = None
    selected_text: Optional[str] = None


class RAGResponse(BaseModel):
    """Generic RAG response"""
    success: bool
    data: Optional[RAGResponseData] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO format string"""
        return value.isoformat()


class RAGQueryCreate(BaseModel):
    """RAG query creation model"""
    query_text: str
    retrieval_mode: RetrievalMode
    chapter_id: Optional[UUID] = None
    selected_text: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# EMBEDDING SERVICE (POST /api/embed)
# ============================================================================

class EmbedRequest(BaseModel):
    """Request to embed text into a vector."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=8191,
        description="Text to embed (1-8191 characters)."
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model (default: text-embedding-3-small, 384 dims)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "ROS 2 is a flexible middleware for robotics.",
                "model": "text-embedding-3-small"
            }
        }
    )


class EmbedResponse(BaseModel):
    """Response containing text embedding vector."""

    embedding: List[float] = Field(..., description="Vector (384 dimensions)")
    dimensions: int = Field(default=384, description="Vector size")
    model: str = Field(default="text-embedding-3-small", description="Model used")
    tokens_used: int = Field(..., ge=0, description="Input tokens consumed")
    cost_usd: float = Field(..., ge=0.0, description="Cost in USD")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embedding": [0.001, -0.021, 0.008],
                "dimensions": 384,
                "model": "text-embedding-3-small",
                "tokens_used": 25,
                "cost_usd": 0.0000005
            }
        }
    )


# ============================================================================
# QUERY/RETRIEVAL SERVICE (POST /api/query)
# ============================================================================

class QueryRequest(BaseModel):
    """Request to retrieve relevant chunks via semantic search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=8191,
        description="Search query or user question"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max chunks to return (default: 5)"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Min similarity threshold (default: 0.5)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How do ROS 2 services work?",
                "top_k": 5,
                "threshold": 0.5
            }
        }
    )


class ChunkResult(BaseModel):
    """A single retrieved text chunk with metadata."""

    id: UUID = Field(..., description="Chunk ID (UUID)")
    chapter_id: UUID = Field(..., description="Source chapter ID")
    chapter_title: str = Field(..., description="Chapter name")
    chunk_index: int = Field(..., ge=0, description="Position in chapter")
    content: str = Field(..., description="Text segment (200-400 tokens)")
    token_count: int = Field(..., ge=100, le=500, description="Token count")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity")
    rank: int = Field(..., ge=1, description="Position in results")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440050",
                "chapter_id": "550e8400-e29b-41d4-a716-446655440001",
                "chapter_title": "Introduction to ROS 2",
                "chunk_index": 0,
                "content": "A ROS 2 service is a request-reply mechanism...",
                "token_count": 250,
                "similarity_score": 0.87,
                "rank": 1
            }
        }
    )


class QueryResponse(BaseModel):
    """Response containing retrieved chunks and metadata."""

    chunks: List[ChunkResult] = Field(..., description="Matched chunks")
    total_found: int = Field(..., ge=0, description="Number returned")
    query_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model used"
    )
    retrieval_latency_ms: int = Field(
        ...,
        ge=0,
        description="Retrieval time (target: <800ms)"
    )
    threshold_used: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold applied"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunks": [],
                "total_found": 0,
                "query_embedding_model": "text-embedding-3-small",
                "retrieval_latency_ms": 347,
                "threshold_used": 0.5
            }
        }
    )


# ============================================================================
# CHATBOT STREAMING SERVICE (POST /api/chatbot/query)
# ============================================================================

class ChatModeEnum(str, Enum):
    """Enumeration of retrieval modes for chatbot."""

    GLOBAL = "global"
    SELECTED_TEXT = "selected_text"


class ChatRequest(BaseModel):
    """Request to generate streamed chatbot response."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=8191,
        description="User question or prompt"
    )
    mode: ChatModeEnum = Field(
        default=ChatModeEnum.GLOBAL,
        description="'global' (search book) or 'selected_text'"
    )
    selected_text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=8191,
        description="Highlighted text (required if mode='selected_text')"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is a ROS 2 service?",
                "mode": "global",
                "selected_text": None
            }
        }
    )


class MetadataEvent(BaseModel):
    """Metadata event sent at start of streaming response."""

    type: str = Field(default="metadata", description="Event type")
    session_id: str = Field(..., description="Session ID")
    query_id: str = Field(..., description="Query ID")
    mode: str = Field(..., description="Retrieval mode")
    retrieved_chunk_count: int = Field(..., description="Chunks retrieved")
    llm_model: str = Field(..., description="LLM model used")


class TokenEvent(BaseModel):
    """Token event sent for each streamed token from LLM."""

    type: str = Field(default="token", description="Event type")
    token: str = Field(..., description="Single token from LLM")
    accumulated_text: str = Field(..., description="Full text so far")


class CompleteEvent(BaseModel):
    """Complete event sent at end of streaming response."""

    type: str = Field(default="complete", description="Event type")
    final_response: str = Field(..., description="Complete response")
    total_tokens: int = Field(..., ge=0, description="Total tokens")
    latency_ms: int = Field(..., ge=0, description="End-to-end latency")
    grounded: bool = Field(..., description="Is response grounded?")


class ErrorEvent(BaseModel):
    """Error event sent if error occurs during streaming."""

    type: str = Field(default="error", description="Event type")
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")


# ============================================================================
# SELECTED TEXT SERVICE (POST /api/selected-text)
# ============================================================================

class SelectedTextRequest(BaseModel):
    """Request to answer question based only on selected text."""

    selected_text: str = Field(
        ...,
        min_length=1,
        max_length=8191,
        description="User-highlighted text passage"
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=8191,
        description="Question about the selected text"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "selected_text": "ROS 2 services provide request-reply communication...",
                "query": "What are the advantages of services?"
            }
        }
    )


class SelectedTextResponse(BaseModel):
    """Response to selected text query with constraint verification."""

    success: bool = Field(..., description="Whether response was generated successfully")
    response_text: str = Field(..., description="Generated answer constrained to selected text")
    used_selection: bool = Field(..., description="Whether selected text was used for response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO format string"""
        return value.isoformat()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "response_text": "Based on the selected text, services provide strict ordering guarantees and enable direct communication between nodes.",
                "used_selection": True,
                "timestamp": "2025-12-16T21:05:00.123456"
            }
        }
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EmbedRequest",
    "EmbedResponse",
    "QueryRequest",
    "ChunkResult",
    "QueryResponse",
    "ChatModeEnum",
    "ChatRequest",
    "MetadataEvent",
    "TokenEvent",
    "CompleteEvent",
    "ErrorEvent",
    "SelectedTextRequest",
    "SelectedTextResponse",
]
