"""
Pydantic models for API request/response schemas.
"""
from typing import Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class QueryMode(str, Enum):
    """Query processing mode."""
    RAG = "rag"  # Full-book retrieval-augmented generation
    SELECTED_TEXT = "selected_text"  # Context from user-selected text only


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatQuery(BaseModel):
    """Request model for /chat endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's question or request"
    )
    selected_text: Optional[str] = Field(
        None,
        max_length=5000,
        description="User-selected text for context (triggers selected_text mode)"
    )
    conversation_history: Optional[list[ConversationMessage]] = Field(
        None,
        max_length=10,
        description="Previous conversation messages for context"
    )

    @field_validator("selected_text")
    @classmethod
    def validate_selected_text(cls, v):
        """Ensure selected text is not empty if provided."""
        if v is not None and v.strip() == "":
            raise ValueError("selected_text cannot be empty if provided")
        return v


class SourceChunk(BaseModel):
    """Citation source from retrieved content."""

    chapter: str = Field(..., description="Chapter title")
    section: Optional[str] = Field(None, description="Section heading (if available)")
    snippet: str = Field(..., description="Relevant text excerpt")


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""

    response: str = Field(..., description="AI-generated response to the query")
    source_chunks: list[SourceChunk] = Field(
        default_factory=list,
        description="Citations from book content (empty for selected_text mode)"
    )
    mode: QueryMode = Field(..., description="Query processing mode used")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response generation timestamp (ISO 8601)"
    )
