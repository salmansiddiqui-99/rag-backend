"""
API request and response models for chatbot endpoints
Defines Pydantic models for request validation and response serialization
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import datetime


class ChatbotQuery(BaseModel):
    """
    Request model for chatbot query endpoint
    Supports three modes: global search, chapter-specific, text-selection
    """
    query: str = Field(..., min_length=1, max_length=1000, description="User query (1-1000 characters)")
    mode: Literal["global", "chapter", "text-selection"] = Field(
        default="global",
        description="Retrieval mode: global (whole book), chapter (specific), or text-selection (from highlighted text)"
    )
    selected_text: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Selected text from document (required if mode='text-selection')"
    )
    chapter_id: Optional[str] = Field(
        default=None,
        description="Chapter ID like 'module-1-chapter-1' (required if mode='chapter')"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is non-empty after stripping"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

    @field_validator("selected_text")
    @classmethod
    def validate_selected_text(cls, v: Optional[str], info) -> Optional[str]:
        """Validate selected_text is present when mode='text-selection'"""
        if info.data.get("mode") == "text-selection":
            if not v or not v.strip():
                raise ValueError("selected_text is required when mode='text-selection'")
        return v

    @field_validator("chapter_id")
    @classmethod
    def validate_chapter_id(cls, v: Optional[str], info) -> Optional[str]:
        """Validate chapter_id is present when mode='chapter'"""
        if info.data.get("mode") == "chapter":
            if not v or not v.strip():
                raise ValueError("chapter_id is required when mode='chapter'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is ROS?",
                "mode": "global"
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: Literal["ok", "unavailable"] = Field(..., description="Service status")
    uptime_seconds: int = Field(..., description="Server uptime in seconds")
    version: Optional[str] = Field(default="1.0.0", description="API version")
    timestamp: str = Field(..., description="ISO8601 timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "uptime_seconds": 3600,
                "version": "1.0.0",
                "timestamp": "2025-12-19T15:30:45.123Z"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Error code (e.g., INVALID_INPUT, NOT_FOUND)")
    details: Optional[dict] = Field(
        default=None,
        description="Additional error details (field-level validation errors, etc.)"
    )
    timestamp: str = Field(..., description="ISO8601 timestamp when error occurred")
    retry_after_seconds: Optional[int] = Field(
        default=None,
        description="Seconds to wait before retrying (for 429 errors)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid query parameter",
                "code": "INVALID_INPUT",
                "details": {
                    "query": "must be between 1 and 1000 characters"
                },
                "timestamp": "2025-12-19T15:30:45.123Z"
            }
        }


class ChatbotTokenLine(BaseModel):
    """Individual token line in streaming NDJSON response"""
    type: Literal["token"] = Field(..., description="Line type: token")
    data: str = Field(..., description="Token text")
    timestamp: str = Field(..., description="ISO8601 timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "token",
                "data": "ROS",
                "timestamp": "2025-12-19T15:30:45.123Z"
            }
        }


class ChatbotMetadataLine(BaseModel):
    """Metadata line at end of streaming NDJSON response"""
    type: Literal["metadata"] = Field(..., description="Line type: metadata")
    data: dict = Field(..., description="Metadata object with chunks_used, query_time_ms, etc.")
    timestamp: str = Field(..., description="ISO8601 timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "metadata",
                "data": {
                    "chunks_used": 3,
                    "query_time_ms": 250,
                    "model": "claude-haiku",
                    "retrieval_mode": "global"
                },
                "timestamp": "2025-12-19T15:30:45.500Z"
            }
        }
