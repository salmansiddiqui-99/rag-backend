"""Pydantic models for Chapter and related entities"""
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from datetime import datetime


class ChapterStatus(str, Enum):
    """Chapter publication status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ChapterBase(BaseModel):
    """Base chapter schema"""
    title: str = Field(..., min_length=10, max_length=150)
    learning_objectives: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Learning objectives (3-5 items)"
    )
    references: List[str] = Field(
        ...,
        min_length=1,
        description="List of citations/sources"
    )

    @field_validator('learning_objectives')
    @classmethod
    def validate_objectives(cls, v: List[str]) -> List[str]:
        for obj in v:
            if not (20 <= len(obj) <= 200):
                raise ValueError('Each objective must be 20-200 characters')
        return v

    @field_validator('references')
    @classmethod
    def validate_references(cls, v: List[str]) -> List[str]:
        for ref in v:
            if not ref or len(ref) == 0:
                raise ValueError('References cannot be empty')
        return v


class ChapterCreate(ChapterBase):
    """Schema for creating a chapter"""
    module_id: UUID
    number: int = Field(..., ge=1, le=12)
    content_markdown: str = Field(..., min_length=1000)
    token_count: Optional[int] = Field(None, ge=0)

    @field_validator('number')
    @classmethod
    def validate_number(cls, v: int) -> int:
        if not (1 <= v <= 12):
            raise ValueError('Chapter number must be between 1 and 12')
        return v

    @field_validator('content_markdown')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if '[Citation:' not in v:
            raise ValueError('Content must include at least one citation')
        return v


class Chapter(ChapterBase):
    """Full chapter schema"""
    id: UUID
    module_id: UUID
    number: int
    content_markdown: str
    token_count: int
    status: ChapterStatus = ChapterStatus.DRAFT
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChapterUpdate(BaseModel):
    """Schema for updating a chapter"""
    title: Optional[str] = Field(None, min_length=10, max_length=150)
    content_markdown: Optional[str] = Field(None, min_length=1000)
    learning_objectives: Optional[List[str]] = None
    references: Optional[List[str]] = None
    status: Optional[ChapterStatus] = None
    token_count: Optional[int] = None


class ChapterListResponse(BaseModel):
    """Schema for listing chapters"""
    id: UUID
    module_id: UUID
    number: int
    title: str
    token_count: int
    status: ChapterStatus
    created_at: datetime


class ChapterResponse(Chapter):
    """Schema for chapter API responses"""
    pass


class ContentChunkCreate(BaseModel):
    """Schema for creating a content chunk"""
    chapter_id: UUID
    section_title: str = Field(..., min_length=5, max_length=100)
    text: str = Field(..., min_length=100, max_length=1000)
    token_count: int = Field(..., ge=100, le=300)


class ContentChunk(ContentChunkCreate):
    """Full content chunk schema"""
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ContentChunkResponse(ContentChunk):
    """Schema for content chunk API responses"""
    pass


class RetrievedChunkCreate(BaseModel):
    """Schema for retrieved chunks in RAG results"""
    query_id: UUID
    chunk_id: UUID
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1, le=5)


class RetrievedChunk(RetrievedChunkCreate):
    """Full retrieved chunk schema"""
    id: UUID

    model_config = ConfigDict(from_attributes=True)


class RetrievedChunkResponse(RetrievedChunk):
    """Schema for retrieved chunk API responses"""
    pass
