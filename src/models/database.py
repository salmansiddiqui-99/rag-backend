"""
SQLAlchemy ORM Models for RAG Chatbot Database

Defines all database entities using SQLAlchemy 2.0:
- Module: Textbook module grouping (4 modules)
- Chapter: Individual textbook units (12 chapters, ~1200 chunks total)
- ContentChunk: Text segments with vector metadata (200-500 tokens each)
- ChatSession: Anonymous user conversation sessions (24-hour retention)
- RAGQuery: Individual user queries and AI responses with metrics
- RetrievedChunk: Junction table linking queries to chunks

All models use PostgreSQL UUID primary keys and comprehensive audit trails.
Database: Neon Serverless PostgreSQL with Qdrant Cloud for vector storage.
"""

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, CheckConstraint,
    UniqueConstraint, Index, func, ARRAY, Numeric, Boolean, Float
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, timedelta

Base = declarative_base()


class Module(Base):
    """Module ORM model"""
    __tablename__ = "modules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(String(500), nullable=False)
    order = Column(Integer, nullable=False, unique=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    chapters = relationship("Chapter", back_populates="module")

    # Constraints
    __table_args__ = (
        CheckConstraint("order >= 1 AND order <= 4"),
    )

    def __repr__(self):
        return f"<Module(id={self.id}, name={self.name}, order={self.order})>"


class Chapter(Base):
    """Chapter ORM model"""
    __tablename__ = "chapters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    module_id = Column(UUID(as_uuid=True), ForeignKey("modules.id"), nullable=False)
    number = Column(Integer, nullable=False)
    title = Column(String(150), nullable=False)
    content_markdown = Column(Text, nullable=False)
    learning_objectives = Column(ARRAY(String), nullable=False)
    references = Column(ARRAY(String), nullable=False)
    token_count = Column(Integer, nullable=False)
    status = Column(String(20), server_default="draft", nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    module = relationship("Module", back_populates="chapters")
    content_chunks = relationship("ContentChunk", back_populates="chapter")
    rag_queries = relationship("RAGQuery", back_populates="chapter")

    # Constraints
    __table_args__ = (
        CheckConstraint("number >= 1 AND number <= 12"),
        CheckConstraint("status IN ('draft', 'published', 'archived')"),
        UniqueConstraint("module_id", "number", name="uq_chapter_module_number"),
        Index("idx_chapters_module", "module_id"),
    )

    def __repr__(self):
        return f"<Chapter(id={self.id}, number={self.number}, title={self.title}, status={self.status})>"


class ContentChunk(Base):
    """
    Content chunk ORM model - Text segment from a chapter (200-500 tokens)
    Links textbook content with Qdrant vector embeddings
    """
    __tablename__ = "content_chunks"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Qdrant Vector Store Reference
    qdrant_id = Column(String(64), unique=True, nullable=False)  # Qdrant point ID

    # Foreign Key
    chapter_id = Column(UUID(as_uuid=True), ForeignKey("chapters.id"), nullable=False)

    # Content
    section_title = Column(String(100), nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)

    # Embedding Metadata
    embedding_model = Column(String(50), default="text-embedding-3-small", nullable=False)
    embedding_dimensions = Column(Integer, default=384, nullable=False)
    embedding_created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Chunk Position
    chunk_index = Column(Integer, nullable=False)  # 0-based position in chapter

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    chapter = relationship("Chapter", back_populates="content_chunks")
    retrieved_chunks = relationship("RetrievedChunk", back_populates="chunk")

    # Constraints
    __table_args__ = (
        CheckConstraint("token_count >= 100 AND token_count <= 500", name="ck_chunk_token_bounds"),
        UniqueConstraint("qdrant_id", name="uq_chunk_qdrant_id"),
        UniqueConstraint("chapter_id", "chunk_index", name="uq_chunk_position"),
        Index("idx_chunks_chapter", "chapter_id"),
        Index("idx_chunks_qdrant", "qdrant_id"),
        Index("idx_chunks_embedding_created", "embedding_created_at"),
    )

    def __repr__(self):
        return f"<ContentChunk(id={self.id}, chapter_id={self.chapter_id}, section={self.section_title}, tokens={self.token_count})>"


class ChatSession(Base):
    """
    User conversation session (anonymous, 24-hour retention)
    Groups multiple RAGQueries into a conversation thread
    """
    __tablename__ = "chat_sessions"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Session Identity
    session_token = Column(String(128), unique=True, nullable=False)  # Browser session ID
    ip_hash = Column(String(64), nullable=False)  # Hashed IP

    # User Context
    browser_type = Column(String(100), nullable=True)  # User-Agent string
    initial_url = Column(String(1024), nullable=True)  # Entry page URL
    mode = Column(String(20), default="global", nullable=False)  # "global" or "selected_text"

    # Session Stats
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens_used = Column(Integer, default=0, nullable=False)
    total_cost_usd = Column(Float, default=0.0, nullable=False)

    # Retention & Expiry
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_activity = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)  # 24-hour TTL

    # Relationships
    rag_queries = relationship("RAGQuery", back_populates="session", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint("message_count >= 0", name="ck_session_message_count"),
        CheckConstraint("total_cost_usd >= 0", name="ck_session_cost_positive"),
        UniqueConstraint("session_token", name="uq_session_token"),
        Index("idx_session_token", "session_token"),
        Index("idx_session_created_at", "created_at"),
        Index("idx_session_expires_at", "expires_at"),
    )

    def __repr__(self):
        return f"<ChatSession(id={self.id}, messages={self.message_count}, expires={self.expires_at})>"


class RAGQuery(Base):
    """
    Individual user query and AI response with comprehensive metrics
    Tracks retrieval performance, LLM quality, and user feedback
    """
    __tablename__ = "rag_queries"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    chapter_id = Column(UUID(as_uuid=True), ForeignKey("chapters.id"), nullable=True)

    # Query Details
    query_text = Column(Text, nullable=False)
    query_tokens = Column(Integer, nullable=False)
    query_embedding_model = Column(String(50), default="text-embedding-3-small")

    # Retrieval Mode & Context
    retrieval_mode = Column(String(20), nullable=False)  # "global" or "selected_text"
    selected_text = Column(Text, nullable=True)  # Highlighted text for selection mode
    retrieved_chunk_count = Column(Integer, default=0)
    retrieval_latency_ms = Column(Integer, nullable=False)
    similarity_threshold_used = Column(Float, default=0.5)
    max_similarity_score = Column(Float, nullable=True)

    # LLM Response
    response_text = Column(Text, nullable=False)
    response_tokens = Column(Integer, nullable=False)
    llm_model = Column(String(50), default="gpt-4o")
    llm_latency_ms = Column(Integer, nullable=False)
    temperature_used = Column(Float, default=0.7)

    # Quality & Grounding
    grounded_in_context = Column(Boolean, default=True)
    response_status = Column(String(20), nullable=False)  # "success", "no_context", "error"
    user_feedback = Column(String(20), nullable=True)  # "helpful", "incorrect", "incomplete"
    feedback_notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("ChatSession", back_populates="rag_queries")
    chapter = relationship("Chapter", back_populates="rag_queries")
    retrieved_chunks = relationship("RetrievedChunk", back_populates="query", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint("retrieval_mode IN ('global', 'selected_text')", name="ck_query_retrieval_mode"),
        CheckConstraint("response_status IN ('success', 'no_context', 'error')", name="ck_query_response_status"),
        CheckConstraint("retrieved_chunk_count >= 0 AND retrieved_chunk_count <= 10", name="ck_query_chunk_count"),
        CheckConstraint("retrieval_latency_ms > 0 AND retrieval_latency_ms < 2000", name="ck_query_retrieval_latency"),
        CheckConstraint("llm_latency_ms > 0 AND llm_latency_ms < 10000", name="ck_query_llm_latency"),
        CheckConstraint("max_similarity_score IS NULL OR (max_similarity_score >= 0 AND max_similarity_score <= 1)", name="ck_query_similarity_bounds"),
        UniqueConstraint("session_id", "created_at", name="uq_query_session_timestamp"),
        Index("idx_queries_session", "session_id"),
        Index("idx_queries_chapter", "chapter_id"),
        Index("idx_queries_created", "created_at"),
        Index("idx_queries_grounded", "grounded_in_context"),
        Index("idx_queries_status", "response_status"),
    )

    def __repr__(self):
        return f"<RAGQuery(id={self.id}, status={self.response_status}, latency_ms={self.retrieval_latency_ms})>"


class RetrievedChunk(Base):
    """
    Junction table: Links RAGQueries to ContentChunks (many-to-many)
    Tracks retrieval rank and similarity score for analytics
    """
    __tablename__ = "retrieved_chunks"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    query_id = Column(UUID(as_uuid=True), ForeignKey("rag_queries.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("content_chunks.id"), nullable=False)

    # Retrieval Metrics
    rank = Column(Integer, nullable=False)  # 1 = most relevant, 5 = least
    similarity_score = Column(Float, nullable=False)  # Cosine similarity 0.0-1.0
    used_in_response = Column(Boolean, default=True)  # Was this chunk used in final response?

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    query = relationship("RAGQuery", back_populates="retrieved_chunks")
    chunk = relationship("ContentChunk", back_populates="retrieved_chunks")

    # Constraints
    __table_args__ = (
        CheckConstraint("rank >= 1 AND rank <= 10", name="ck_retrieved_rank_bounds"),
        CheckConstraint("similarity_score >= 0 AND similarity_score <= 1", name="ck_retrieved_similarity_bounds"),
        UniqueConstraint("query_id", "chunk_id", name="uq_retrieved_query_chunk"),
        UniqueConstraint("query_id", "rank", name="uq_retrieved_query_rank"),
        Index("idx_retrieved_query", "query_id"),
        Index("idx_retrieved_chunk", "chunk_id"),
        Index("idx_retrieved_rank", "rank"),
        Index("idx_retrieved_similarity", "similarity_score"),
    )

    def __repr__(self):
        return f"<RetrievedChunk(query={self.query_id}, rank={self.rank}, similarity={self.similarity_score:.2f})>"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Base",
    "Module",
    "Chapter",
    "ContentChunk",
    "ChatSession",
    "RAGQuery",
    "RetrievedChunk",
]
