"""
SQLAlchemy models for document metadata storage in Neon Postgres.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DocumentMetadata(Base):
    """
    Document metadata for tracking indexed content.

    State transitions:
    unindexed → indexing → indexed → stale → re-indexing
    """

    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(512), nullable=False, unique=True, index=True)
    content_hash = Column(String(64), nullable=False)  # SHA-256 hash
    last_indexed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    chunk_count = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint('file_path', name='uq_file_path'),
    )

    def __repr__(self):
        return f"<DocumentMetadata(file_path='{self.file_path}', chunks={self.chunk_count})>"
