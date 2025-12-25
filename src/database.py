"""Database session and engine management for the RAG Chatbot API"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

from src.config import settings

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verify connections are alive before using them
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    """
    Dependency function for FastAPI to provide database sessions.

    Yields a database session for each request and ensures proper cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_sync() -> Session:
    """
    Synchronous version of get_db for non-async contexts.

    Returns a database session that must be closed manually.
    """
    return SessionLocal()

# For backward compatibility with existing code that expects a simple session
def create_session() -> Session:
    """
    Create a database session (wrapper for compatibility).

    Remember to close the session when done: session.close()
    """
    return SessionLocal()

__all__ = ["engine", "SessionLocal", "get_db", "get_db_sync", "create_session"]