"""Application configuration and environment loading"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # Server configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Physical AI Textbook API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Backend API for AI/Spec-Driven Book Creation"
    DEBUG: bool = False

    # Database configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/physical_ai_book"
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # -----------------------------
    # Qdrant configuration (FIXED)
    # -----------------------------
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None

    # Canonical collection name (single source of truth)
    QDRANT_COLLECTION: str = "chapter_chunks"
    QDRANT_COLLECTION_NAME: str = "chapter_chunks"

    QDRANT_VECTOR_SIZE: int = 1024
    QDRANT_DISTANCE_METRIC: str = "Cosine"

    # OpenAI configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 2000
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_TIMEOUT: int = 30

    # Claude / Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_CODE_API_KEY: Optional[str] = None
    CLAUDE_CODE_TIMEOUT: int = 60
    CLAUDE_CODE_MAX_RETRIES: int = 3
    CLAUDE_CODE_RETRY_BACKOFF: float = 2.0

    # GitHub
    GITHUB_TOKEN: Optional[str] = None
    GITHUB_REPO_OWNER: str = "yourname"
    GITHUB_REPO_NAME: str = "physical_ai_book"

    # RAG configuration
    RAG_TOP_K: int = 5
    RAG_TOP_K_CHUNKS: Optional[int] = None
    RAG_SIMILARITY_THRESHOLD: float = 0.5
    RAG_CONTEXT_MAX_TOKENS: int = 3000
    RAG_MAX_RESPONSE_TOKENS: Optional[int] = None
    RAG_RESPONSE_TOKENS: Optional[int] = None

    # Chapter generation
    CHAPTER_MIN_LENGTH: int = 1000
    CHAPTER_MAX_LENGTH: int = 5000
    CHAPTER_GENERATION_TIMEOUT: int = 300

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001,https://salmansiddiqui-99.github.io"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS"
    CORS_ALLOW_HEADERS: str = "Content-Type,Authorization"

    # Feature flags
    FEATURE_RAG_ENABLED: bool = True
    FEATURE_CHAPTER_GENERATION_ENABLED: bool = True
    FEATURE_TEXT_SELECTION_MODE: Optional[bool] = None
    FEATURE_PERFORMANCE_LOGGING: Optional[bool] = None
    FEATURE_HALLUCINATION_DETECTION: Optional[bool] = None
    FEATURE_STREAMING_RESPONSES: Optional[bool] = None

    # OpenRouter configuration (REQUIRED - only LLM provider supported)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    COHERE_API_KEY: Optional[str] = None

    LLM_PROVIDER: str = "openrouter"  # Must be "openrouter" - only supported provider
    LLM_DEFAULT_MODEL: Optional[str] = None

    # Chat & limits
    CHAT_LOG_RETENTION_DAYS: Optional[int] = None
    CHAT_MAX_MESSAGES_PER_SESSION: Optional[int] = None
    RATE_LIMIT_REQUESTS_PER_MINUTE: Optional[int] = None

    # Performance
    REQUEST_TIMEOUT: int = 30
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3

    model_config = ConfigDict(
        env_file=str(env_file) if env_file.exists() else None,
        case_sensitive=True,
        extra="ignore",
    )

    # -----------------------------
    # Helper methods
    # -----------------------------
    def get_cors_origins(self) -> list:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin]

    def get_cors_methods(self) -> list:
        return [method.strip() for method in self.CORS_ALLOW_METHODS.split(",")]

    def get_cors_headers(self) -> list:
        return [header.strip() for header in self.CORS_ALLOW_HEADERS.split(",")]

    def validate_required_keys(self) -> None:
        required_keys = [
            "COHERE_API_KEY",
            "QDRANT_URL",
            "DATABASE_URL",
            "OPENROUTER_API_KEY",  # OpenRouter is REQUIRED
        ]

        missing = [key for key in required_keys if not getattr(self, key, None)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


# Instantiate settings
settings = Settings()




def configure_logging() -> logging.Logger:
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logger = logging.getLogger("physical_ai_book")
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    if settings.LOG_FORMAT == "json":
        try:
            from pythonjsonlogger.jsonlogger import JsonFormatter
            formatter = JsonFormatter()
        except ImportError:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Configure logging at import time
logger = configure_logging()
