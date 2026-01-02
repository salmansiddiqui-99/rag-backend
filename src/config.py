"""
Configuration module for loading and validating environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration from environment variables."""

    # API Keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # Service URLs
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    NEON_DB_URL: str = os.getenv("NEON_DB_URL", "")

    # CORS Configuration
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,https://localhost:3000"
    )

    # Application Settings
    APP_VERSION: str = "1.0.0"
    APP_NAME: str = "Physical AI Book RAG Chatbot"

    # OpenRouter Configuration
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "mistralai/devstral-2512:free"

    # Cohere Configuration
    COHERE_MODEL: str = "embed-english-v3.0"
    COHERE_EMBEDDING_DIM: int = 1024

    # RAG Configuration
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    TOP_K_CHUNKS: int = 5  # number of chunks to retrieve

    # Validation
    MAX_QUERY_TOKENS: int = 1000
    MAX_SELECTED_TEXT_CHARS: int = 5000
    MAX_CONVERSATION_HISTORY: int = 10

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate that all required environment variables are set.

        Returns:
            List of missing/invalid configuration keys.
        """
        errors = []

        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY is not set")
        if not cls.COHERE_API_KEY:
            errors.append("COHERE_API_KEY is not set")
        if not cls.QDRANT_URL:
            errors.append("QDRANT_URL is not set")
        if not cls.QDRANT_API_KEY:
            errors.append("QDRANT_API_KEY is not set")
        if not cls.NEON_DB_URL:
            errors.append("NEON_DB_URL is not set")

        return errors

    @classmethod
    def get_cors_origins(cls) -> list[str]:
        """Parse CORS_ORIGINS string into list."""
        return [origin.strip() for origin in cls.CORS_ORIGINS.split(",")]


# Global config instance
config = Config()
