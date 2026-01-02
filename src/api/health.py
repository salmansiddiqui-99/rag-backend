"""
Health check endpoint for monitoring service status and dependencies.
"""
from fastapi import APIRouter
from src.config import config

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns service status and external dependency connectivity status.

    Returns:
        dict: Health status information including:
            - status: "healthy" or "degraded"
            - version: Application version
            - services: Connectivity status for external dependencies
    """
    services_status = {
        "qdrant": "unknown",  # Will be checked when vector_service is implemented
        "neon": "unknown",     # Will be checked when db_service is implemented
        "cohere": "unknown",   # Will be checked when embedding_service is implemented
        "openrouter": "unknown"  # Will be checked when generation_service is implemented
    }

    # Simple validation check
    config_errors = config.validate()
    status = "degraded" if config_errors else "healthy"

    # If config is valid, mark services as "configured"
    if not config_errors:
        services_status = {
            "qdrant": "configured" if config.QDRANT_URL and config.QDRANT_API_KEY else "not_configured",
            "neon": "configured" if config.NEON_DB_URL else "not_configured",
            "cohere": "configured" if config.COHERE_API_KEY else "not_configured",
            "openrouter": "configured" if config.OPENROUTER_API_KEY else "not_configured"
        }

    return {
        "status": status,
        "version": config.APP_VERSION,
        "services": services_status
    }
