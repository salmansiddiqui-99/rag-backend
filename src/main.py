"""
FastAPI application entry point for RAG chatbot backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import config
from src.api.health import router as health_router
from src.api.chat import router as chat_router
from src.api.ingest import router as ingest_router

# Create FastAPI app instance
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="Backend API for Physical AI & Humanoid Robotics Book RAG chatbot"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["health"])
app.include_router(chat_router, tags=["chat"])
app.include_router(ingest_router, tags=["ingest"])


@app.on_event("startup")
async def startup_event():
    """Run validation on startup."""
    errors = config.validate()
    if errors:
        print("WARNING: Configuration warnings:")
        for error in errors:
            print(f"  - {error}")
        print("\nTIP: Update your .env file with required API keys")
    else:
        print(f"SUCCESS: {config.APP_NAME} v{config.APP_VERSION} started successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }
