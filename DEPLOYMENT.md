# Railway Deployment Configuration

This project is configured for deployment on Railway using a Dockerfile.

## Deployment Approach

- **Dockerfile**: Used for containerized deployment on Railway
- **Procfile**: Removed to ensure Dockerfile takes precedence
- **Port Configuration**: The application reads the PORT environment variable provided by Railway
- **Startup Script**: Uses `start_server.py` to properly handle environment variables

## Environment Variables Required

Make sure to set these environment variables in your Railway deployment:

- `OPENROUTER_API_KEY` (required)
- `QDRANT_API_KEY` (if using Qdrant cloud)
- `DATABASE_URL` (PostgreSQL connection string)
- `QDRANT_URL` (Qdrant connection URL)

## Startup Process

The application uses a custom startup script (`start_server.py`) that properly handles the PORT environment variable provided by Railway, ensuring the uvicorn server starts on the correct port.