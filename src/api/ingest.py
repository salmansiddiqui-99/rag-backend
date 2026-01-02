"""
Ingestion endpoint for triggering content indexing.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
import os

router = APIRouter()


class IngestRequest(BaseModel):
    """Request model for /ingest endpoint."""
    content_dir: str


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint."""
    indexed_files: int
    total_chunks: int
    duration_seconds: float
    status: str


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest):
    """
    Trigger content ingestion from MDX files.

    This endpoint runs the ingestion script to index book content.

    Args:
        request: IngestRequest with content_dir path

    Returns:
        IngestResponse with ingestion statistics

    Raises:
        HTTPException 400: If content directory is invalid
        HTTPException 500: If ingestion fails
    """
    content_dir = request.content_dir

    # Validate content directory exists
    if not os.path.exists(content_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Content directory not found: {content_dir}"
        )

    try:
        # Run ingestion script
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "scripts",
            "ingest.py"
        )

        result = subprocess.run(
            ["python", script_path, "--content-dir", content_dir],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            raise Exception(f"Ingestion script failed: {result.stderr}")

        # Parse output for statistics
        # TODO: Enhance ingestion script to output JSON for easier parsing
        # For now, return success with placeholder stats
        return IngestResponse(
            indexed_files=0,  # Will be populated from script output
            total_chunks=0,   # Will be populated from script output
            duration_seconds=0.0,  # Will be populated from script output
            status="success"
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Ingestion timed out after 10 minutes"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )
