"""API endpoints for chapter management"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Query, HTTPException, status, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.models.chapter import (
    Chapter, ChapterCreate, ChapterUpdate, ChapterListResponse, ChapterResponse
)
from src.models.database import Chapter as ChapterDB, ContentChunk as ContentChunkDB, Module as ModuleDB
from src.errors import NotFoundError
from src.services.chapter_gen import ChapterGenerationService
from src.services.validation import ContentValidationService
from src.services.chapter_storage import ChapterStorageService
from src.services.chunking import ChunkingService
from src.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chapters", tags=["chapters"])

# Service instances
chapter_gen_service = ChapterGenerationService()
validation_service = ContentValidationService()
storage_service = ChapterStorageService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()

# In-memory job tracking (for MVP - use database in production)
generation_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/generate", response_model=dict)
async def generate_chapter(
    request: dict,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Generate a new chapter using Claude Code subagent.

    Request body:
    - module_id: UUID of the module (string)
    - chapter_number: Chapter number (1-12)
    - title: Chapter title
    - description: Chapter description

    Response:
    - status: "processing"
    - job_id: Background job ID for tracking
    """
    import uuid
    from datetime import datetime

    module_id = request.get("module_id")
    chapter_number = request.get("chapter_number")
    title = request.get("title")
    description = request.get("description")

    if not all([module_id, chapter_number, title, description]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: module_id, chapter_number, title, description",
        )

    job_id = str(uuid.uuid4())

    # Initialize job tracking
    generation_jobs[job_id] = {
        "status": "processing",
        "module_id": module_id,
        "chapter_number": chapter_number,
        "title": title,
        "created_at": datetime.utcnow(),
    }

    # Queue background task
    background_tasks.add_task(
        _generate_chapter_background,
        job_id=job_id,
        module_id=module_id,
        chapter_number=chapter_number,
        title=title,
        description=description,
    )

    return {
        "status": "processing",
        "job_id": job_id,
    }


async def _generate_chapter_background(
    job_id: str,
    module_id: str,
    chapter_number: int,
    title: str,
    description: str,
) -> None:
    """Background task for chapter generation."""
    try:
        # Generate content
        logger.info(f"[{job_id}] Starting chapter generation")
        content = await chapter_gen_service.generate_chapter(
            module_id=module_id,
            chapter_number=chapter_number,
            title=title,
            description=description,
        )

        # Validate content
        logger.info(f"[{job_id}] Validating chapter content")
        validation_result = validation_service.validate_chapter(content)

        if not validation_result["passed"]:
            raise ValueError(
                f"Validation failed: {', '.join(validation_result['critical_issues'])}"
            )

        # Save chapter to filesystem
        logger.info(f"[{job_id}] Saving chapter to filesystem")
        chapter_path = storage_service.save_chapter_markdown(
            module_id=module_id,
            chapter_number=chapter_number,
            title=title,
            content=content,
        )

        # Commit to git
        logger.info(f"[{job_id}] Committing to git")
        storage_service.commit_to_git(
            files=[chapter_path],
            message=f"Add chapter {chapter_number}: {title}",
        )

        # Chunk content
        logger.info(f"[{job_id}] Chunking chapter content")
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id=module_id,  # For now, use module_id as chapter reference
        )

        # Embed chunks
        logger.info(f"[{job_id}] Embedding chunks")
        embedded_chunks = embedding_service.embed_chunks_batch(chunks)

        # Upsert to Qdrant
        logger.info(f"[{job_id}] Upserting to Qdrant")
        embedding_service.upsert_to_qdrant(embedded_chunks)

        # Calculate token count
        token_count = sum(chunk["token_count"] for chunk in chunks)

        # Update job status
        generation_jobs[job_id] = {
            "status": "completed",
            "module_id": module_id,
            "chapter_number": chapter_number,
            "title": title,
            "chapter_path": str(chapter_path),
            "token_count": token_count,
            "chunks_created": len(embedded_chunks),
        }

        logger.info(f"[{job_id}] Chapter generation completed successfully")

    except Exception as e:
        logger.error(f"[{job_id}] Chapter generation failed: {str(e)}")
        generation_jobs[job_id] = {
            "status": "failed",
            "error": str(e),
        }


@router.get("/jobs/{job_id}", response_model=dict)
async def get_generation_job(job_id: str) -> dict:
    """
    Get the status of a chapter generation job.

    Response:
    - job_id: Job ID
    - status: "pending" | "processing" | "completed" | "failed"
    - chapter_path: str (if completed)
    - token_count: int (if completed)
    - error: str (if failed)
    """
    if job_id not in generation_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return {
        "job_id": job_id,
        **generation_jobs[job_id],
    }


from src.database import SessionLocal
from src.models.database import Chapter as ChapterModel

from fastapi import Depends

# Database dependency
def get_db_session():
    logger.info("get_db_session dependency called")
    db = SessionLocal()
    try:
        logger.info("Database session created")
        yield db
    finally:
        logger.info("Closing database session")
        db.close()

@router.get("", response_model=List[ChapterListResponse])
async def list_chapters(
    module_id: Optional[str] = Query(None),
    chapter_status: Optional[str] = Query(None, alias="status"),
) -> List[ChapterListResponse]:
    """
    List all chapters with optional filtering.

    Query parameters:
    - module_id: Filter by module ID
    - status: Filter by status (draft, published, archived)

    Response: List of chapters with metadata
    """
    logger.info(f"list_chapters endpoint called with module_id={module_id}, status={chapter_status}")
    # Force an obvious log to make sure this is called
    print("DEBUG: list_chapters function is being executed!")
    logger.info("DEBUG: list_chapters function is being executed!")

    from src.database import SessionLocal
    from sqlalchemy.orm import Session

    # Create session directly for testing
    db: Session = SessionLocal()
    try:
        logger.info("Starting database query...")
        # Build query with optional filters
        query = db.query(ChapterModel)

        if module_id:
            logger.info(f"Filtering by module_id: {module_id}")
            # Convert module_id to UUID if needed
            query = query.filter(ChapterModel.module_id == module_id)

        if chapter_status:
            logger.info(f"Filtering by status: {chapter_status}")
            query = query.filter(ChapterModel.status == chapter_status)

        # Execute query
        logger.info("Executing query...")
        chapters_db = query.all()
        logger.info(f"Query returned {len(chapters_db)} results from database")

        # Convert to response format
        chapters = []
        logger.info(f"Converting {len(chapters_db)} chapters to response format...")
        for i, chapter_db in enumerate(chapters_db):
            logger.info(f"Processing chapter {i+1}: {chapter_db.title} (ID: {chapter_db.id})")
            try:
                chapter_response = ChapterListResponse(
                    id=chapter_db.id,
                    module_id=chapter_db.module_id,
                    number=chapter_db.number,
                    title=chapter_db.title,
                    token_count=chapter_db.token_count,
                    status=chapter_db.status,
                    created_at=chapter_db.created_at
                )
                chapters.append(chapter_response)
                logger.info(f"Successfully created response for chapter: {chapter_response.title}")
            except Exception as convert_error:
                logger.error(f"Error converting chapter {chapter_db.id} to response: {convert_error}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Skip this chapter and continue with others
                continue

        logger.info(f"Successfully converted {len(chapters)} chapters, returning results")
        return chapters

    except Exception as e:
        logger.error(f"Failed to list chapters: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to empty list if database is unavailable
        return []
    finally:
        db.close()


@router.get("/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(chapter_id: str) -> ChapterResponse:
    """
    Get a single chapter by ID.

    Response: Full chapter content with metadata
    """
    # For MVP, return mock data
    logger.info(f"Retrieving chapter {chapter_id}")
    raise NotFoundError("Chapter", chapter_id)


@router.post("/validate", response_model=dict)
async def validate_chapter(request: dict) -> dict:
    """
    Validate a chapter for correctness and completeness.

    Request body:
    - chapter_id: Chapter UUID (or pass content directly)
    - content: Chapter content (optional, for direct validation)

    Returns:
    - passed: bool
    - citations: Citation validation details
    - code_blocks: Code block validation details
    - claims: Claim validation details
    - critical_issues: Blocking issues
    - warnings: Non-blocking issues
    """
    content = request.get("content")

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required field: content",
        )

    result = validation_service.validate_chapter(content)

    return {
        "passed": result["passed"],
        "citations": result["citations"],
        "code_blocks": result["code_blocks"],
        "claims": result["claims"],
        "critical_issues": result["critical_issues"],
        "warnings": result["warnings"],
    }
