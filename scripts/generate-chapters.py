#!/usr/bin/env python3
"""
Batch Chapter Generation Script - Generates all 12 chapters in parallel.

Usage:
    python generate-chapters.py [--max-workers 4] [--output-dir textbook/docs]
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directories to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.src.services.chapter_gen import ChapterGenerationService
from backend.src.services.validation import ContentValidationService
from backend.src.services.chapter_storage import ChapterStorageService
from backend.src.services.chunking import ChunkingService
from backend.src.services.embedding import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Chapter specifications
CHAPTERS_SPEC: List[Dict[str, Any]] = [
    # Module 1: ROS 2
    {
        "module_id": "module1",
        "chapter_number": 1,
        "title": "ROS 2 Basics: Architecture and Communication",
        "description": "Fundamentals of ROS 2, pub/sub architecture, and workspace setup",
    },
    {
        "module_id": "module1",
        "chapter_number": 2,
        "title": "Humanoid Control with ROS 2",
        "description": "Control humanoid robots using ROS 2 services, actions, and state machines",
    },
    {
        "module_id": "module1",
        "chapter_number": 3,
        "title": "URDF and Robot Descriptions",
        "description": "Define humanoid robot structure using URDF, joints, and kinematics",
    },
    # Module 2: Gazebo & Unity
    {
        "module_id": "module2",
        "chapter_number": 4,
        "title": "Gazebo Simulation Fundamentals",
        "description": "Set up realistic physics simulations for humanoid robots in Gazebo",
    },
    {
        "module_id": "module2",
        "chapter_number": 5,
        "title": "Sensor Simulation and Perception",
        "description": "Simulate cameras, LiDAR, IMU sensors for robot perception tasks",
    },
    {
        "module_id": "module2",
        "chapter_number": 6,
        "title": "Unity for Robot Visualization",
        "description": "Visualize simulated humanoid robots in Unity game engine",
    },
    # Module 3: NVIDIA Isaac
    {
        "module_id": "module3",
        "chapter_number": 7,
        "title": "NVIDIA Isaac Sim for Humanoids",
        "description": "High-performance simulation and digital twins using Isaac Sim",
    },
    {
        "module_id": "module3",
        "chapter_number": 8,
        "title": "Isaac Perception and AI",
        "description": "Integrate computer vision and AI models in Isaac Sim",
    },
    {
        "module_id": "module3",
        "chapter_number": 9,
        "title": "Nav2 and Bipedal Navigation",
        "description": "Implement autonomous navigation for bipedal humanoid robots",
    },
    # Module 4: VLA & Capstone
    {
        "module_id": "module4",
        "chapter_number": 10,
        "title": "Voice and Language Action (VLA) Systems",
        "description": "Build voice-to-action systems for humanoid robot control",
    },
    {
        "module_id": "module4",
        "chapter_number": 11,
        "title": "Cognitive Planning and Reasoning",
        "description": "Implement planning algorithms for complex humanoid tasks",
    },
    {
        "module_id": "module4",
        "chapter_number": 12,
        "title": "Capstone: Autonomous Humanoid System",
        "description": "Build a complete autonomous humanoid system integrating all modules",
    },
]


async def generate_all_chapters(
    max_workers: int = 4,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Generate all 12 chapters in parallel with concurrency control.

    Args:
        max_workers: Maximum concurrent generation tasks
        output_dir: Output directory for chapters

    Returns:
        Results dict with summary and details
    """
    logger.info("=" * 80)
    logger.info("BATCH CHAPTER GENERATION")
    logger.info("=" * 80)
    logger.info(f"Generating {len(CHAPTERS_SPEC)} chapters with {max_workers} concurrent workers")
    logger.info(f"Start time: {datetime.utcnow().isoformat()}")

    # Initialize services
    chapter_gen_service = ChapterGenerationService()
    validation_service = ContentValidationService()
    storage_service = ChapterStorageService(repo_root=Path.cwd())
    chunking_service = ChunkingService()
    embedding_service = EmbeddingService()

    # Ensure Qdrant collection exists
    logger.info("Creating Qdrant collection...")
    embedding_service.create_or_update_collection()

    # Generate chapters with concurrency control
    semaphore = asyncio.Semaphore(max_workers)

    async def generate_with_semaphore(spec: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            chapter_num = spec["chapter_number"]
            title = spec["title"]
            try:
                logger.info(f"\n[Chapter {chapter_num}] Starting generation: {title}")

                # Generate content
                content = await chapter_gen_service.generate_chapter(
                    module_id=spec["module_id"],
                    chapter_number=chapter_num,
                    title=title,
                    description=spec["description"],
                )
                logger.info(f"[Chapter {chapter_num}] ✓ Content generated ({len(content)} chars)")

                # Validate content
                validation_result = validation_service.validate_chapter(content)
                if not validation_result["passed"]:
                    raise ValueError(
                        f"Validation failed: {', '.join(validation_result['critical_issues'])}"
                    )
                logger.info(f"[Chapter {chapter_num}] ✓ Content validated")

                # Save to filesystem
                chapter_path = storage_service.save_chapter_markdown(
                    module_id=spec["module_id"],
                    chapter_number=chapter_num,
                    title=title,
                    content=content,
                )
                logger.info(f"[Chapter {chapter_num}] ✓ Saved to {chapter_path}")

                # Commit to git
                storage_service.commit_to_git(
                    files=[chapter_path],
                    message=f"Add chapter {chapter_num}: {title}",
                )
                logger.info(f"[Chapter {chapter_num}] ✓ Committed to git")

                # Chunk and embed
                chunks = chunking_service.chunk_chapter(
                    content=content,
                    chapter_id=spec["module_id"],
                )
                logger.info(f"[Chapter {chapter_num}] ✓ Chunked into {len(chunks)} chunks")

                embedded_chunks = embedding_service.embed_chunks_batch(chunks)
                logger.info(f"[Chapter {chapter_num}] ✓ Embedded {len(embedded_chunks)} chunks")

                embedding_service.upsert_to_qdrant(embedded_chunks)
                logger.info(f"[Chapter {chapter_num}] ✓ Indexed in Qdrant")

                token_count = sum(chunk["token_count"] for chunk in chunks)

                return {
                    "success": True,
                    "chapter_number": chapter_num,
                    "title": title,
                    "chapter_path": str(chapter_path),
                    "content_size": len(content),
                    "chunks_count": len(embedded_chunks),
                    "token_count": token_count,
                }

            except Exception as e:
                logger.error(f"[Chapter {chapter_num}] ✗ Failed: {str(e)}")
                return {
                    "success": False,
                    "chapter_number": chapter_num,
                    "title": title,
                    "error": str(e),
                }

    # Run all chapter generations
    results = await asyncio.gather(
        *[generate_with_semaphore(spec) for spec in CHAPTERS_SPEC],
        return_exceptions=False,
    )

    # Summarize results
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 80)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    logger.info(f"✓ Successful: {len(successful)}/{len(results)}")
    logger.info(f"✗ Failed: {len(failed)}/{len(results)}")

    if successful:
        total_tokens = sum(r["token_count"] for r in successful)
        total_chunks = sum(r["chunks_count"] for r in successful)
        logger.info(f"  Total tokens generated: {total_tokens}")
        logger.info(f"  Total chunks indexed: {total_chunks}")

        logger.info("\nSuccessful chapters:")
        for r in successful:
            logger.info(
                f"  - Chapter {r['chapter_number']}: {r['title']} "
                f"({r['token_count']} tokens, {r['chunks_count']} chunks)"
            )

    if failed:
        logger.error("\nFailed chapters:")
        for r in failed:
            logger.error(f"  - Chapter {r['chapter_number']}: {r['title']} - {r['error']}")

    # Get Qdrant stats
    try:
        stats = embedding_service.get_collection_stats()
        logger.info(f"\nQdrant collection stats: {stats['point_count']} points indexed")
    except Exception as e:
        logger.warning(f"Could not retrieve Qdrant stats: {str(e)}")

    logger.info(f"\nEnd time: {datetime.utcnow().isoformat()}")
    logger.info("=" * 80)

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate all 12 chapters in parallel"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum concurrent generation tasks (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("textbook/docs"),
        help="Output directory for chapters",
    )

    args = parser.parse_args()

    # Run async generation
    result = asyncio.run(
        generate_all_chapters(
            max_workers=args.max_workers,
            output_dir=args.output_dir,
        )
    )

    # Exit with appropriate code
    exit_code = 0 if result["failed"] == 0 else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
