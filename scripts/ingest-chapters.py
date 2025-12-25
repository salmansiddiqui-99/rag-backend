"""
Script to ingest all chapters from textbook/docs into Qdrant vector store and PostgreSQL

Reads markdown files from textbook/docs directory, chunks them, embeds with OpenAI,
and stores in both Qdrant (vectors) and PostgreSQL (metadata). Creates ContentChunk
records linked to Chapters.

Usage:
    python scripts/ingest-chapters.py
    python scripts/ingest-chapters.py --recreate  # Drop and recreate collections
    python scripts/ingest-chapters.py --verbose   # Enable debug logging
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.config import settings
from src.models.database import Base, Chapter, Module, ContentChunk
from src.services.embedding import EmbeddingService
from src.services.chunking import ChunkingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChapterIngestion:
    """Manages chapter ingestion into Qdrant and PostgreSQL"""

    def __init__(self, recreate_collections: bool = False):
        """Initialize ingestion with database and vector store clients"""
        # Database setup
        self.engine = create_engine(settings.DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.db_session = Session(self.engine)

        # Services
        self.embedding_service = EmbeddingService()
        self.chunking_service = ChunkingService(target_chunk_size=200)

        # Qdrant setup
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

        self.collection_name = settings.QDRANT_COLLECTION_NAME or "physical_ai_chapters"
        self.recreate_collections = recreate_collections

        # Textbook directory
        self.textbook_dir = Path(__file__).parent.parent.parent / "textbook" / "docs"
        if not self.textbook_dir.exists():
            raise FileNotFoundError(f"Textbook directory not found: {self.textbook_dir}")

        logger.info(f"Ingestion initialized: {self.textbook_dir}")

    def setup_collections(self) -> bool:
        """Create Qdrant collection if needed"""
        try:
            # Check if collection exists
            try:
                self.qdrant_client.get_collection(self.collection_name)
                if self.recreate_collections:
                    logger.info(f"Deleting existing collection '{self.collection_name}'")
                    self.qdrant_client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return True
            except Exception:
                pass

            # Create collection
            logger.info(f"Creating Qdrant collection '{self.collection_name}'")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info("✓ Collection created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to set up collection: {e}")
            return False

    def ensure_modules(self) -> dict:
        """Ensure 4 module records exist in database"""
        modules = {}
        module_names = [
            "ROS 2 Fundamentals & Architecture",
            "Gazebo Simulation & Physics",
            "NVIDIA Isaac & AI Pipelines",
            "Capstone: VLA-Based Humanoid Control"
        ]

        for idx, name in enumerate(module_names, 1):
            existing = self.db_session.query(Module).filter(Module.order == idx).first()
            if existing:
                modules[idx] = existing
            else:
                module = Module(
                    order=idx,
                    name=name,
                    description=f"Module {idx}: {name}",
                )
                self.db_session.add(module)
                modules[idx] = module

        self.db_session.commit()
        logger.info(f"✓ Ensured {len(modules)} modules exist")
        return modules

    def find_chapters(self) -> dict:
        """Find and organize chapter files by module"""
        chapters_by_module = {}

        # Scan docs directory for markdown files
        for module_dir in sorted(self.textbook_dir.glob("module*")):
            if not module_dir.is_dir():
                continue

            module_num = int(module_dir.name.replace("module", ""))
            chapters_by_module[module_num] = []

            for chapter_file in sorted(module_dir.glob("*.md")):
                chapters_by_module[module_num].append(chapter_file)

        total_chapters = sum(len(chs) for chs in chapters_by_module.values())
        logger.info(f"Found {total_chapters} chapters across {len(chapters_by_module)} modules")

        return chapters_by_module

    def ingest_chapter(
        self,
        chapter_file: Path,
        module_id: str,
        chapter_number: int,
    ) -> bool:
        """
        Ingest a single chapter: chunk, embed, and store

        Args:
            chapter_file: Path to markdown file
            module_id: Module UUID
            chapter_number: Chapter number within module

        Returns:
            True if successful
        """
        try:
            # Read chapter content
            with open(chapter_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                logger.warning(f"Empty chapter file: {chapter_file}")
                return False

            # Extract title from first heading or filename
            lines = content.split("\n")
            title = None
            for line in lines:
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break
            if not title:
                title = chapter_file.stem.replace("-", " ").title()

            # Check if chapter already exists (by module_id + number)
            existing = self.db_session.query(Chapter).filter(
                Chapter.module_id == module_id,
                Chapter.number == chapter_number
            ).first()

            if existing:
                logger.info(f"Chapter already exists: {existing.title}")
                return True

            # Chunk content
            chunks = self.chunking_service.chunk_chapter(
                content=content,
                chapter_id=str(uuid4()),
                section_title=title
            )

            if not chunks:
                logger.warning(f"No chunks extracted from {chapter_file}")
                return False

            logger.info(f"Extracted {len(chunks)} chunks from {chapter_file.name}")

            # Create Chapter record
            chapter = Chapter(
                module_id=module_id,
                number=chapter_number,
                title=title,
                content_markdown=content,
                learning_objectives=[],
                references=[],
                token_count=len(content) // 4,  # Rough estimate
            )
            self.db_session.add(chapter)
            self.db_session.flush()  # Get chapter.id

            # Process and store chunks (filter by token count: 100-500)
            vectors_to_upsert = []
            valid_chunk_idx = 0
            for chunk in chunks:
                token_count = chunk.get("token_count", len(chunk["text"]) // 4)

                # Skip chunks outside token bounds (must be 100-500 tokens)
                if token_count < 100 or token_count > 500:
                    logger.debug(
                        f"Skipping chunk with {token_count} tokens (must be 100-500): "
                        f"{chunk.get('section_title', '')[:50]}"
                    )
                    continue

                # Embed chunk
                embedding = self.embedding_service.embed_text(chunk["text"])
                qdrant_id = str(uuid4())

                # Create ContentChunk record
                content_chunk = ContentChunk(
                    qdrant_id=qdrant_id,
                    chapter_id=chapter.id,
                    section_title=chunk.get("section_title", ""),
                    text=chunk["text"],
                    token_count=token_count,
                    embedding_model="text-embedding-3-small",
                    embedding_dimensions=384,
                    embedding_created_at=datetime.utcnow(),
                    chunk_index=valid_chunk_idx,
                )
                self.db_session.add(content_chunk)
                valid_chunk_idx += 1

                # Prepare Qdrant vector
                vectors_to_upsert.append(
                    PointStruct(
                        id=int(qdrant_id.replace("-", "")[:16], 16) % (2**63),  # Convert UUID to int
                        vector=embedding,
                        payload={
                            "chapter_id": str(chapter.id),
                            "chunk_id": str(content_chunk.id),
                            "section_title": chunk.get("section_title", ""),
                            "chunk_index": valid_chunk_idx - 1,
                        }
                    )
                )

            # Commit database changes
            self.db_session.commit()
            logger.info(f"✓ Stored {len(chunks)} chunks in PostgreSQL for {title}")

            # Upload vectors to Qdrant
            if vectors_to_upsert:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=vectors_to_upsert,
                )
                logger.info(f"✓ Uploaded {len(vectors_to_upsert)} vectors to Qdrant")

            return True

        except Exception as e:
            logger.error(f"Failed to ingest {chapter_file}: {e}")
            self.db_session.rollback()
            return False

    def ingest_all(self) -> dict:
        """
        Ingest all chapters from textbook directory

        Returns:
            Summary dict with counts
        """
        logger.info("Starting chapter ingestion...")

        # Setup collections
        if not self.setup_collections():
            logger.error("Failed to setup Qdrant collection")
            return {"success": False, "total": 0, "ingested": 0, "failed": 0}

        # Ensure modules exist
        modules = self.ensure_modules()

        # Find chapters
        chapters_by_module = self.find_chapters()

        # Ingest chapters
        total = 0
        ingested = 0
        failed = 0

        for module_num in sorted(chapters_by_module.keys()):
            module_id = modules[module_num].id
            chapter_files = chapters_by_module[module_num]

            for ch_num, chapter_file in enumerate(chapter_files, 1):
                total += 1
                logger.info(
                    f"[{total}] Ingesting Module {module_num} Chapter {ch_num}: "
                    f"{chapter_file.name}"
                )

                success = self.ingest_chapter(
                    chapter_file=chapter_file,
                    module_id=module_id,
                    chapter_number=ch_num,
                )

                if success:
                    ingested += 1
                else:
                    failed += 1

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info(f"  Total chapters: {total}")
        logger.info(f"  Successfully ingested: {ingested}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info("=" * 60)

        return {
            "success": failed == 0,
            "total": total,
            "ingested": ingested,
            "failed": failed,
        }

    def cleanup(self):
        """Close database connection"""
        self.db_session.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Ingest chapters from textbook into Qdrant and PostgreSQL"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection (deletes existing data)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        ingestion = ChapterIngestion(recreate_collections=args.recreate)
        result = ingestion.ingest_all()
        ingestion.cleanup()

        if not result["success"]:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
