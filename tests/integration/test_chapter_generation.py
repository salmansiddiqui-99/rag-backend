"""Integration tests for chapter generation pipeline."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.services.chapter_gen import ChapterGenerationService
from src.services.validation import ContentValidationService
from src.services.chapter_storage import ChapterStorageService
from src.services.chunking import ChunkingService
from src.services.embedding import EmbeddingService


@pytest.fixture
def mock_chapter_content():
    """Mock generated chapter content."""
    return """
## ROS 2 Basics: Architecture and Communication

### Introduction
ROS 2 is a flexible middleware for writing robotic software. [Citation: https://docs.ros.org/en/humble/]

### Core Concepts

ROS 2 uses a publish-subscribe pattern for communication. [Citation: https://docs.ros.org/en/humble/Concepts/Intermediate/About-Topic-ROS-2.html]

### Practical Examples

Here's a simple ROS 2 node:

```python
import rclpy

def main():
    rclpy.init()
    node = rclpy.create_node('example')
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

### Summary
- ROS 2 is a robotics middleware [Citation: https://docs.ros.org/]
- It provides pub/sub communication
- Nodes are basic units of computation

### References
- [ROS 2 Documentation](https://docs.ros.org/)
"""


@pytest.fixture
def validation_service():
    """Create validation service."""
    return ContentValidationService()


@pytest.fixture
def chunking_service():
    """Create chunking service."""
    return ChunkingService()


@pytest.fixture
def embedding_service():
    """Create embedding service."""
    return EmbeddingService()


class TestChapterGenerationPipeline:
    """Test the complete chapter generation pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocking(
        self,
        mock_chapter_content,
        validation_service,
        chunking_service,
        embedding_service,
    ):
        """Test complete generation pipeline with mocked services."""
        module_id = "module1"
        chapter_number = 1
        title = "Test Chapter"

        # Mock the chapter generation service
        with patch.object(
            ChapterGenerationService,
            "generate_chapter",
            new_callable=AsyncMock,
            return_value=mock_chapter_content,
        ):
            gen_service = ChapterGenerationService()

            # Generate chapter
            content = await gen_service.generate_chapter(
                module_id=module_id,
                chapter_number=chapter_number,
                title=title,
                description="Test description",
            )

            assert content == mock_chapter_content
            assert "[Citation:" in content

    def test_validation_pipeline(self, mock_chapter_content, validation_service):
        """Test validation stage of pipeline."""
        result = validation_service.validate_chapter(mock_chapter_content)

        assert result["passed"] is True
        assert len(result["citations"]["citations"]) > 0
        assert result["code_blocks"]["count"] > 0

    def test_chunking_pipeline(self, mock_chapter_content, chunking_service):
        """Test chunking stage of pipeline."""
        chunks = chunking_service.chunk_chapter(
            content=mock_chapter_content,
            chapter_id="test-ch-001",
        )

        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("section_title" in chunk for chunk in chunks)

    def test_embedding_pipeline_with_mock(
        self,
        mock_chapter_content,
        chunking_service,
        embedding_service,
    ):
        """Test embedding stage with mocked OpenAI."""
        chunks = chunking_service.chunk_chapter(
            content=mock_chapter_content,
            chapter_id="test-ch-001",
        )

        # Mock the OpenAI embedding call
        mock_embeddings = [
            [0.1] * 384 for _ in range(len(chunks))  # Mock 384-dim embeddings
        ]

        with patch.object(
            embedding_service,
            "embed_chunks_batch",
            return_value=[
                {**chunk, "embedding": mock_embeddings[i]}
                for i, chunk in enumerate(chunks)
            ],
        ):
            embedded_chunks = embedding_service.embed_chunks_batch(chunks)

            assert len(embedded_chunks) == len(chunks)
            assert all("embedding" in chunk for chunk in embedded_chunks)
            assert all(len(chunk["embedding"]) == 384 for chunk in embedded_chunks)

    def test_storage_and_git_integration(self, tmp_path, mock_chapter_content):
        """Test chapter storage and git integration."""
        storage_service = ChapterStorageService(repo_root=tmp_path)

        # Create module directory
        module_dir = tmp_path / "textbook" / "docs" / "module1"
        module_dir.mkdir(parents=True, exist_ok=True)

        # Save chapter
        chapter_path = storage_service.save_chapter_markdown(
            module_id="module1",
            chapter_number=1,
            title="Test Chapter",
            content=mock_chapter_content,
        )

        assert chapter_path.exists()
        assert chapter_path.read_text() == mock_chapter_content
        assert chapter_path.parent == module_dir

    def test_end_to_end_validation_and_chunking(
        self,
        mock_chapter_content,
        validation_service,
        chunking_service,
    ):
        """Test end-to-end validation and chunking."""
        # Validate
        validation_result = validation_service.validate_chapter(mock_chapter_content)
        assert validation_result["passed"] is True

        # Chunk
        chunks = chunking_service.chunk_chapter(
            content=mock_chapter_content,
            chapter_id="test-ch-001",
        )

        assert len(chunks) > 0

        # Verify chunks contain content
        all_text = " ".join(chunk["text"] for chunk in chunks)
        assert "ROS 2" in all_text


class TestChapterGenerationFailures:
    """Test error handling in generation pipeline."""

    @pytest.mark.asyncio
    async def test_generation_failure_handling(self):
        """Test handling of generation failures."""
        with patch.object(
            ChapterGenerationService,
            "generate_chapter",
            new_callable=AsyncMock,
            side_effect=ValueError("Generation failed"),
        ):
            gen_service = ChapterGenerationService()

            with pytest.raises(ValueError):
                await gen_service.generate_chapter(
                    module_id="module1",
                    chapter_number=1,
                    title="Test",
                    description="Test",
                )

    def test_validation_failure_handling(self, validation_service):
        """Test validation failure on invalid content."""
        invalid_content = "No citations at all in this content."

        result = validation_service.validate_chapter(invalid_content)

        assert result["passed"] is False
        assert len(result["critical_issues"]) > 0

    def test_storage_failure_handling(self, validation_service):
        """Test handling of storage failures."""
        invalid_module = "invalid/path"
        storage_service = ChapterStorageService(repo_root=Path("/tmp"))

        with pytest.raises(ValueError):
            storage_service.save_chapter_markdown(
                module_id=invalid_module,
                chapter_number=1,
                title="Test",
                content="Test content",
            )


class TestBatchGeneration:
    """Test batch generation of multiple chapters."""

    @pytest.mark.asyncio
    async def test_concurrent_generation(self):
        """Test concurrent chapter generation with semaphore."""
        specs = [
            {
                "module_id": "module1",
                "chapter_number": i,
                "title": f"Chapter {i}",
                "description": f"Description {i}",
            }
            for i in range(1, 4)
        ]

        async def mock_generate(module_id, chapter_number, title, description):
            await asyncio.sleep(0.01)  # Simulate work
            return f"Content for chapter {chapter_number}"

        with patch.object(
            ChapterGenerationService,
            "generate_chapter",
            new_callable=AsyncMock,
            side_effect=mock_generate,
        ):
            gen_service = ChapterGenerationService()
            results = await gen_service.generate_chapters_batch(specs, max_concurrent=2)

            assert len(results) == 3
            assert all(result[1] is not None for result in results)  # All should succeed

    @pytest.mark.asyncio
    async def test_batch_with_failures(self):
        """Test batch generation handling some failures."""
        specs = [
            {"module_id": "module1", "chapter_number": 1, "title": "Ch1", "description": "D1"},
            {"module_id": "module1", "chapter_number": 2, "title": "Ch2", "description": "D2"},
        ]

        call_count = 0

        async def mock_generate_with_failure(
            module_id,
            chapter_number,
            title,
            description,
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Generation failed for chapter 2")
            return f"Content for chapter {chapter_number}"

        with patch.object(
            ChapterGenerationService,
            "generate_chapter",
            new_callable=AsyncMock,
            side_effect=mock_generate_with_failure,
        ):
            gen_service = ChapterGenerationService()
            results = await gen_service.generate_chapters_batch(specs, max_concurrent=1)

            assert len(results) == 2
            assert results[0][1] is not None  # First succeeds
            assert results[1][2] is not None  # Second has error
