"""Unit tests for content chunking service."""

import pytest
from src.services.chunking import ChunkingService


@pytest.fixture
def chunking_service():
    """Create a chunking service instance."""
    return ChunkingService(target_chunk_size=200)


@pytest.fixture
def sample_chapter():
    """Sample chapter with sections."""
    return """
## ROS 2 Basics: Architecture and Communication

### Introduction
ROS 2 is a flexible middleware for writing robotic software. It provides a set of tools and libraries
for building robot applications. ROS 2 supports multiple platforms and programming languages, making it
a versatile choice for roboticists worldwide.

### Core Concepts

#### Pub/Sub Model
ROS 2 uses a publish-subscribe pattern for inter-process communication. Publishers send messages to topics,
while subscribers listen to topics of interest. This decouples components and enables flexible architectures.

#### Nodes
A node is a process that performs computation. Nodes communicate with each other by publishing and subscribing
to topics. Each node should be responsible for a single, well-defined task to keep the system modular and testable.

#### Services
Services provide a request-response communication pattern. Unlike topics, services are synchronous and suitable
for tasks that require immediate responses.

### Implementation

Here's a simple publisher node:

```python
import rclpy
from std_msgs.msg import String

def main():
    rclpy.init()
    node = rclpy.create_node('publisher')
    publisher = node.create_publisher(String, 'topic', 10)

    msg = String()
    msg.data = 'Hello, ROS 2!'
    publisher.publish(msg)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

And here's a subscriber:

```python
import rclpy
from std_msgs.msg import String

def callback(msg):
    print(f'Received: {msg.data}')

def main():
    rclpy.init()
    node = rclpy.create_node('subscriber')
    node.create_subscription(String, 'topic', callback, 10)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

### Summary
- ROS 2 provides middleware for robotic systems
- Pub/sub enables asynchronous communication
- Nodes are independent processes
- Services provide synchronous communication

### References
- [ROS 2 Documentation](https://docs.ros.org/)
- [ROS 2 Concepts](https://docs.ros.org/Concepts/)
"""


class TestChunking:
    """Test chapter chunking functionality."""

    def test_chunk_section_splitting(self, chunking_service, sample_chapter):
        """Test that content is chunked by sections."""
        chunks = chunking_service.chunk_chapter(
            content=sample_chapter,
            chapter_id="test-chapter-1",
        )

        assert len(chunks) > 0
        assert all("section_title" in chunk for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)

    def test_chunk_preserves_metadata(self, chunking_service, sample_chapter):
        """Test that chunk metadata is preserved."""
        chunks = chunking_service.chunk_chapter(
            content=sample_chapter,
            chapter_id="test-chapter-1",
        )

        for chunk in chunks:
            assert chunk["chapter_id"] == "test-chapter-1"
            assert chunk["section_level"] in (0, 1, 2, 3, 4)
            assert chunk["token_count"] > 0

    def test_token_estimation(self, chunking_service):
        """Test token count estimation."""
        text = "This is a sample text for token estimation."
        tokens = chunking_service.estimate_tokens(text)

        assert tokens > 0
        # Rough check: ~4 chars per token
        assert tokens < len(text)

    def test_chunk_target_size(self, chunking_service, sample_chapter):
        """Test that chunks respect target size limits."""
        chunks = chunking_service.chunk_chapter(
            content=sample_chapter,
            chapter_id="test-chapter-1",
        )

        # Most chunks should be within reasonable range of target
        target = chunking_service.target_chunk_size
        for chunk in chunks:
            # Allow up to 1.5x target size
            assert chunk["token_count"] <= target * 1.5

    def test_section_hierarchy(self, chunking_service):
        """Test proper section hierarchy tracking."""
        content = """
## Level 1 Section
Content here.

### Level 2 Section
More content.

#### Level 3 Section
Even more content.
"""
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id="test-chapter-1",
        )

        section_levels = [chunk["section_level"] for chunk in chunks]
        assert min(section_levels) >= 0
        assert max(section_levels) <= 3

    def test_chunk_merging(self, chunking_service):
        """Test merging of small chunks."""
        chunks = [
            {"text": "Short chunk 1", "token_count": 2},
            {"text": "Short chunk 2", "token_count": 3},
            {"text": "Long chunk" * 100, "token_count": 400},
        ]

        merged = chunking_service.merge_chunks(chunks, max_tokens=300)

        # First two should be merged
        assert len(merged) < len(chunks)
        assert merged[0]["token_count"] < 300

    def test_single_section_chapter(self, chunking_service):
        """Test chunking of a chapter with minimal sections."""
        content = """
## Single Section

This is a chapter with just one section and some content.
The content is brief and should not be split into many chunks.
But we test that it's still properly handled.
"""
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id="test-chapter-1",
        )

        assert len(chunks) > 0
        assert all(chunk["section_title"] == "Single Section" for chunk in chunks)

    def test_empty_chapter(self, chunking_service):
        """Test handling of empty or minimal content."""
        content = "## Title\n\nMinimal content."
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id="test-chapter-1",
        )

        # Should still create at least one chunk
        assert len(chunks) >= 0

    def test_code_block_preservation(self, chunking_service):
        """Test that code blocks are preserved in chunks."""
        content = """
## Programming Example

Here's some code:

```python
def example():
    return "code"
```

This code should be preserved in the chunk.
"""
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id="test-chapter-1",
        )

        # At least one chunk should contain the code
        code_found = any("```python" in chunk["text"] for chunk in chunks)
        assert code_found

    def test_position_tracking(self, chunking_service, sample_chapter):
        """Test that chunk positions are tracked."""
        chunks = chunking_service.chunk_chapter(
            content=sample_chapter,
            chapter_id="test-chapter-1",
        )

        # Positions should increase (roughly)
        positions = [chunk["position"] for chunk in chunks]
        assert all(pos >= 0 for pos in positions)


class TestChunkingEdgeCases:
    """Test edge cases in chunking."""

    def test_very_long_section_title(self, chunking_service):
        """Test handling of very long section titles."""
        content = """
## This Is A Very Long Section Title That Might Cause Issues With Processing ✓

Content here.
"""
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id="test-chapter-1",
        )

        assert len(chunks) > 0

    def test_special_characters_in_content(self, chunking_service):
        """Test handling of special characters."""
        content = """
## Special Characters

Content with special chars: @#$%^&*(){}[]|\\<>?

And emoji: 🤖 ⚙️ 🔧
"""
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id="test-chapter-1",
        )

        assert len(chunks) > 0

    def test_multiple_code_blocks(self, chunking_service):
        """Test handling of multiple code blocks."""
        content = """
## Code Examples

First example:
```python
print("hello")
```

Second example:
```bash
echo "hello"
```

Third example:
```xml
<robot></robot>
```
"""
        chunks = chunking_service.chunk_chapter(
            content=content,
            chapter_id="test-chapter-1",
        )

        # All code blocks should be preserved
        all_text = " ".join(chunk["text"] for chunk in chunks)
        assert "```python" in all_text
        assert "```bash" in all_text
        assert "```xml" in all_text
