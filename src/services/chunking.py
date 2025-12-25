"""
Content Chunking Service - Splits chapters into sections for vector indexing.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ChunkingService:
    """Splits chapter content into semantic chunks for embedding and retrieval."""

    def __init__(self, target_chunk_size: int = 200, tokenizer: Any = None):
        """
        Initialize chunking service.

        Args:
            target_chunk_size: Target tokens per chunk (approximate)
            tokenizer: Optional tokenizer for accurate token counting (uses OpenAI by default)
        """
        self.target_chunk_size = target_chunk_size
        self.tokenizer = tokenizer
        self.header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text using rough heuristic.
        For production, use OpenAI's tokenizer.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))

        # Rough heuristic: 1 token ≈ 4 characters
        return len(text) // 4

    def chunk_chapter(
        self,
        content: str,
        chapter_id: str,
        section_title: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Split chapter content into semantic chunks by section headers.

        Args:
            content: Full chapter Markdown content
            chapter_id: Unique chapter identifier
            section_title: Optional parent section title

        Returns:
            List of chunk dicts with keys:
            - chapter_id: Chapter identifier
            - section_title: Section header
            - section_level: Header level (1-4)
            - text: Chunk text
            - token_count: Estimated tokens
            - position: Character position in original
        """
        chunks = []
        lines = content.split("\n")
        current_section = section_title
        current_section_level = 0
        current_chunk = []
        current_chunk_tokens = 0
        position = 0

        for line_idx, line in enumerate(lines):
            header_match = self.header_pattern.match(line)

            if header_match:
                # Found a header - save current chunk if non-empty
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text and current_chunk_tokens > 0:
                        chunks.append(
                            {
                                "chapter_id": chapter_id,
                                "section_title": current_section,
                                "section_level": current_section_level,
                                "text": chunk_text,
                                "token_count": current_chunk_tokens,
                                "position": position,
                            }
                        )

                # Start new section
                hashes, title = header_match.groups()
                current_section = title.strip()
                current_section_level = len(hashes)
                current_chunk = [line]
                current_chunk_tokens = self.estimate_tokens(line)
                position = len("\n".join(lines[:line_idx]))

            else:
                # Regular content line
                line_tokens = self.estimate_tokens(line)

                # Check if adding this line would exceed target
                if (current_chunk_tokens + line_tokens > self.target_chunk_size * 1.5
                    and current_chunk):
                    # Save current chunk
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text and current_chunk_tokens > 0:
                        chunks.append(
                            {
                                "chapter_id": chapter_id,
                                "section_title": current_section,
                                "section_level": current_section_level,
                                "text": chunk_text,
                                "token_count": current_chunk_tokens,
                                "position": position,
                            }
                        )
                    # Start new chunk (but keep the section header)
                    if current_section:
                        current_chunk = [f"### {current_section}"]
                        current_chunk_tokens = self.estimate_tokens(current_chunk[0])
                    else:
                        current_chunk = []
                        current_chunk_tokens = 0

                # Add line to current chunk
                current_chunk.append(line)
                current_chunk_tokens += line_tokens
                position += len(line) + 1  # +1 for newline

        # Save final chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text and current_chunk_tokens > 0:
                chunks.append(
                    {
                        "chapter_id": chapter_id,
                        "section_title": current_section,
                        "section_level": current_section_level,
                        "text": chunk_text,
                        "token_count": current_chunk_tokens,
                        "position": position,
                    }
                )

        logger.info(
            f"Chunked chapter {chapter_id} into {len(chunks)} chunks "
            f"(avg {sum(c['token_count'] for c in chunks) // max(len(chunks), 1)} tokens)"
        )

        return chunks

    def merge_chunks(self, chunks: List[Dict[str, Any]], max_tokens: int = 300) -> List[Dict[str, Any]]:
        """
        Merge small chunks to reach minimum token threshold.

        Args:
            chunks: List of chunks from chunk_chapter()
            max_tokens: Maximum tokens per merged chunk

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged = []
        current_merged = None

        for chunk in chunks:
            if current_merged is None:
                current_merged = chunk.copy()
            elif current_merged["token_count"] + chunk["token_count"] <= max_tokens:
                # Merge with current
                current_merged["text"] += "\n\n" + chunk["text"]
                current_merged["token_count"] += chunk["token_count"]
            else:
                # Save current and start new
                merged.append(current_merged)
                current_merged = chunk.copy()

        # Don't forget the last chunk
        if current_merged:
            merged.append(current_merged)

        logger.info(f"Merged chunks: {len(chunks)} → {len(merged)}")
        return merged
