"""
Chapter Generation Service - Orchestrates Claude Code subagent for chapter creation.
"""

import asyncio
import logging
from typing import Optional
from pathlib import Path

from anthropic import Anthropic

from src.models.chapter import Chapter, ChapterCreate
from src.config import settings

logger = logging.getLogger(__name__)


class ChapterGenerationService:
    """Orchestrates chapter generation using Claude Code subagent API."""

    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.prompt_template_path = Path(__file__).parent / "chapter_gen_prompt.txt"
        self.max_retries = 3
        self.base_delay = 1  # seconds

    def _load_prompt_template(self) -> str:
        """Load the chapter generation prompt template."""
        if not self.prompt_template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self.prompt_template_path}")
        with open(self.prompt_template_path, "r") as f:
            return f.read()

    def _format_prompt(
        self,
        module_id: str,
        chapter_number: int,
        title: str,
        description: str,
    ) -> str:
        """Format the chapter generation prompt with specific context."""
        template = self._load_prompt_template()
        return template.format(
            module_id=module_id,
            chapter_number=chapter_number,
            title=title,
            description=description,
        )

    async def generate_chapter(
        self,
        module_id: str,
        chapter_number: int,
        title: str,
        description: str,
    ) -> str:
        """
        Generate a chapter using Claude Code subagent.

        Args:
            module_id: Unique identifier for the module
            chapter_number: Chapter number (1-12)
            title: Chapter title
            description: Brief chapter description

        Returns:
            Markdown content of the generated chapter

        Raises:
            ValueError: If chapter generation fails after retries
            FileNotFoundError: If prompt template not found
        """
        prompt = self._format_prompt(module_id, chapter_number, title, description)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Generating chapter {chapter_number} ({title}), attempt {attempt}/{self.max_retries}"
                )

                # Use Claude Opus for high-quality content generation
                message = self.client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=8000,  # Allow up to ~5000 tokens of content
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                content = message.content[0].text

                # Validate that response is Markdown (starts with ##)
                if not content.strip().startswith("##"):
                    raise ValueError("Generated content does not start with Markdown heading")

                logger.info(f"Successfully generated chapter {chapter_number}")
                return content

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for chapter {chapter_number}: {str(e)}"
                )

                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to generate chapter {chapter_number} after {self.max_retries} attempts")
                    raise ValueError(
                        f"Chapter generation failed after {self.max_retries} retries: {str(e)}"
                    )

        raise ValueError("Unexpected error in chapter generation")

    async def generate_chapters_batch(
        self,
        chapters_spec: list[dict],
        max_concurrent: int = 2,
    ) -> list[tuple[str, Optional[str], Optional[str]]]:
        """
        Generate multiple chapters in parallel with concurrency control.

        Args:
            chapters_spec: List of dicts with keys: module_id, chapter_number, title, description
            max_concurrent: Maximum concurrent generation tasks (for token efficiency)

        Returns:
            List of tuples (chapter_number, content, error)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(spec: dict) -> tuple[str, Optional[str], Optional[str]]:
            async with semaphore:
                try:
                    content = await self.generate_chapter(
                        module_id=spec["module_id"],
                        chapter_number=spec["chapter_number"],
                        title=spec["title"],
                        description=spec["description"],
                    )
                    return spec["chapter_number"], content, None
                except Exception as e:
                    logger.error(f"Error generating chapter {spec['chapter_number']}: {str(e)}")
                    return spec["chapter_number"], None, str(e)

        results = await asyncio.gather(
            *[generate_with_semaphore(spec) for spec in chapters_spec],
            return_exceptions=False,
        )

        return results
