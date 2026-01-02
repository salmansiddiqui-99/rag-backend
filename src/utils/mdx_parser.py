"""
MDX file parsing utilities for extracting plain text from book chapters.

Note: This is a simplified implementation that uses regex for MDX parsing.
For production, consider using unified + remark ecosystem (JavaScript/Node.js).
"""
import re
from typing import Dict, Any, Optional
from pathlib import Path


def parse_mdx(file_path: str) -> Dict[str, Any]:
    """
    Parse MDX file and extract plain text content with metadata.

    Args:
        file_path: Path to MDX file

    Returns:
        Dictionary with:
            - text: Extracted plain text content
            - chapter_title: Chapter title from frontmatter or filename
            - headings: List of section headings found
            - metadata: Frontmatter data if present

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MDX file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract frontmatter (YAML between ---)
    frontmatter = {}
    content_without_frontmatter = content

    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        content_without_frontmatter = content[frontmatter_match.end():]

        # Simple YAML parsing for title
        title_match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', frontmatter_text, re.MULTILINE)
        if title_match:
            frontmatter['title'] = title_match.group(1)

    # Extract chapter title (from frontmatter or first H1)
    chapter_title = frontmatter.get('title')
    if not chapter_title:
        h1_match = re.search(r'^#\s+(.+)$', content_without_frontmatter, re.MULTILINE)
        if h1_match:
            chapter_title = h1_match.group(1).strip()
        else:
            # Fallback to filename
            chapter_title = path.stem.replace('-', ' ').replace('_', ' ').title()

    # Extract all headings (for section tracking)
    headings = []
    for match in re.finditer(r'^#{1,6}\s+(.+)$', content_without_frontmatter, re.MULTILINE):
        heading_text = match.group(1).strip()
        headings.append(heading_text)

    # Remove MDX components and JSX syntax
    text = content_without_frontmatter
    text = re.sub(r'<[^>]+>', '', text)  # Remove JSX tags
    text = re.sub(r'import\s+.*?from\s+["\'].*?["\'];?', '', text)  # Remove imports
    text = re.sub(r'export\s+.*?;', '', text)  # Remove exports

    # Remove code blocks (preserve content but mark as code)
    text = re.sub(r'```[\s\S]*?```', '[CODE BLOCK OMITTED]', text)
    text = re.sub(r'`[^`]+`', '', text)  # Remove inline code

    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '[IMAGE OMITTED]', text)

    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove markdown formatting
    text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)  # Bold/italic
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Heading markers
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)  # List markers
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = text.strip()

    return {
        'text': text,
        'chapter_title': chapter_title,
        'headings': headings,
        'metadata': frontmatter
    }


def extract_section_heading(text: str, chunk_start_pos: int, all_headings: list[str]) -> Optional[str]:
    """
    Determine which section heading a chunk belongs to.

    Args:
        text: Full document text
        chunk_start_pos: Starting position of chunk in text
        all_headings: List of all headings in document

    Returns:
        Section heading name or None
    """
    # Find the last heading before chunk_start_pos
    text_before_chunk = text[:chunk_start_pos]

    for heading in reversed(all_headings):
        if heading in text_before_chunk:
            return heading

    return None
