"""
Chapter Storage Service - Handles chapter persistence to filesystem and git.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChapterStorageService:
    """Manages chapter persistence to filesystem and git versioning."""

    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize storage service.

        Args:
            repo_root: Root path of the git repository (defaults to project root)
        """
        self._repo_root = repo_root
        self._docs_root = None

    @property
    def repo_root(self) -> Path:
        """Lazily resolve repo root on first access."""
        if self._repo_root is None:
            # Try to find git root, but don't fail if git isn't available (e.g., Railway)
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                    timeout=5,
                )
                if result.returncode == 0:
                    self._repo_root = Path(result.stdout.strip())
                else:
                    self._repo_root = Path.cwd()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # git not available or timeout - use current directory
                self._repo_root = Path.cwd()
        return self._repo_root

    @property
    def docs_root(self) -> Path:
        """Lazily resolve docs root on first access."""
        if self._docs_root is None:
            self._docs_root = self.repo_root / "textbook" / "docs"
        return self._docs_root

    def save_chapter_markdown(
        self,
        module_id: str,
        chapter_number: int,
        title: str,
        content: str,
    ) -> Path:
        """
        Save chapter Markdown to filesystem.

        Args:
            module_id: Module identifier (e.g., "module1")
            chapter_number: Chapter number (1-12)
            title: Chapter title (used for filename)
            content: Markdown content

        Returns:
            Path to saved file

        Raises:
            IOError: If file cannot be written
            ValueError: If paths are invalid
        """
        # Extract module number from module_id (e.g., "module1" -> 1)
        try:
            module_num = int("".join(filter(str.isdigit, module_id)))
        except (ValueError, IndexError):
            raise ValueError(f"Invalid module_id: {module_id}")

        if not (1 <= chapter_number <= 12):
            raise ValueError(f"Invalid chapter_number: {chapter_number}")

        # Create module directory
        module_dir = self.docs_root / f"module{module_num}"
        module_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: 01-chapter-title.md
        # Sanitize title for filename
        sanitized_title = (
            title.lower()
            .replace(" ", "-")
            .replace(":", "")
            .replace("&", "and")
            .replace("/", "-")
        )
        # Keep first 50 chars
        sanitized_title = sanitized_title[:50].rstrip("-")

        filename = f"{chapter_number:02d}-{sanitized_title}.md"
        filepath = module_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved chapter to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed to write chapter file {filepath}: {str(e)}")
            raise IOError(f"Cannot write chapter to {filepath}: {str(e)}")

    def commit_to_git(
        self,
        files: List[Path],
        message: str,
    ) -> bool:
        """
        Commit files to git with a descriptive message.

        Args:
            files: List of file paths to commit
            message: Commit message

        Returns:
            True if commit succeeded, False otherwise
        """
        if not files:
            logger.warning("No files to commit")
            return False

        try:
            # Convert to relative paths from repo root
            relative_files = []
            for file in files:
                if isinstance(file, str):
                    file = Path(file)
                try:
                    rel_path = file.relative_to(self.repo_root)
                    relative_files.append(str(rel_path))
                except ValueError:
                    # File is not in repo
                    relative_files.append(str(file))

            # Stage files
            stage_cmd = ["git", "-C", str(self.repo_root), "add"] + relative_files
            result = subprocess.run(stage_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"git add failed: {result.stderr}")
                return False

            # Commit
            commit_cmd = [
                "git",
                "-C",
                str(self.repo_root),
                "commit",
                "-m",
                message,
            ]
            result = subprocess.run(commit_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                # Commit may fail if nothing changed; not necessarily an error
                logger.warning(f"git commit: {result.stderr}")
                return False

            logger.info(f"Committed {len(relative_files)} files: {message}")
            return True

        except Exception as e:
            logger.error(f"Git operation failed: {str(e)}")
            return False

    def chapter_exists(self, module_id: str, chapter_number: int) -> bool:
        """
        Check if chapter file already exists.

        Args:
            module_id: Module identifier
            chapter_number: Chapter number

        Returns:
            True if chapter file exists
        """
        try:
            module_num = int("".join(filter(str.isdigit, module_id)))
            module_dir = self.docs_root / f"module{module_num}"

            # Check for any file matching the chapter number
            for file in module_dir.glob(f"{chapter_number:02d}-*.md"):
                return True
            return False
        except Exception:
            return False

    def get_chapter_path(self, module_id: str, chapter_number: int) -> Optional[Path]:
        """
        Get the path to an existing chapter file.

        Args:
            module_id: Module identifier
            chapter_number: Chapter number

        Returns:
            Path to chapter file if exists, None otherwise
        """
        try:
            module_num = int("".join(filter(str.isdigit, module_id)))
            module_dir = self.docs_root / f"module{module_num}"

            # Find file matching the chapter number
            matches = list(module_dir.glob(f"{chapter_number:02d}-*.md"))
            if matches:
                return matches[0]
            return None
        except Exception:
            return None

    def list_chapters(self) -> List[Path]:
        """
        List all chapter files in the docs directory.

        Returns:
            List of paths to chapter Markdown files
        """
        if not self.docs_root.exists():
            return []

        chapters = []
        for module_dir in self.docs_root.glob("module*"):
            if module_dir.is_dir():
                chapters.extend(module_dir.glob("*.md"))

        return sorted(chapters)
