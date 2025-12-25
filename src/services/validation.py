"""
Content Validation Service - Validates chapter content for citations, claims, and code syntax.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ContentValidationService:
    """Validates chapter content for quality, citations, and code validity."""

    # Regex patterns
    CITATION_PATTERN = r"\[Citation:\s*([^\]]+)\]"
    CODE_BLOCK_PATTERN = r"```(\w+)?\n([\s\S]*?)```"
    CLAIM_PATTERN = r"^(?![\s#\[\(]+)([A-Z][\w\s\-,;:\.]+[.!?])(?=\s|$)"

    def __init__(self):
        self.citation_regex = re.compile(self.CITATION_PATTERN, re.IGNORECASE)
        self.code_regex = re.compile(self.CODE_BLOCK_PATTERN)
        self.claim_regex = re.compile(self.CLAIM_PATTERN, re.MULTILINE)

    def validate_citations(self, content: str) -> Dict[str, Any]:
        """
        Extract and validate citations in content.

        Args:
            content: Markdown content to validate

        Returns:
            Dict with keys:
            - citations: List of found citation sources
            - count: Number of citations found
            - has_citations: Boolean (True if >= 1 citations)
            - issues: List of validation issues
        """
        citations = self.citation_regex.findall(content)
        unique_citations = list(set(citations))

        issues = []
        if len(citations) == 0:
            issues.append("No citations found in content")

        if len(unique_citations) < 3:
            issues.append(
                f"Insufficient unique citations ({len(unique_citations)}). Recommended: >= 5"
            )

        return {
            "citations": unique_citations,
            "count": len(citations),
            "has_citations": len(citations) > 0,
            "issues": issues,
        }

    def validate_code_blocks(self, content: str) -> Dict[str, Any]:
        """
        Validate code blocks for syntax and structure.

        Args:
            content: Markdown content to validate

        Returns:
            Dict with keys:
            - code_blocks: List of found code blocks (lang, code)
            - count: Number of code blocks found
            - issues: List of validation issues
        """
        code_matches = self.code_regex.findall(content)
        code_blocks = [{"language": lang or "plain", "code": code} for lang, code in code_matches]

        issues = []

        if len(code_blocks) == 0:
            issues.append("No code examples found in content")

        for i, block in enumerate(code_blocks):
            # Check for Python syntax
            if block["language"].lower() in ("python", "py"):
                if not self._validate_python_syntax(block["code"]):
                    issues.append(
                        f"Code block {i+1} ({block['language']}) has invalid Python syntax"
                    )

            # Check for bash syntax
            if block["language"].lower() in ("bash", "sh"):
                if not self._validate_bash_syntax(block["code"]):
                    issues.append(
                        f"Code block {i+1} ({block['language']}) has invalid bash syntax"
                    )

            # Check for URDF XML
            if block["language"].lower() in ("xml", "urdf"):
                if not self._validate_xml_syntax(block["code"]):
                    issues.append(
                        f"Code block {i+1} ({block['language']}) has invalid XML syntax"
                    )

        return {
            "code_blocks": code_blocks,
            "count": len(code_blocks),
            "issues": issues,
        }

    def validate_claims(self, content: str) -> Dict[str, Any]:
        """
        Extract claims and flag those without nearby citations.

        Args:
            content: Markdown content to validate

        Returns:
            Dict with keys:
            - claims: List of extracted claims
            - uncited_claims: List of claims without nearby citations
            - issues: List of validation issues
        """
        lines = content.split("\n")
        claims = []
        uncited_claims = []

        for i, line in enumerate(lines):
            # Skip headers, code blocks, and empty lines
            if line.startswith("#") or line.startswith("```") or line.strip() == "":
                continue

            # Extract potential claims (sentences starting with capital letter)
            matches = self.claim_regex.findall(line)
            for match in matches:
                claim = {
                    "text": match[:100],  # First 100 chars
                    "line_number": i + 1,
                    "line_text": line[:150],
                }

                # Check for nearby citation (within 3 lines)
                has_nearby_citation = False
                for j in range(max(0, i - 3), min(len(lines), i + 4)):
                    if "[Citation:" in lines[j]:
                        has_nearby_citation = True
                        break

                if has_nearby_citation:
                    claims.append(claim)
                else:
                    uncited_claims.append(claim)

        issues = []
        if len(uncited_claims) > 0:
            issues.append(
                f"{len(uncited_claims)} claims found without nearby citations (within 3 lines)"
            )

        return {
            "total_claims": len(claims) + len(uncited_claims),
            "cited_claims": len(claims),
            "uncited_claims": uncited_claims,
            "issues": issues,
        }

    def _validate_python_syntax(self, code: str) -> bool:
        """Check Python syntax validity."""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def _validate_bash_syntax(self, code: str) -> bool:
        """Basic bash syntax validation (simplified)."""
        # Check for balanced braces, quotes
        issues = 0

        # Count braces
        if code.count("{") != code.count("}"):
            issues += 1
        if code.count("[") != code.count("]"):
            issues += 1

        # Count quotes
        if code.count('"') % 2 != 0:
            issues += 1
        if code.count("'") % 2 != 0:
            issues += 1

        return issues == 0

    def _validate_xml_syntax(self, code: str) -> bool:
        """Basic XML syntax validation."""
        try:
            import xml.etree.ElementTree as ET

            ET.fromstring(code)
            return True
        except Exception:
            return False

    def validate_chapter(self, content: str) -> Dict[str, Any]:
        """
        Perform comprehensive validation on chapter content.

        Args:
            content: Full chapter Markdown content

        Returns:
            Dict with keys:
            - passed: Boolean (True if all critical checks pass)
            - citations: Citation validation result
            - code_blocks: Code block validation result
            - claims: Claim validation result
            - critical_issues: List of blocking issues
            - warnings: List of non-blocking issues
        """
        citations = self.validate_citations(content)
        code_blocks = self.validate_code_blocks(content)
        claims = self.validate_claims(content)

        # Aggregate issues
        critical_issues = []
        warnings = []

        # Critical: must have citations
        if not citations["has_citations"]:
            critical_issues.append("No citations found in content")

        # Warning: should have multiple unique citations
        if citations["count"] < 5:
            warnings.extend(citations["issues"])

        # Warning: should have code examples
        if code_blocks["count"] < 2:
            warnings.append("Recommended: 2+ code examples per chapter")

        warnings.extend(code_blocks["issues"])
        warnings.extend(claims["issues"])

        passed = len(critical_issues) == 0

        return {
            "passed": passed,
            "citations": citations,
            "code_blocks": code_blocks,
            "claims": claims,
            "critical_issues": critical_issues,
            "warnings": warnings,
        }
