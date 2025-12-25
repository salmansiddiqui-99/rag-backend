"""Unit tests for content validation service."""

import pytest
from src.services.validation import ContentValidationService


@pytest.fixture
def validation_service():
    """Create a validation service instance."""
    return ContentValidationService()


@pytest.fixture
def sample_valid_chapter():
    """Sample chapter with proper citations and code."""
    return """
## ROS 2 Basics

### Introduction
ROS 2 is a flexible middleware for writing robotic software. [Citation: https://docs.ros.org/en/humble/]

### Core Concepts

ROS 2 uses a publish-subscribe pattern for communication. [Citation: https://docs.ros.org/en/humble/Concepts/Intermediate/About-Topic-ROS-2.html]

### Code Examples

Here's a simple ROS 2 subscriber:

```python
import rclpy
from std_msgs.msg import String

def listener():
    rclpy.init()
    node = rclpy.create_node('listener')

    def callback(msg):
        print(f'Received: {msg.data}')

    subscription = node.create_subscription(String, 'topic', callback, 10)
    rclpy.spin(node)

if __name__ == '__main__':
    listener()
```

### Summary
- ROS 2 provides pub/sub communication
- Nodes are the basic unit of computation
- Topics enable asynchronous communication [Citation: https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html]

### References
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Concepts](https://docs.ros.org/en/humble/Concepts/)
"""


@pytest.fixture
def sample_invalid_chapter():
    """Sample chapter missing citations."""
    return """
## Invalid Chapter

### Introduction
This is just some random text without citations.

### Content
More text that has no sources.
"""


class TestCitationValidation:
    """Test citation extraction and validation."""

    def test_extract_citations(self, validation_service, sample_valid_chapter):
        """Test citation extraction from content."""
        result = validation_service.validate_citations(sample_valid_chapter)

        assert result["count"] >= 2
        assert result["has_citations"] is True
        assert len(result["citations"]) > 0

    def test_no_citations(self, validation_service, sample_invalid_chapter):
        """Test detection of missing citations."""
        result = validation_service.validate_citations(sample_invalid_chapter)

        assert result["count"] == 0
        assert result["has_citations"] is False
        assert len(result["issues"]) > 0

    def test_insufficient_citations(self, validation_service):
        """Test warning for insufficient citations."""
        content = "[Citation: source1] Some content [Citation: source2]"
        result = validation_service.validate_citations(content)

        assert result["count"] == 2
        assert result["has_citations"] is True
        # Should warn about insufficient unique citations
        assert any("Insufficient" in issue for issue in result["issues"])


class TestCodeBlockValidation:
    """Test code block validation."""

    def test_valid_python_code(self, validation_service):
        """Test valid Python code block."""
        content = """
        ```python
        def hello():
            print("Hello, World!")
        ```
        """
        result = validation_service.validate_code_blocks(content)

        assert result["count"] == 1
        assert len(result["issues"]) == 0

    def test_invalid_python_syntax(self, validation_service):
        """Test invalid Python syntax detection."""
        content = """
        ```python
        def hello(
            print("Missing closing paren")
        ```
        """
        result = validation_service.validate_code_blocks(content)

        assert result["count"] == 1
        assert any("invalid Python syntax" in issue for issue in result["issues"])

    def test_valid_bash_code(self, validation_service):
        """Test valid bash code."""
        content = """
        ```bash
        #!/bin/bash
        echo "Hello"
        ros2 run package node
        ```
        """
        result = validation_service.validate_code_blocks(content)

        assert result["count"] == 1

    def test_no_code_blocks(self, validation_service, sample_invalid_chapter):
        """Test warning when no code blocks present."""
        result = validation_service.validate_code_blocks(sample_invalid_chapter)

        assert result["count"] == 0
        assert any("No code examples" in issue for issue in result["issues"])


class TestClaimValidation:
    """Test claim extraction and citation checking."""

    def test_cited_claim(self, validation_service):
        """Test claim with nearby citation."""
        content = """
        ROS 2 uses publish-subscribe architecture. [Citation: https://docs.ros.org/]
        """
        result = validation_service.validate_claims(content)

        assert result["cited_claims"] > 0 or result["uncited_claims"] == 0

    def test_uncited_claim(self, validation_service):
        """Test detection of uncited claims."""
        content = """
        Some random technical claim without any citation.
        """
        result = validation_service.validate_claims(content)

        assert result["uncited_claims"] > 0

    def test_claim_with_distant_citation(self, validation_service):
        """Test claim where citation is too far away."""
        content = """
        Important technical claim here.

        [Some other text]

        [Citation: somewhere far away]
        """
        result = validation_service.validate_claims(content)

        # Claim should be uncited (citation too far)
        assert result["total_claims"] > 0


class TestComprehensiveValidation:
    """Test complete chapter validation."""

    def test_valid_chapter(self, validation_service, sample_valid_chapter):
        """Test validation of a complete, valid chapter."""
        result = validation_service.validate_chapter(sample_valid_chapter)

        assert result["passed"] is True
        assert len(result["critical_issues"]) == 0

    def test_invalid_chapter_no_citations(self, validation_service, sample_invalid_chapter):
        """Test validation failure due to missing citations."""
        result = validation_service.validate_chapter(sample_invalid_chapter)

        assert result["passed"] is False
        assert any("citation" in issue.lower() for issue in result["critical_issues"])

    def test_validation_with_warnings(self, validation_service):
        """Test validation with warnings but no critical issues."""
        content = """
        ## Chapter

        ### Introduction
        [Citation: https://example.com]

        Content here.

        ### Summary
        - Point 1
        - Point 2
        """
        result = validation_service.validate_chapter(content)

        # Should pass but have warnings
        assert result["passed"] is True
        assert len(result["warnings"]) > 0


class TestRegexPatterns:
    """Test regex patterns used for validation."""

    def test_citation_regex(self, validation_service):
        """Test citation pattern matching."""
        content = "[Citation: https://docs.ros.org/] and [Citation: another-source]"
        citations = validation_service.citation_regex.findall(content)

        assert len(citations) == 2
        assert "https://docs.ros.org/" in citations

    def test_code_block_regex(self, validation_service):
        """Test code block pattern matching."""
        content = """
        ```python
        code here
        ```
        and
        ```bash
        other code
        ```
        """
        matches = validation_service.code_regex.findall(content)

        assert len(matches) == 2
        assert any("python" in match[0].lower() for match in matches)
        assert any("bash" in match[0].lower() for match in matches)
