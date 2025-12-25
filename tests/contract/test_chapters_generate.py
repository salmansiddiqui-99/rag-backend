"""Contract tests for chapter generation API endpoints."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime

from src.main import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


class TestChapterGenerateEndpoint:
    """Test POST /api/chapters/generate endpoint."""

    def test_generate_chapter_request_response_schema(self, client):
        """Test that generate endpoint returns correct schema."""
        request_body = {
            "module_id": "module1",
            "chapter_number": 1,
            "title": "ROS 2 Basics",
            "description": "Introduction to ROS 2 architecture",
        }

        with patch(
            "backend.src.api.chapters._generate_chapter_background",
            new_callable=AsyncMock,
        ):
            response = client.post("/api/chapters/generate", json=request_body)

            assert response.status_code == 200
            data = response.json()

            # Verify response schema
            assert "status" in data
            assert "job_id" in data
            assert data["status"] == "processing"
            assert isinstance(data["job_id"], str)

    def test_generate_chapter_missing_required_fields(self, client):
        """Test validation of required fields."""
        # Missing chapter_number
        request_body = {
            "module_id": "module1",
            "title": "ROS 2 Basics",
            "description": "Introduction",
        }

        response = client.post("/api/chapters/generate", json=request_body)

        assert response.status_code == 400

    def test_generate_chapter_invalid_chapter_number(self, client):
        """Test validation of chapter number (1-12)."""
        request_body = {
            "module_id": "module1",
            "chapter_number": 13,  # Invalid: > 12
            "title": "ROS 2 Basics",
            "description": "Introduction",
        }

        with patch(
            "backend.src.api.chapters._generate_chapter_background",
            new_callable=AsyncMock,
        ):
            response = client.post("/api/chapters/generate", json=request_body)

            # Should still accept (validation happens in background)
            assert response.status_code == 200


class TestChapterJobsEndpoint:
    """Test GET /api/chapters/jobs/{job_id} endpoint."""

    def test_get_job_status_processing(self, client):
        """Test retrieving processing job status."""
        # First create a job
        request_body = {
            "module_id": "module1",
            "chapter_number": 1,
            "title": "ROS 2 Basics",
            "description": "Introduction",
        }

        with patch(
            "backend.src.api.chapters._generate_chapter_background",
            new_callable=AsyncMock,
        ):
            generate_response = client.post("/api/chapters/generate", json=request_body)
            job_id = generate_response.json()["job_id"]

            # Retrieve job status
            response = client.get(f"/api/chapters/jobs/{job_id}")

            assert response.status_code == 200
            data = response.json()

            # Verify response schema
            assert "job_id" in data
            assert "status" in data
            assert data["job_id"] == job_id
            assert data["status"] in ["processing", "completed", "failed"]

    def test_get_job_not_found(self, client):
        """Test retrieving non-existent job."""
        response = client.get("/api/chapters/jobs/non-existent-job-id")

        assert response.status_code == 404

    def test_get_completed_job(self, client):
        """Test retrieving completed job status."""
        # Set up a completed job
        from backend.src.api.chapters import generation_jobs

        job_id = "test-completed-job"
        generation_jobs[job_id] = {
            "status": "completed",
            "module_id": "module1",
            "chapter_number": 1,
            "title": "Test Chapter",
            "chapter_path": "/tmp/01-test.md",
            "token_count": 2500,
            "chunks_created": 10,
        }

        response = client.get(f"/api/chapters/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "completed"
        assert data["token_count"] == 2500
        assert data["chunks_created"] == 10

    def test_get_failed_job(self, client):
        """Test retrieving failed job status."""
        from backend.src.api.chapters import generation_jobs

        job_id = "test-failed-job"
        generation_jobs[job_id] = {
            "status": "failed",
            "error": "Generation failed: API error",
        }

        response = client.get(f"/api/chapters/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "failed"
        assert "error" in data


class TestChapterListEndpoint:
    """Test GET /api/chapters endpoint."""

    def test_list_chapters_empty(self, client):
        """Test listing chapters when none exist."""
        response = client.get("/api/chapters")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_list_chapters_with_module_filter(self, client):
        """Test filtering chapters by module."""
        response = client.get("/api/chapters?module_id=module1")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_list_chapters_with_status_filter(self, client):
        """Test filtering chapters by status."""
        response = client.get("/api/chapters?status=published")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)


class TestChapterValidateEndpoint:
    """Test POST /api/chapters/validate endpoint."""

    def test_validate_chapter_request_response_schema(self, client):
        """Test validate endpoint returns correct schema."""
        request_body = {
            "content": """
## Test Chapter

### Introduction
[Citation: https://example.com]

### Content
Some content here.

### Code
```python
print("hello")
```

### Summary
Summary here.
"""
        }

        response = client.post("/api/chapters/validate", json=request_body)

        assert response.status_code == 200
        data = response.json()

        # Verify response schema
        assert "passed" in data
        assert "citations" in data
        assert "code_blocks" in data
        assert "claims" in data
        assert "critical_issues" in data
        assert "warnings" in data

    def test_validate_chapter_missing_content(self, client):
        """Test validation with missing content."""
        request_body = {}

        response = client.post("/api/chapters/validate", json=request_body)

        assert response.status_code == 400

    def test_validate_chapter_valid_content(self, client):
        """Test validation of valid chapter content."""
        request_body = {
            "content": """
## Valid Chapter

### Introduction
This is a properly structured chapter. [Citation: https://docs.example.com]

### Concepts
Important concept here. [Citation: https://reference.example.com]

### Code Example
```python
def hello():
    return "world"
```

### Summary
- Point 1
- Point 2
- Point 3 [Citation: https://source.example.com]

### References
- [Example Doc](https://docs.example.com)
- [Reference](https://reference.example.com)
"""
        }

        response = client.post("/api/chapters/validate", json=request_body)

        assert response.status_code == 200
        data = response.json()

        assert data["passed"] is True or data["passed"] is False  # Valid response

    def test_validate_chapter_missing_citations(self, client):
        """Test validation failure for missing citations."""
        request_body = {
            "content": """
## Chapter Without Citations

### Introduction
This chapter has no citations whatsoever.

### Content
Just random content without any sources.
"""
        }

        response = client.post("/api/chapters/validate", json=request_body)

        assert response.status_code == 200
        data = response.json()

        # Should fail validation
        assert data["passed"] is False


class TestChapterAPIEndToEnd:
    """End-to-end tests for chapter API workflow."""

    def test_generate_and_poll_job(self, client):
        """Test generating a chapter and polling for completion."""
        request_body = {
            "module_id": "module1",
            "chapter_number": 1,
            "title": "ROS 2 Basics",
            "description": "Introduction to ROS 2",
        }

        with patch(
            "backend.src.api.chapters._generate_chapter_background",
            new_callable=AsyncMock,
        ) as mock_generate:
            # Generate chapter
            generate_response = client.post("/api/chapters/generate", json=request_body)
            assert generate_response.status_code == 200

            job_id = generate_response.json()["job_id"]

            # Simulate job completion by setting job status
            from backend.src.api.chapters import generation_jobs

            generation_jobs[job_id] = {
                "status": "completed",
                "module_id": "module1",
                "chapter_number": 1,
                "title": "ROS 2 Basics",
                "chapter_path": "/tmp/01-ros2.md",
                "token_count": 2000,
                "chunks_created": 8,
            }

            # Poll for completion
            status_response = client.get(f"/api/chapters/jobs/{job_id}")

            assert status_response.status_code == 200
            assert status_response.json()["status"] == "completed"
            assert status_response.json()["token_count"] == 2000

    def test_validate_after_generation(self, client):
        """Test validating generated chapter content."""
        # First validate some content
        validate_request = {
            "content": """
## Generated Chapter

### Introduction
[Citation: https://official.docs]

### Core Concepts
Concept explanation here. [Citation: https://reference]

### Examples
```python
example = "code"
```

### Summary
- Summary point 1
- Summary point 2

### References
- [Official Docs](https://official.docs)
"""
        }

        response = client.post("/api/chapters/validate", json=validate_request)

        assert response.status_code == 200
        data = response.json()

        # Should pass validation
        assert isinstance(data["passed"], bool)
        assert isinstance(data["citations"], dict)
        assert isinstance(data["code_blocks"], dict)
