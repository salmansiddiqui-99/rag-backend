"""Unit tests for ChatbotService response generation and verification (T060)"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

from src.services.chatbot_service import ChatbotService
from src.models.rag import RetrievedChunkData


@pytest.fixture
def chatbot_service():
    """Create ChatbotService instance with mocked OpenAI client"""
    with patch('src.services.chatbot_service.openai.Client'):
        service = ChatbotService()
        service.openai_client = Mock()
        return service


@pytest.fixture
def sample_chunks():
    """Create sample retrieved chunks for testing"""
    return [
        RetrievedChunkData(
            chunk_id=uuid4(),
            chapter_id=uuid4(),
            section_title="ROS 2 Basics",
            text="ROS 2 is a middleware for robotics development. It provides pub/sub messaging.",
            similarity_score=0.95,
            rank=1
        ),
        RetrievedChunkData(
            chunk_id=uuid4(),
            chapter_id=uuid4(),
            section_title="Gazebo Simulation",
            text="Gazebo is a physics simulator used for testing robotic applications.",
            similarity_score=0.87,
            rank=2
        )
    ]


class TestGenerateResponse:
    """Tests for generate_response method (streaming and non-streaming)"""

    def test_generate_response_with_context(self, chatbot_service, sample_chunks):
        """generate_response should generate response with provided context"""
        mock_stream = Mock()
        mock_stream.text_stream = ["Hello", " from", " ROS"]
        chatbot_service.openai_client.messages.stream.return_value.__enter__ = Mock(return_value=mock_stream)
        chatbot_service.openai_client.messages.stream.return_value.__exit__ = Mock(return_value=False)

        tokens = list(chatbot_service.generate_response(
            query_text="What is ROS 2?",
            chunks=sample_chunks,
            stream=True
        ))

        assert tokens == ["Hello", " from", " ROS"]
        chatbot_service.openai_client.messages.stream.assert_called_once()

    def test_generate_response_empty_context(self, chatbot_service):
        """generate_response should handle empty context gracefully"""
        tokens = list(chatbot_service.generate_response(
            query_text="What is ROS?",
            chunks=[],
            stream=True
        ))

        assert tokens == ["I cannot answer this based on the available content."]

    def test_generate_response_non_streaming(self, chatbot_service, sample_chunks):
        """generate_response with stream=False should return full response"""
        mock_response = Mock()
        mock_response.content = [Mock(text="This is the full response")]
        chatbot_service.openai_client.messages.create.return_value = mock_response

        tokens = list(chatbot_service.generate_response(
            query_text="Explain ROS 2",
            chunks=sample_chunks,
            stream=False
        ))

        assert tokens == ["This is the full response"]
        chatbot_service.openai_client.messages.create.assert_called_once()

    def test_generate_response_system_prompt_enforced(self, chatbot_service, sample_chunks):
        """T055: System prompt should enforce context-only constraints"""
        mock_stream = Mock()
        mock_stream.text_stream = ["Response"]
        chatbot_service.openai_client.messages.stream.return_value.__enter__ = Mock(return_value=mock_stream)
        chatbot_service.openai_client.messages.stream.return_value.__exit__ = Mock(return_value=False)

        chatbot_service.generate_response(
            query_text="What is ROS?",
            chunks=sample_chunks,
            stream=True
        )

        # Check system prompt
        call_args = chatbot_service.openai_client.messages.stream.call_args
        system_prompt = call_args[1]["system"]
        assert "ONLY using the provided textbook context" in system_prompt
        assert "CRITICAL CONSTRAINT" in system_prompt

    def test_generate_response_api_error(self, chatbot_service, sample_chunks):
        """generate_response should raise ValueError on API error"""
        chatbot_service.openai_client.messages.stream.side_effect = Exception("API Error")

        with pytest.raises(ValueError, match="Failed to generate response"):
            list(chatbot_service.generate_response(
                query_text="Test",
                chunks=sample_chunks,
                stream=True
            ))


class TestBuildContext:
    """Tests for _build_context method"""

    def test_build_context_formats_chunks(self, chatbot_service, sample_chunks):
        """_build_context should format chunks with sources"""
        context = chatbot_service._build_context(sample_chunks)

        assert "ROS 2 Basics" in context
        assert "Gazebo Simulation" in context
        assert "Source 1" in context
        assert "Source 2" in context
        assert "0.95" in context  # Similarity score
        assert "0.87" in context

    def test_build_context_empty_list(self, chatbot_service):
        """_build_context with empty list should return empty string"""
        context = chatbot_service._build_context([])
        assert context == ""


class TestValidateResponseContainsContext:
    """Tests for validate_response_contains_context method"""

    def test_validate_response_with_keywords(self, chatbot_service, sample_chunks):
        """Response with context keywords should be validated"""
        response = "ROS 2 is a middleware for robotics with Gazebo simulation."

        is_valid = chatbot_service.validate_response_contains_context(response, sample_chunks)
        assert is_valid is True

    def test_validate_response_without_keywords(self, chatbot_service, sample_chunks):
        """Response without context keywords should fail validation"""
        response = "This is completely unrelated content about agriculture."

        is_valid = chatbot_service.validate_response_contains_context(response, sample_chunks)
        assert is_valid is False

    def test_validate_response_empty_chunks(self, chatbot_service):
        """validate_response_contains_context with empty chunks should return False"""
        is_valid = chatbot_service.validate_response_contains_context("Any response", [])
        assert is_valid is False


class TestCheckContextSufficiency:
    """Tests for check_context_sufficiency method"""

    def test_context_sufficient(self, chatbot_service, sample_chunks):
        """Sufficient context should return True"""
        is_sufficient, reason = chatbot_service.check_context_sufficiency(sample_chunks)

        assert is_sufficient is True
        assert reason == "Context sufficient"

    def test_context_insufficient_no_chunks(self, chatbot_service):
        """No chunks should indicate insufficient context"""
        is_sufficient, reason = chatbot_service.check_context_sufficiency([])

        assert is_sufficient is False
        assert "No context chunks" in reason

    def test_context_insufficient_low_similarity(self, chatbot_service):
        """Low similarity chunks should indicate insufficient context"""
        low_sim_chunks = [
            RetrievedChunkData(
                chunk_id=uuid4(),
                chapter_id=uuid4(),
                section_title="Test",
                text="content",
                similarity_score=0.5,  # Below default threshold of 0.75
                rank=1
            )
        ]

        is_sufficient, reason = chatbot_service.check_context_sufficiency(low_sim_chunks)

        assert is_sufficient is False
        assert "similarity" in reason.lower()


class TestFilterLowConfidenceResponses:
    """Tests for filter_low_confidence_responses method (T056)"""

    def test_filter_response_high_confidence(self, chatbot_service, sample_chunks):
        """Responses with high keyword overlap should pass"""
        response = "ROS 2 middleware and Gazebo simulation are key tools."

        final_response, is_confident = chatbot_service.filter_low_confidence_responses(
            response,
            sample_chunks,
            confidence_threshold=0.5
        )

        assert final_response == response
        assert is_confident is True

    def test_filter_response_low_confidence(self, chatbot_service, sample_chunks):
        """T058: Responses with low keyword overlap should be filtered"""
        response = "The weather today is sunny and pleasant."

        final_response, is_confident = chatbot_service.filter_low_confidence_responses(
            response,
            sample_chunks,
            confidence_threshold=0.5
        )

        assert is_confident is False
        assert "cannot answer" in final_response.lower()

    def test_filter_response_empty_chunks(self, chatbot_service):
        """T058: Should return fallback for empty chunks"""
        final_response, is_confident = chatbot_service.filter_low_confidence_responses(
            "Any response",
            [],
            confidence_threshold=0.5
        )

        assert is_confident is False
        assert "cannot answer" in final_response.lower()

    def test_filter_response_confidence_scoring(self, chatbot_service, mocker):
        """T056: Confidence should be calculated as term overlap ratio"""
        chunks = [
            RetrievedChunkData(
                chunk_id=uuid4(),
                chapter_id=uuid4(),
                section_title="Test",
                text="robot middleware simulation physics",
                similarity_score=0.9,
                rank=1
            )
        ]

        logger_mock = mocker.patch('src.services.chatbot_service.logger')

        # Response with 2 of 4 context terms = 50% confidence
        response = "robot simulation discussed"

        final_response, is_confident = chatbot_service.filter_low_confidence_responses(
            response,
            chunks,
            confidence_threshold=0.5
        )

        # Should log confidence score
        assert logger_mock.debug.called
        call_args = logger_mock.debug.call_args[0][0]
        assert "Response confidence" in call_args


class TestVerifyGroundingInContext:
    """Tests for verify_grounding_in_context method (T057)"""

    def test_verify_grounding_high_keyword_overlap(self, chatbot_service, sample_chunks):
        """T057: Response with high keyword overlap should be verified"""
        response = "ROS 2 middleware provides pub/sub messaging for robotics development."

        verification = chatbot_service.verify_grounding_in_context(response, sample_chunks)

        assert verification["verified"] is True
        assert verification["keyword_score"] > 0.3
        assert verification["has_hallucination_marker"] is False

    def test_verify_grounding_low_keyword_overlap(self, chatbot_service, sample_chunks):
        """Response with low keyword overlap should fail verification"""
        response = "The sunset is beautiful today."

        verification = chatbot_service.verify_grounding_in_context(response, sample_chunks)

        assert verification["verified"] is False
        assert verification["keyword_score"] < 0.3

    def test_verify_grounding_hallucination_markers(self, chatbot_service, sample_chunks):
        """T057: Responses with hallucination markers should fail verification"""
        response = "According to my training data, ROS 2 is a middleware."

        verification = chatbot_service.verify_grounding_in_context(response, sample_chunks)

        assert verification["has_hallucination_marker"] is True
        assert verification["verified"] is False

    def test_verify_grounding_refusal_only(self, chatbot_service, sample_chunks):
        """Pure refusal responses should be detected"""
        response = "I cannot answer this."

        verification = chatbot_service.verify_grounding_in_context(response, sample_chunks)

        assert verification["is_refusal"] is True
        assert verification["verified"] is False

    def test_verify_grounding_empty_response(self, chatbot_service, sample_chunks):
        """Empty response should return unverified"""
        verification = chatbot_service.verify_grounding_in_context("", sample_chunks)
        assert verification["verified"] is False

    def test_verify_grounding_empty_chunks(self, chatbot_service):
        """Empty chunks should return unverified"""
        verification = chatbot_service.verify_grounding_in_context("Any response", [])
        assert verification["verified"] is False


class TestCountTokens:
    """Tests for count_tokens method"""

    def test_count_tokens_approximation(self, chatbot_service):
        """count_tokens should use 4-character approximation"""
        text = "This is a test"  # 14 characters
        tokens = chatbot_service.count_tokens(text)

        assert tokens == 3  # 14 // 4 = 3

    def test_count_tokens_empty(self, chatbot_service):
        """count_tokens with empty string should return 0"""
        tokens = chatbot_service.count_tokens("")
        assert tokens == 0
