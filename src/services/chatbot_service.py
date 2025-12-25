"""Chatbot service for generating RAG-based responses using LLM adapters"""
import logging
from typing import List, AsyncGenerator, Iterator

from src.config import settings
from src.models.rag import RetrievedChunkData, ResponseStatus
from src.llm import create_llm_adapter, LLMCompletionParams, LLMMessage

logger = logging.getLogger(__name__)


class ChatbotService:
    """Service for generating context-aware responses using LLMs"""

    def __init__(self):
        """Initialize chatbot service with LLM adapter"""
        self._llm_adapter = None
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.temperature = settings.OPENAI_TEMPERATURE

    @property
    def llm_adapter(self):
        """Lazily initialize LLM adapter on first access."""
        if self._llm_adapter is None:
            self._llm_adapter = create_llm_adapter(settings)
            logger.info(
                f"Initialized LLM adapter: provider={self._llm_adapter.provider_name}, "
                f"model={self._llm_adapter.model_name}"
            )
        return self._llm_adapter

    def generate_response(
        self,
        query_text: str,
        chunks: List[RetrievedChunkData],
        stream: bool = True
    ) -> Iterator[str]:
        """
        Generate a response using retrieved chunks as context.

        The response is strictly constrained to use only the provided context.
        If no relevant context is found, the chatbot will decline to answer.

        Args:
            query_text: The user's question
            chunks: List of retrieved content chunks to use as context
            stream: Whether to stream tokens or return full response

        Yields:
            Token strings (if stream=True) or single response string (if stream=False)

        Raises:
            ValueError: If context is insufficient or generation fails
        """
        # Build context from chunks
        context = self._build_context(chunks)

        if not context.strip():
            logger.warning(f"No context available for query: {query_text}")
            yield "I cannot answer this based on the available content."
            return

        # T055: Build strict system prompt to prevent hallucination
        system_prompt = (
            "You are an assistant for a Physical AI & Humanoid Robotics textbook. "
            "CRITICAL CONSTRAINT: Answer ONLY using the provided textbook context. "
            "Do NOT use general knowledge, external sources, or training data. "
            "\n"
            "Rules:\n"
            "1. Base all answers exclusively on the context provided.\n"
            "2. If context doesn't contain the answer, respond: 'I cannot answer this based on the available content.'\n"
            "3. Never assume or infer beyond what is explicitly stated in the context.\n"
            "4. If uncertain, say you cannot answer from available content.\n"
            "5. Cite section titles when relevant.\n"
        )

        user_message = f"""Context from the textbook:
{context}

---

Student Question: {query_text}

Please answer the student's question using ONLY the provided context above.
If the context doesn't contain relevant information, say you cannot answer based on available content."""

        try:
            # Create completion parameters
            params = LLMCompletionParams(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_message)
                ],
                model=self.llm_adapter.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )

            if stream:
                # Stream response using adapter
                for chunk in self.llm_adapter.create_completion_stream(params):
                    # Extract text from OpenAI-format chunk
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
            else:
                # Generate full response at once
                response = self.llm_adapter.create_completion(params)
                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].get("message", {})
                    yield message.get("content", "")

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")

    def _build_context(self, chunks: List[RetrievedChunkData]) -> str:
        """
        Build a context string from retrieved chunks.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted context string for the prompt
        """
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.section_title} (Similarity: {chunk.similarity_score:.2f})]"
                f"\n{chunk.text}\n"
            )

        return "\n".join(context_parts)

    def validate_response_contains_context(
        self,
        response_text: str,
        chunks: List[RetrievedChunkData]
    ) -> bool:
        """
        Simple heuristic check that response relates to provided context.

        This is a basic validation to ensure the response isn't purely from general knowledge.
        A more sophisticated approach would use embedding similarity.

        Args:
            response_text: Generated response text
            chunks: Chunks used for context

        Returns:
            True if response appears to use the context, False otherwise
        """
        if not chunks or not response_text:
            return False

        # Extract key terms from chunks (simple keyword approach)
        context_keywords = set()
        for chunk in chunks:
            # Get first few significant words from each chunk
            words = chunk.text.lower().split()
            for word in words[:20]:  # First 20 words per chunk
                if len(word) > 4:  # Skip short words
                    context_keywords.add(word.strip(".,;:!?"))

        # Check if response uses context keywords
        response_lower = response_text.lower()
        keyword_matches = sum(1 for kw in context_keywords if kw in response_lower)

        # Need at least some keyword overlap to consider it context-based
        return keyword_matches > 0

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses the LLM adapter's token counting method if available,
        otherwise falls back to simple heuristic.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        try:
            return self.llm_adapter.count_tokens(text)
        except Exception:
            # Fallback: Simple approximation (1 token ~ 4 characters)
            return len(text) // 4

    def check_context_sufficiency(
        self,
        chunks: List[RetrievedChunkData],
        min_similarity: float = 0.75
    ) -> tuple[bool, str]:
        """
        Check if retrieved context is sufficient to answer a question.

        Args:
            chunks: Retrieved chunks
            min_similarity: Minimum similarity score to consider relevant

        Returns:
            Tuple of (is_sufficient, reason_if_not)
        """
        if not chunks:
            return False, "No context chunks retrieved"

        # Check if we have at least some relevant chunks
        relevant_chunks = [c for c in chunks if c.similarity_score >= min_similarity]

        if not relevant_chunks:
            return False, f"No chunks with similarity > {min_similarity}"

        # Check total context size
        total_tokens = sum(self.count_tokens(c.text) for c in relevant_chunks)
        if total_tokens < 100:
            return False, f"Retrieved context too small ({total_tokens} tokens)"

        return True, "Context sufficient"

    def filter_low_confidence_responses(
        self,
        response_text: str,
        chunks: List[RetrievedChunkData],
        confidence_threshold: float = 0.5
    ) -> tuple[str, bool]:
        """
        T056: Filter responses that appear to lack grounding in context.

        Checks if response contains key concepts from retrieved context.
        If confidence is low, returns fallback message.

        Args:
            response_text: Generated response
            chunks: Chunks used for context
            confidence_threshold: Confidence threshold (0-1)

        Returns:
            Tuple of (final_response, is_confident)
        """
        if not chunks or not response_text:
            return "I cannot answer this based on the available content.", False

        # Extract important terms from context (nouns, technical terms)
        context_terms = set()
        for chunk in chunks:
            # Get significant words (length > 5 chars, lowercase)
            words = chunk.text.lower().split()
            for word in words:
                cleaned = word.strip(".,;:!?()")
                if len(cleaned) > 5:
                    context_terms.add(cleaned)

        # Check response alignment with context
        response_lower = response_text.lower()
        matching_terms = sum(1 for term in context_terms if term in response_lower)

        # T056: Calculate confidence score
        if context_terms:
            confidence = matching_terms / len(context_terms)
        else:
            confidence = 0.0

        logger.debug(
            f"Response confidence: {confidence:.2f} "
            f"({matching_terms}/{len(context_terms)} terms matched)"
        )

        # T058: Return fallback if confidence too low
        if confidence < confidence_threshold:
            logger.warning(f"Low confidence response (score: {confidence:.2f})")
            return "I cannot answer this based on the available content.", False

        return response_text, True

    def verify_grounding_in_context(
        self,
        response_text: str,
        chunks: List[RetrievedChunkData]
    ) -> dict:
        """
        T057: Verify response is grounded in provided context.

        Uses multiple heuristics to detect hallucinations:
        1. Keyword overlap with context
        2. Absence of common hallucination markers
        3. Response length consistency

        Args:
            response_text: Generated response
            chunks: Chunks used for context

        Returns:
            Dictionary with verification results
        """
        if not chunks or not response_text:
            return {"verified": False, "reason": "No context or response"}

        # Check 1: Keyword overlap
        context_keywords = set()
        for chunk in chunks:
            words = chunk.text.lower().split()
            for word in words[:30]:  # Sample first 30 words per chunk
                if len(word.strip(".,;:!?()")) > 4:
                    context_keywords.add(word.strip(".,;:!?()"))

        response_lower = response_text.lower()
        keyword_matches = sum(1 for kw in context_keywords if kw in response_lower)
        keyword_score = keyword_matches / len(context_keywords) if context_keywords else 0

        # Check 2: Hallucination markers
        hallucination_phrases = [
            "i don't have information",
            "i'm not sure",
            "my knowledge cutoff",
            "according to my training",
            "in my experience",
        ]
        has_hallucination_marker = any(phrase in response_lower for phrase in hallucination_phrases)

        # Check 3: Response is not just "I cannot answer"
        is_refusal = "cannot answer" in response_lower and len(response_text) < 100

        verified = keyword_score > 0.3 and not has_hallucination_marker and not is_refusal

        logger.info(
            f"Grounding verification: "
            f"keyword_score={keyword_score:.2f}, "
            f"has_marker={has_hallucination_marker}, "
            f"verified={verified}"
        )

        return {
            "verified": verified,
            "keyword_score": keyword_score,
            "has_hallucination_marker": has_hallucination_marker,
            "is_refusal": is_refusal
        }
