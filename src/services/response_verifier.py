"""Response verification service for validating RAG response quality and context adherence"""
import logging
from typing import List, Dict
import cohere
import re

from src.config import settings
from src.models.rag import RetrievedChunkData

logger = logging.getLogger(__name__)


class ResponseVerifier:
    """Service for verifying that responses are grounded in provided context"""

    def __init__(self):
        """Initialize verifier with Cohere"""
        if settings.COHERE_API_KEY:
            self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        else:
            self.cohere_client = None
        self.embedding_model = "embed-english-v3.0"  # Cohere's embedding model

    def verify_context_only(
        self,
        response_text: str,
        chunks: List[RetrievedChunkData],
        similarity_threshold: float = 0.70
    ) -> Dict:
        """
        Verify that a response is grounded in the provided context.

        Uses semantic similarity matching to check if response sentences relate to context.

        Args:
            response_text: Generated response to verify
            chunks: Context chunks used to generate response
            similarity_threshold: Minimum similarity to consider a match (0-1)

        Returns:
            Dictionary with verification results:
            {
                "verified": bool,
                "overall_similarity": float,
                "total_sentences": int,
                "grounded_sentences": int,
                "non_grounded_sentences": List[str],
                "similarity_scores": List[float],
                "confidence": str ("high", "medium", "low")
            }
        """
        if not response_text or not chunks:
            return {
                "verified": False,
                "overall_similarity": 0.0,
                "total_sentences": 0,
                "grounded_sentences": 0,
                "non_grounded_sentences": [],
                "similarity_scores": [],
                "confidence": "low",
                "reason": "Empty response or context"
            }

        try:
            # Split response into sentences
            sentences = self._split_sentences(response_text)
            if not sentences:
                return {
                    "verified": False,
                    "overall_similarity": 0.0,
                    "total_sentences": 0,
                    "grounded_sentences": 0,
                    "non_grounded_sentences": [],
                    "similarity_scores": [],
                    "confidence": "low",
                    "reason": "No sentences to verify"
                }

            # Embed context chunks
            context_embeddings = []
            for chunk in chunks:
                try:
                    embedding = self._embed_text(chunk.text)
                    context_embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed chunk: {e}")
                    continue

            if not context_embeddings:
                return {
                    "verified": False,
                    "overall_similarity": 0.0,
                    "total_sentences": len(sentences),
                    "grounded_sentences": 0,
                    "non_grounded_sentences": sentences,
                    "similarity_scores": [],
                    "confidence": "low",
                    "reason": "Failed to embed context"
                }

            # Check each sentence against context
            grounded_sentences = []
            non_grounded_sentences = []
            similarity_scores = []

            for sentence in sentences:
                try:
                    sentence_embedding = self._embed_text(sentence)
                    max_similarity = self._compute_max_similarity(
                        sentence_embedding,
                        context_embeddings
                    )
                    similarity_scores.append(max_similarity)

                    if max_similarity >= similarity_threshold:
                        grounded_sentences.append(sentence)
                    else:
                        non_grounded_sentences.append(sentence)

                except Exception as e:
                    logger.warning(f"Failed to verify sentence: {e}")
                    non_grounded_sentences.append(sentence)
                    similarity_scores.append(0.0)

            # Calculate overall statistics
            grounded_count = len(grounded_sentences)
            total_count = len(sentences)
            overall_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

            # Determine confidence level
            if grounded_count == total_count:
                confidence = "high"
            elif grounded_count >= total_count * 0.8:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "verified": grounded_count >= total_count * 0.8,  # At least 80% grounded
                "overall_similarity": round(overall_similarity, 3),
                "total_sentences": total_count,
                "grounded_sentences": grounded_count,
                "non_grounded_sentences": non_grounded_sentences,
                "similarity_scores": [round(s, 3) for s in similarity_scores],
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Response verification failed: {e}")
            return {
                "verified": False,
                "overall_similarity": 0.0,
                "total_sentences": 0,
                "grounded_sentences": 0,
                "non_grounded_sentences": [],
                "similarity_scores": [],
                "confidence": "low",
                "error": str(e)
            }

    def _embed_text(self, text: str) -> List[float]:
        """
        Embed text using Cohere embeddings.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If embedding fails
        """
        try:
            if not self.cohere_client:
                raise ValueError("Cohere client not initialized")

            response = self.cohere_client.embed(
                texts=[text],
                model=self.embedding_model,
                input_type="search_document"  # Required for Cohere V2
            )
            # Access embeddings correctly for Cohere V2 API
            embeddings_list = response.embeddings.float if hasattr(response.embeddings, 'float') else response.embeddings
            return embeddings_list[0]
        except Exception as e:
            logger.error(f"Embedding failed for text: {text[:50]}... - {e}")
            raise ValueError(f"Embedding failed: {str(e)}")

    def _compute_max_similarity(
        self,
        query_embedding: List[float],
        context_embeddings: List[List[float]]
    ) -> float:
        """
        Compute maximum cosine similarity between query and context embeddings.

        Args:
            query_embedding: Query embedding vector
            context_embeddings: List of context embedding vectors

        Returns:
            Maximum similarity score (0-1)
        """
        import numpy as np

        query_vec = np.array(query_embedding)
        max_sim = 0.0

        for context_vec in context_embeddings:
            # Cosine similarity
            similarity = np.dot(query_vec, context_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(context_vec) + 1e-8
            )
            max_sim = max(max_sim, similarity)

        return float(max_sim)

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting by common delimiters
        # A more sophisticated approach would use NLTK or spaCy
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def flag_unsourced_claims(
        self,
        response_text: str,
        chunks: List[RetrievedChunkData]
    ) -> List[Dict]:
        """
        Identify claims in the response that might not be sourced from context.

        Args:
            response_text: Response text to analyze
            chunks: Context chunks

        Returns:
            List of potentially unsourced claims
        """
        sentences = self._split_sentences(response_text)
        unsourced = []

        # Extract keywords from context
        context_keywords = set()
        for chunk in chunks:
            words = chunk.text.lower().split()
            context_keywords.update(w.strip(".,;:!?()[]") for w in words if len(w) > 3)

        # Check each sentence
        for sentence in sentences:
            words = set(w.lower().strip(".,;:!?()[]") for w in sentence.split() if len(w) > 3)
            overlap = len(words & context_keywords)
            overlap_ratio = overlap / len(words) if words else 0

            # Flag sentences with low overlap as potentially unsourced
            if overlap_ratio < 0.3 and len(sentence) > 20:
                unsourced.append({
                    "sentence": sentence,
                    "context_overlap": round(overlap_ratio, 3),
                    "keywords_matched": overlap
                })

        return unsourced

    def get_verification_summary(
        self,
        verification_result: Dict
    ) -> str:
        """
        Generate a human-readable summary of verification results.

        Args:
            verification_result: Result from verify_context_only()

        Returns:
            Summary string
        """
        if "error" in verification_result:
            return f"Verification error: {verification_result['error']}"

        verified = verification_result.get("verified", False)
        confidence = verification_result.get("confidence", "unknown")
        grounded = verification_result.get("grounded_sentences", 0)
        total = verification_result.get("total_sentences", 0)
        similarity = verification_result.get("overall_similarity", 0.0)

        status = "✓ VERIFIED" if verified else "✗ NOT VERIFIED"
        return (
            f"{status} ({confidence} confidence)\n"
            f"Context grounding: {grounded}/{total} sentences ({100*grounded//total if total else 0}%)\n"
            f"Average similarity: {similarity:.3f}"
        )
