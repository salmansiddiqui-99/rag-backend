"""
AI generation service using OpenAI SDK via OpenRouter.
"""
from openai import OpenAI
from typing import Optional, List, Dict
from src.config import config


class GenerationService:
    """Wrapper for OpenRouter AI generation via OpenAI SDK."""

    def __init__(self):
        """Initialize OpenAI client configured for OpenRouter."""
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY
        )
        self.model = config.OPENROUTER_MODEL

    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate AI response using OpenRouter.

        Args:
            prompt: System prompt with context and query
            conversation_history: Previous conversation messages (optional)

        Returns:
            AI-generated response text

        Raises:
            Exception: If OpenRouter API call fails
        """
        messages = []

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current prompt as user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            # Handle rate limits and API errors
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                raise Exception("Rate limit exceeded. Please try again in a moment.")
            else:
                raise Exception(f"AI generation failed: {error_msg}")


# Global singleton instance
generation_service = GenerationService()
